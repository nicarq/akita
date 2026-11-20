//! Frequency-major cache for an immutable source with a subring fold challenge.

use crate::sources::poly_helpers::ValidatedSparseChallenges;
use akita_algebra::ntt::butterfly::{forward_ntt, inverse_ntt};
use akita_algebra::ntt::tables::Q64_PRIMES;
use akita_algebra::ntt::{MontCoeff, NttKernelPlan, NttPrime, NttTwiddles};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::{checked, AkitaError};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::ops::Range;

pub(super) const RING_D: usize = 2048;
pub(super) const DIGITS: usize = 4;
pub(super) const LOG_BASIS: u32 = 16;
const STRIDE: usize = 16;
const SUBRING_D: usize = RING_D / STRIDE;
const PRIME: NttPrime<i32> = Q64_PRIMES[0];
const P: i64 = PRIME.p as i64;
const DOT_BATCH: usize = 32;

type Dot = fn(&[[i32; STRIDE]], &[i32]) -> [i32; STRIDE];

pub(super) struct FixedFoldNtt {
    positions: usize,
    live_rings: usize,
    blocks: usize,
    #[cfg(akita_fold_metal)]
    metal: std::sync::OnceLock<Option<MetalKey>>,
    // [position][digit][frequency][block][residue]
    spectra: Vec<[i32; STRIDE]>,
    twiddles: NttTwiddles<i32, SUBRING_D>,
    ntt: NttKernelPlan,
    dot: Dot,
    #[cfg(any(akita_packing_sme, akita_fold_metal))]
    pub(super) packing: Option<super::packing_sme::PackingStripes>,
}

#[cfg(akita_fold_metal)]
struct MetalKey(std::ptr::NonNull<std::ffi::c_void>);
#[cfg(akita_fold_metal)]
// SAFETY: Metal buffers are thread safe and this retained key is read only.
unsafe impl Send for MetalKey {}
#[cfg(akita_fold_metal)]
// SAFETY: All commands borrow the immutable key, and each has a distinct output.
unsafe impl Sync for MetalKey {}
#[cfg(akita_fold_metal)]
impl Drop for MetalKey {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn akita_metal_fold_release(key: *mut std::ffi::c_void);
        }
        // SAFETY: The retained buffer is released once before its host storage drops.
        unsafe { akita_metal_fold_release(self.0.as_ptr()) }
    }
}

impl std::fmt::Debug for FixedFoldNtt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedFoldNtt")
            .field("positions", &self.positions)
            .field("live_rings", &self.live_rings)
            .finish_non_exhaustive()
    }
}

fn zero_vec<T: Copy>(len: usize, value: T) -> Result<Vec<T>, AkitaError> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(|_| {
        AkitaError::InvalidInput(format!("fixed fold NTT could not reserve {len} elements"))
    })?;
    data.resize(len, value);
    Ok(data)
}

impl FixedFoldNtt {
    pub(super) fn new<F: Field + CanonicalEncoding>(
        rings: &[CyclotomicRing<F, RING_D>],
        positions: usize,
    ) -> Result<Self, AkitaError> {
        if positions == 0 || F::MODULUS_BITS > DIGITS as u32 * LOG_BASIS {
            return Err(AkitaError::InvalidInput(
                "fixed fold NTT requires positions and a field of at most 64 bits".into(),
            ));
        }
        let blocks = rings.len().div_ceil(positions);
        let spectra_len =
            checked::product([positions, DIGITS, SUBRING_D, blocks]).ok_or_else(|| {
                AkitaError::InvalidInput("fixed fold NTT cache size overflows usize".into())
            })?;
        let mut spectra = zero_vec(spectra_len, [0i32; STRIDE])?;
        let ntt = NttKernelPlan::detect::<i32>();
        let twiddles = NttTwiddles::compute(PRIME);
        let params = BalancedDecomposePow2Params::new(DIGITS, LOG_BASIS);
        if blocks != 0 {
            cfg_chunks_mut!(spectra, DIGITS * SUBRING_D * blocks)
                .enumerate()
                .for_each(|(position, spectra)| {
                    let mut digits = [[0i16; RING_D]; DIGITS];
                    for (block, ring) in rings.iter().skip(position).step_by(positions).enumerate()
                    {
                        ring.balanced_decompose_pow2_i16_into(&mut digits, &params);
                        for (digit, source) in digits.iter().enumerate() {
                            for residue in 0..STRIDE {
                                let mut transformed: [MontCoeff<i32>; SUBRING_D] =
                                    std::array::from_fn(|i| {
                                        PRIME
                                            .from_canonical(i32::from(source[i * STRIDE + residue]))
                                    });
                                forward_ntt(&mut transformed, PRIME, &twiddles, ntt);
                                for (frequency, value) in transformed.into_iter().enumerate() {
                                    spectra[(digit * SUBRING_D + frequency) * blocks + block]
                                        [residue] = PRIME.center(value.raw());
                                }
                            }
                        }
                    }
                });
        }
        let dot: Dot = dot_scalar;
        #[cfg(target_arch = "aarch64")]
        let dot = if akita_algebra::ntt::neon::use_neon_ntt() {
            dot_neon
        } else {
            dot
        };
        Ok(Self {
            positions,
            live_rings: rings.len(),
            blocks,
            #[cfg(akita_fold_metal)]
            metal: std::sync::OnceLock::new(),
            spectra,
            twiddles,
            ntt,
            dot,
            #[cfg(any(akita_packing_sme, akita_fold_metal))]
            packing: super::packing_sme::PackingStripes::new(rings, positions),
        })
    }

    #[cfg(akita_fold_metal)]
    fn metal_fold<const D: usize>(
        &self,
        range: &Range<usize>,
        weights: &[i32],
        output: &mut [[i32; D]],
    ) -> bool {
        #[repr(C)]
        struct Shape {
            blocks: u32,
            begin: u32,
            end: u32,
            rows: u32,
            prime: [i32; 4],
        }
        unsafe extern "C" {
            fn akita_metal_fold_key(
                source: *const std::ffi::c_char,
                keys: *const i32,
                bytes: usize,
            ) -> *mut std::ffi::c_void;
            fn akita_metal_fold(
                key: *mut std::ffi::c_void,
                shape: *const Shape,
                weights: *const i32,
                tables: *const i32,
                out: *mut i32,
            ) -> i32;
        }
        let Ok(blocks) = u32::try_from(self.blocks) else {
            return false;
        };
        let Ok(rows) = u32::try_from(output.len()) else {
            return false;
        };
        if D != RING_D
            || rows > u32::MAX / 4096
            || blocks > u32::MAX / 128
            || range.end > self.blocks
            || range.start > range.end
        {
            return false;
        }
        let shape = Shape {
            blocks,
            begin: range.start as u32,
            end: range.end as u32,
            rows,
            prime: [PRIME.p, PRIME.pinv, PRIME.mont, PRIME.montsq],
        };
        let tables = self
            .twiddles
            .negacyclic_tables()
            .map(|table| table.map(MontCoeff::raw));
        let weights: Vec<_> = weights
            .iter()
            .map(|&x| PRIME.from_canonical(x).raw())
            .collect();
        const SOURCE: &str = concat!(include_str!("fixed_fold.metal"), "\0");
        let Some(key) = self.metal.get_or_init(|| {
            // SAFETY: The key allocation is immutable and belongs to this struct for the handle's lifetime.
            let handle = unsafe {
                akita_metal_fold_key(
                    SOURCE.as_ptr().cast(),
                    self.spectra.as_ptr().cast(),
                    std::mem::size_of_val(self.spectra.as_slice()),
                )
            };
            std::ptr::NonNull::new(handle).map(MetalKey)
        }) else {
            return false;
        };
        // SAFETY: Each input is live until the synchronous dispatch completes. The output holds rows * D words and has an exclusive borrow.
        unsafe {
            akita_metal_fold(
                key.0.as_ptr(),
                &shape,
                weights.as_ptr(),
                tables.as_ptr().cast(),
                output.as_mut_ptr().cast(),
            ) != 0
        }
    }

    fn fill_cpu<const D: usize>(
        &self,
        blocks: &Range<usize>,
        spectrum_challenges: &[i32],
        part: &mut [[i32; D]],
    ) {
        cfg_chunks_mut!(part, 1)
            .enumerate()
            .for_each(|(destination, row)| {
                let mut spectrum = [[0i32; STRIDE]; SUBRING_D];
                for (frequency, spectrum) in spectrum.iter_mut().enumerate() {
                    let key = (destination * SUBRING_D + frequency) * self.blocks;
                    let challenge = frequency * self.blocks;
                    *spectrum = (self.dot)(
                        &self.spectra[key + blocks.start..key + blocks.end],
                        &spectrum_challenges[challenge + blocks.start..challenge + blocks.end],
                    );
                }
                for residue in 0..STRIDE {
                    let mut values: [MontCoeff<i32>; SUBRING_D] =
                        std::array::from_fn(|frequency| {
                            MontCoeff::from_raw(spectrum[frequency][residue])
                        });
                    inverse_ntt(&mut values, PRIME, &self.twiddles, self.ntt);
                    for (i, value) in values.into_iter().enumerate() {
                        row[0][i * STRIDE + residue] = PRIME.center(PRIME.to_canonical(value));
                    }
                }
            });
    }

    pub(super) fn positions(&self) -> usize {
        self.positions
    }

    pub(super) fn fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        chunks: &[Range<usize>],
        metal_kernels: bool,
    ) -> Result<Option<Vec<Vec<[i32; D]>>>, AkitaError> {
        #[cfg(not(akita_fold_metal))]
        let _ = metal_kernels;
        if D != RING_D {
            return Ok(None);
        }
        let active = self.blocks.min(challenges.len());
        let mut active_chunks = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let Some(selected) = challenges.get(chunk.clone()) else {
                return Ok(None);
            };
            let first_ring = chunk
                .start
                .saturating_mul(self.positions)
                .min(self.live_rings);
            let last_ring = chunk
                .end
                .saturating_mul(self.positions)
                .min(self.live_rings);
            ValidatedSparseChallenges::<D>::new(selected, last_ring - first_ring, self.positions)?;
            let range = chunk.start.min(active)..chunk.end.min(active);
            let mut mass = 0i64;
            for challenge in &challenges[range.clone()] {
                for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                    if !(position as usize).is_multiple_of(STRIDE) {
                        return Ok(None);
                    }
                    mass += i64::from(coefficient.unsigned_abs());
                    // This bound is for the signed integer sum of every chunk,
                    // before the response's norm and acceptance checks.
                    if mass * 32768 > P / 2 {
                        return Ok(None);
                    }
                }
            }
            active_chunks.push(range);
        }
        let mut spectrum_challenges = zero_vec(
            checked::product([SUBRING_D, self.blocks]).ok_or_else(|| {
                AkitaError::InvalidInput("fixed fold challenge size overflows usize".into())
            })?,
            0,
        )?;
        let mut used = zero_vec(active, false)?;
        for range in &active_chunks {
            used[range.clone()].fill(true);
        }
        for (block, (challenge, &used)) in challenges.iter().zip(&used).enumerate() {
            if !used {
                continue;
            }
            let mut transformed = [MontCoeff::default(); SUBRING_D];
            for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                let index = position as usize / STRIDE;
                let value = PRIME.from_canonical(i32::from(coefficient));
                transformed[index] =
                    PRIME.reduce_range(MontCoeff::from_raw(transformed[index].raw() + value.raw()));
            }
            forward_ntt(&mut transformed, PRIME, &self.twiddles, self.ntt);
            for (frequency, value) in transformed.into_iter().enumerate() {
                // A canonical challenge preserves the key's Montgomery domain
                // in the widened pointwise integer product.
                spectrum_challenges[frequency * self.blocks + block] =
                    PRIME.center(PRIME.to_canonical(value));
            }
        }

        let width = checked::product([self.positions, DIGITS]).ok_or_else(|| {
            AkitaError::InvalidInput("fixed fold output size overflows usize".into())
        })?;
        let mut rows = Vec::with_capacity(chunks.len());
        for blocks in active_chunks {
            let mut part = zero_vec(width, [0i32; D])?;
            #[cfg(akita_fold_metal)]
            if metal_kernels
                && !blocks.is_empty()
                && self.metal_fold(&blocks, &spectrum_challenges, &mut part)
            {
                rows.push(part);
                continue;
            }
            if !blocks.is_empty() {
                self.fill_cpu(&blocks, &spectrum_challenges, &mut part);
            }
            rows.push(part);
        }
        Ok(Some(rows))
    }
}

// Both operands are centered residues. Even 32 products are strictly below
// 2^63 in absolute value because p < 2^30.
fn dot_scalar(key: &[[i32; STRIDE]], challenge: &[i32]) -> [i32; STRIDE] {
    let mut total = [0i64; STRIDE];
    for (key, challenge) in key.chunks(DOT_BATCH).zip(challenge.chunks(DOT_BATCH)) {
        let mut tile = [0i64; STRIDE];
        for (row, &weight) in key.iter().zip(challenge) {
            for (sum, &value) in tile.iter_mut().zip(row) {
                *sum += i64::from(value) * i64::from(weight);
            }
        }
        for (sum, contribution) in total.iter_mut().zip(tile) {
            *sum += contribution % P;
        }
    }
    total.map(|v| (v % P) as i32)
}

#[cfg(target_arch = "aarch64")]
fn dot_neon(key: &[[i32; STRIDE]], challenge: &[i32]) -> [i32; STRIDE] {
    use std::arch::aarch64::*;

    let mut total = [0i128; STRIDE];
    // SAFETY: the caller selected NEON and every vector load stays in a
    // full sixteen-residue row. Each batch satisfies the scalar i64 bound.
    unsafe {
        for (key, challenge) in key.chunks(DOT_BATCH).zip(challenge.chunks(DOT_BATCH)) {
            let mut sums = [vdupq_n_s64(0); STRIDE / 2];
            for (row, &weight) in key.iter().zip(challenge) {
                let weight = vdupq_n_s32(weight);
                for i in 0..STRIDE / 4 {
                    let value = vld1q_s32(row.as_ptr().add(4 * i));
                    sums[2 * i] = vmlal_s32(sums[2 * i], vget_low_s32(value), vget_low_s32(weight));
                    sums[2 * i + 1] = vmlal_high_s32(sums[2 * i + 1], value, weight);
                }
            }
            let mut lanes = [0i64; STRIDE];
            for (index, sum) in sums.into_iter().enumerate() {
                vst1q_s64(lanes.as_mut_ptr().add(2 * index), sum);
            }
            for (sum, value) in total.iter_mut().zip(lanes) {
                *sum += i128::from(value);
            }
        }
    }
    const RADIX_MOD_P: i64 = ((1u128 << 64) % (P as u128)) as i64;
    total.map(|v| {
        let high = (v >> 64) as i64;
        let low = v as u64;
        ((high * RADIX_MOD_P + (low % P as u64) as i64) % P) as i32
    })
}

#[cfg(all(test, akita_fold_metal))]
#[path = "../../../tests/unit/metal_fixed_fold.rs"]
mod metal_tests;
