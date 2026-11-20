use std::sync::OnceLock;

use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_types::FpExtEncoding;
#[cfg(akita_packing_sme)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field};

use super::fixed_fold_ntt::RING_D;
use crate::arithmetic::coefficient_packing::FusedPackingWeights;
use crate::opaque::SubringCoefficientPackingPlan;

const Q: u64 = u64::MAX - 58;
const STRIDE: usize = 16;
#[cfg(akita_packing_sme)]
const STRIPE: usize = 16;
const OUTPUT: usize = 2 * (RING_D / STRIDE);
const MAX_WORDS: usize = 1 << 30;
#[cfg(akita_packing_sme)]
const MAX_UNCACHED_WORDS: usize = 3 << 26;

#[cfg(akita_packing_sme)]
type BlockKernel = unsafe extern "C" fn(*const u64, *const u64, usize, *mut u64) -> std::ffi::c_int;

#[cfg(akita_packing_sme)]
unsafe extern "C" {
    fn akita_packing_sme_available() -> std::ffi::c_int;
    fn akita_packing_sme_striped(
        source: *const u64,
        weights: *const u64,
        terms: usize,
        output: *mut u64,
    ) -> std::ffi::c_int;
    fn akita_packing_sme_source(
        source: *const u64,
        weights: *const u64,
        terms: usize,
        output: *mut u64,
    ) -> std::ffi::c_int;
}

#[cfg(akita_packing_sme)]
fn available() -> bool {
    static SUPPORT: OnceLock<bool> = OnceLock::new();
    // SAFETY: The platform query accesses no Rust memory.
    *SUPPORT.get_or_init(|| unsafe { akita_packing_sme_available() != 0 })
}

#[cfg(not(akita_packing_sme))]
fn available() -> bool {
    false
}

pub(super) struct PackingStripes {
    positions: usize,
    source_words: usize,
    #[cfg(akita_packing_sme)]
    words: OnceLock<Option<Vec<u64>>>,
    #[cfg(akita_fold_metal)]
    metal: OnceLock<Option<PackingKey>>,
}

#[cfg(akita_fold_metal)]
struct PackingKey {
    handle: std::ptr::NonNull<std::ffi::c_void>,
    owner: std::sync::Arc<Vec<u64>>,
}
#[cfg(akita_fold_metal)]
// SAFETY: Metal permits concurrent buffer reads, and the retained host allocation is immutable.
unsafe impl Send for PackingKey {}
#[cfg(akita_fold_metal)]
// SAFETY: Each dispatch has separate output and scratch buffers.
unsafe impl Sync for PackingKey {}
#[cfg(akita_fold_metal)]
impl Drop for PackingKey {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn akita_metal_fold_release(handle: *mut std::ffi::c_void);
        }
        // SAFETY: Release the retained Metal reference before Rust drops its backing allocation.
        unsafe { akita_metal_fold_release(self.handle.as_ptr()) }
    }
}

impl PackingStripes {
    pub(super) fn new<F: Field + CanonicalEncoding>(
        rings: &[CyclotomicRing<F, RING_D>],
        positions: usize,
    ) -> Option<Self> {
        if F::CHALLENGE_SET_LIMBS != [Q, 0, 0, 0]
            || !(64..=1024).contains(&positions)
            || !rings.len().is_multiple_of(positions)
            || rings.is_empty()
            || F::canonical_u64_slice(rings[0].coefficients()).is_none()
        {
            return None;
        }
        let len = rings
            .len()
            .checked_mul(RING_D)
            .filter(|&n| n <= MAX_WORDS)?;
        Some(Self {
            positions,
            source_words: len,
            #[cfg(akita_packing_sme)]
            words: OnceLock::new(),
            #[cfg(akita_fold_metal)]
            metal: OnceLock::new(),
        })
    }

    #[cfg(akita_fold_metal)]
    fn metal_key(&self, owner: &std::sync::Arc<Vec<u64>>) -> Option<&PackingKey> {
        if self.source_words > owner.len() {
            return None;
        }
        unsafe extern "C" {
            fn akita_metal_pack_key(
                shader: *const std::ffi::c_char,
                source: *const u64,
                bytes: usize,
            ) -> *mut std::ffi::c_void;
        }
        const SHADER: &str = concat!(include_str!("packing.metal"), "\0");
        self.metal
            .get_or_init(|| {
                // SAFETY: The handle retains the immutable allocation until Metal releases it.
                let ptr = unsafe {
                    akita_metal_pack_key(
                        SHADER.as_ptr().cast(),
                        owner.as_ptr(),
                        self.source_words * std::mem::size_of::<u64>(),
                    )
                };
                std::ptr::NonNull::new(ptr).map(|handle| PackingKey {
                    handle,
                    owner: owner.clone(),
                })
            })
            .as_ref()
            .filter(|key| std::sync::Arc::ptr_eq(&key.owner, owner))
    }

    #[cfg(akita_fold_metal)]
    pub(super) fn weighted_rows(
        &self,
        owner: &std::sync::Arc<Vec<u64>>,
        weights: &[[u64; 2]],
    ) -> Option<Vec<[u64; 2]>> {
        if self.source_words != owner.len()
            || !(1..=64).contains(&weights.len())
            || !owner.len().is_multiple_of(weights.len())
            || weights.iter().flatten().any(|&w| w >= Q)
        {
            return None;
        }
        let cells = owner.len() / weights.len();
        if cells < 65536 {
            return None;
        }
        let key = self.metal_key(owner)?;
        let mut output = Vec::<[u64; 2]>::new();
        output.try_reserve_exact(cells).ok()?;
        unsafe extern "C" {
            fn akita_metal_weighted_rows(
                handle: *mut std::ffi::c_void,
                cells: u32,
                rows: u32,
                weights: *const [u64; 2],
                output: *mut [u64; 2],
            ) -> i32;
        }
        // SAFETY: The key contains rows * cells words. The command writes both words of every output before returning success.
        unsafe {
            if akita_metal_weighted_rows(
                key.handle.as_ptr(),
                cells as u32,
                weights.len() as u32,
                weights.as_ptr(),
                output.as_mut_ptr(),
            ) == 0
            {
                return None;
            }
            output.set_len(cells);
        }
        Some(output)
    }

    #[cfg(akita_packing_sme)]
    fn transpose(&self, source: &[u64]) -> Option<Vec<u64>> {
        let positions = self.positions;
        let len = source.len();
        let block_len = positions * RING_D;
        let uncached_blocks = (len / block_len / 3).min(MAX_UNCACHED_WORDS / block_len);
        let cached_len = len - uncached_blocks * block_len;
        let mut words = Vec::new();
        words.try_reserve_exact(cached_len).ok()?;
        words.resize(cached_len, 0);
        cfg_chunks_mut!(words, block_len)
            .enumerate()
            .for_each(|(block, out)| {
                let block_source = &source[block * block_len..][..block_len];
                for (position, source) in block_source.chunks_exact(RING_D).enumerate() {
                    for (tile, coordinates) in source.chunks_exact(STRIPE * STRIDE).enumerate() {
                        let start = (tile * positions + position) * STRIPE * STRIDE;
                        for (residue, values) in out[start..][..STRIPE * STRIDE]
                            .chunks_exact_mut(STRIPE)
                            .enumerate()
                        {
                            for (slot, coordinate) in
                                values.iter_mut().zip(coordinates.chunks_exact(STRIDE))
                            {
                                *slot = coordinate[residue];
                            }
                        }
                    }
                }
            });
        Some(words)
    }

    pub(super) fn contract<F, E, const D: usize>(
        &self,
        source_vars: usize,
        source: &[F],
        weights: &FusedPackingWeights<'_, E>,
        _metal_owner: Option<&std::sync::Arc<Vec<u64>>>,
    ) -> Result<Option<Vec<F>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + FpExtEncoding<F>,
    {
        let tables = weights.tables();
        let point = tables.point();
        let geometry = point.geometry();
        let full_positions = self.source_words / RING_D;
        if D != RING_D
            || F::CHALLENGE_SET_LIMBS != [Q, 0, 0, 0]
            || E::DEGREE != 2
            || geometry.extension_degree() != 2
            || geometry.subring_embedding_stride() != STRIDE
            || geometry.partial_base_field_width() != OUTPUT
            || point.num_positions_per_block() != self.positions
            || source.len() != self.source_words
            || point.num_live_positions() < full_positions
            || weights.values().len() != self.positions * STRIDE
        {
            return Ok(None);
        }
        SubringCoefficientPackingPlan { tables }.validate::<D>(source_vars)?;
        if _metal_owner.is_none() && !available() {
            return Ok(None);
        }
        let active_blocks = full_positions / self.positions;
        let len = checked::product([point.num_live_blocks(), OUTPUT]).ok_or_else(|| {
            AkitaError::InvalidInput("coefficient-packing output length overflow".into())
        })?;
        let Some(active_len) = active_blocks.checked_mul(OUTPUT).filter(|&n| n <= len) else {
            return Ok(None);
        };

        let mut canonical_weights = Vec::<[u64; 2]>::new();
        canonical_weights
            .try_reserve_exact(weights.values().len())
            .map_err(|_| {
                AkitaError::InvalidInput("coefficient-packing weight allocation failed".into())
            })?;
        for value in weights.values() {
            let [real, imaginary] = value.ext_coords() else {
                return Ok(None);
            };
            let (Some(real), Some(imaginary)) = (real.to_u64_checked(), imaginary.to_u64_checked())
            else {
                return Ok(None);
            };
            canonical_weights.push([real, imaginary]);
        }
        let Some(raw) = F::canonical_u64_slice(source) else {
            return Ok(None);
        };
        let mut out = Vec::new();
        out.try_reserve_exact(len).map_err(|_| {
            AkitaError::InvalidInput("coefficient-packing output allocation failed".into())
        })?;
        out.resize(len, F::zero());
        #[cfg(akita_fold_metal)]
        if let Some(owner) = _metal_owner.filter(|owner| {
            active_blocks >= 8
                && self.positions.is_multiple_of(64)
                && owner.as_ptr() == raw.as_ptr()
        }) {
            unsafe extern "C" {
                fn akita_metal_pack(
                    handle: *mut std::ffi::c_void,
                    weights: *const u64,
                    positions: u32,
                    blocks: u32,
                    output: *mut u64,
                ) -> std::ffi::c_int;
            }
            if let Some(key) = self.metal_key(owner) {
                let mut coordinates = vec![0u64; active_len];
                // SAFETY: The checked shape bounds the source, weights and output for the synchronous dispatch.
                let completed = unsafe {
                    akita_metal_pack(
                        key.handle.as_ptr(),
                        canonical_weights.as_ptr().cast(),
                        self.positions as u32,
                        active_blocks as u32,
                        coordinates.as_mut_ptr(),
                    )
                };
                if completed != 0 {
                    for (value, word) in out.iter_mut().zip(coordinates) {
                        *value = F::from_u64(word);
                    }
                    return Ok(Some(out));
                }
            }
        }
        #[cfg(not(akita_packing_sme))]
        {
            Ok(None)
        }
        #[cfg(akita_packing_sme)]
        {
            if !available() {
                return Ok(None);
            }
            let Some(words) = self.words.get_or_init(|| self.transpose(raw)).as_ref() else {
                return Ok(None);
            };
            let words_per_block = self.positions * RING_D;
            let complete =
                cfg_chunks_mut!(out[..active_len], OUTPUT)
                    .enumerate()
                    .all(|(block, output)| {
                        let mut partial = [0u64; OUTPUT];
                        let range = block * words_per_block..(block + 1) * words_per_block;
                        let (words, contract): (&[u64], BlockKernel) =
                            match words.get(range.clone()) {
                                Some(packed) => (packed, akita_packing_sme_striped),
                                None => (&raw[range], akita_packing_sme_source),
                            };
                        // SAFETY: Both kernels read one full block and its weight pairs
                        // and write OUTPUT words. The runtime checks SME support.
                        let ok = unsafe {
                            contract(
                                words.as_ptr(),
                                canonical_weights.as_ptr().cast(),
                                self.positions * STRIDE,
                                partial.as_mut_ptr(),
                            )
                        };
                        for (output, raw) in output.iter_mut().zip(partial) {
                            *output = F::from_u64(raw);
                        }
                        ok != 0
                    });
            Ok(complete.then_some(out))
        }
    }
}

#[cfg(test)]
#[path = "packing_sme/tests/stripes.rs"]
mod tests;
