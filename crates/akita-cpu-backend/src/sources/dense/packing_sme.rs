use std::sync::OnceLock;

use akita_algebra::CyclotomicRing;
use akita_error::{checked, AkitaError};
use akita_types::FpExtEncoding;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field};

use super::fixed_fold_ntt::RING_D;
use crate::arithmetic::coefficient_packing::FusedPackingWeights;
use crate::opaque::SubringCoefficientPackingPlan;

const Q: u64 = u64::MAX - 58;
const STRIDE: usize = 16;
const STRIPE: usize = 16;
const OUTPUT: usize = 2 * (RING_D / STRIDE);
const MAX_WORDS: usize = 1 << 30;

unsafe extern "C" {
    fn akita_packing_sme_available() -> std::ffi::c_int;
    fn akita_packing_sme_striped(
        source: *const u64,
        weights: *const u64,
        terms: usize,
        output: *mut u64,
    ) -> std::ffi::c_int;
}

fn available() -> bool {
    static SUPPORT: OnceLock<bool> = OnceLock::new();
    // SAFETY: The platform query accesses no Rust memory.
    *SUPPORT.get_or_init(|| unsafe { akita_packing_sme_available() != 0 })
}

pub(super) struct PackingStripes {
    positions: usize,
    words: Vec<u64>,
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
            || !available()
        {
            return None;
        }
        let len = rings
            .len()
            .checked_mul(RING_D)
            .filter(|&n| n <= MAX_WORDS)?;
        let block_len = positions * RING_D;
        let mut words = Vec::new();
        words.try_reserve_exact(len).ok()?;
        words.resize(len, 0);
        let complete = cfg_chunks_mut!(words, block_len)
            .enumerate()
            .all(|(block, out)| {
                for (position, ring) in rings[block * positions..][..positions].iter().enumerate() {
                    let Some(source) = F::canonical_u64_slice(ring.coefficients()) else {
                        return false;
                    };
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
                true
            });
        complete.then_some(Self { positions, words })
    }

    pub(super) fn contract<F, E, const D: usize>(
        &self,
        source_vars: usize,
        source_coeff_len: usize,
        weights: &FusedPackingWeights<'_, E>,
    ) -> Result<Option<Vec<F>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F> + FpExtEncoding<F>,
    {
        let tables = weights.tables();
        let point = tables.point();
        let geometry = point.geometry();
        let full_positions = self.words.len() / RING_D;
        if D != RING_D
            || F::CHALLENGE_SET_LIMBS != [Q, 0, 0, 0]
            || E::DEGREE != 2
            || geometry.extension_degree() != 2
            || geometry.subring_embedding_stride() != STRIDE
            || geometry.partial_base_field_width() != OUTPUT
            || point.num_positions_per_block() != self.positions
            || source_coeff_len != self.words.len()
            || point.num_live_positions() < full_positions
            || weights.values().len() != self.positions * STRIDE
        {
            return Ok(None);
        }
        SubringCoefficientPackingPlan { tables }.validate::<D>(source_vars)?;
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
        let mut out = Vec::new();
        out.try_reserve_exact(len).map_err(|_| {
            AkitaError::InvalidInput("coefficient-packing output allocation failed".into())
        })?;
        out.resize(len, F::zero());
        let words_per_block = self.positions * RING_D;
        let complete =
            cfg_chunks_mut!(out[..active_len], OUTPUT)
                .enumerate()
                .all(|(block, output)| {
                    let mut partial = [0u64; OUTPUT];
                    let words = &self.words[block * words_per_block..][..words_per_block];
                    // SAFETY: The constructor checks SME support and lays out full
                    // blocks. The weights contain one pair per position and residue.
                    // Each call has its own fixed-size output.
                    let ok = unsafe {
                        akita_packing_sme_striped(
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

#[cfg(test)]
#[path = "packing_sme/tests/stripes.rs"]
mod tests;
