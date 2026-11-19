use super::CpuPreparedSetup;
use crate::sources::block_digits::{DigitShape, I16Rows};
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_params::balanced_signed_digit_abs_bound;
use akita_types::{NttCacheKey, NttTransformDomain};
#[allow(unused_imports)]
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::collections::BTreeMap;
use std::ops::Range;

#[tracing::instrument(skip_all, name = "dense_commit_rows_exact_i16")]
pub(super) fn commit_rows_i16<F, S, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    n_a: usize,
    row_width: usize,
    source: &S,
    block_ranges: &[Range<usize>],
    shape: DigitShape,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    S: I16Rows<D>,
{
    let rhs_abs_bound = balanced_signed_digit_abs_bound(shape.log_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("invalid signed digit basis".into()))?;
    prepared.with_shared_ntt::<D, _>(
        NttCacheKey::from_matrix_shape(
            D,
            n_a,
            row_width,
            NttTransformDomain::ExactNegacyclicI16 {
                width: row_width,
                rhs_abs_bound,
            },
        )?,
        |ntt| {
            let uniform_values = cfg_iter!(block_ranges)
                .map(|rings| source.uniform_block_value(rings.clone()))
                .collect::<Vec<_>>();
            let mut uniform_blocks = BTreeMap::new();
            let representatives = uniform_values
                .into_iter()
                .zip(block_ranges)
                .enumerate()
                .map(|(index, (value, rings))| {
                    value.map_or(index, |value| {
                        *uniform_blocks.entry((value, rings.len())).or_insert(index)
                    })
                })
                .collect::<Vec<_>>();
            let fill = |scratch: &mut Option<S::I16Scratch>,
                        (index, rings): (usize, &Range<usize>)| {
                if representatives[index] != index {
                    return Ok(Vec::new());
                }
                if rings.is_empty() {
                    return Ok(vec![CyclotomicRing::zero(); n_a]);
                }
                let scratch = scratch
                    .get_or_insert_with(|| source.i16_scratch(shape, row_width / shape.num_digits));
                let rhs = source.i16_rows(rings.clone(), shape, scratch)?;
                ntt.mat_vec_i16::<F>(shape.log_basis, n_a, rhs)
            };
            #[cfg(feature = "parallel")]
            let result = cfg_iter!(block_ranges)
                .enumerate()
                .map_init(|| None, fill)
                .collect::<Result<Vec<_>, _>>();
            #[cfg(not(feature = "parallel"))]
            let result = {
                let mut rhs = None;
                block_ranges
                    .iter()
                    .enumerate()
                    .map(|block| fill(&mut rhs, block))
                    .collect::<Result<Vec<_>, _>>()
            };
            let mut rows = result?;
            for (index, representative) in representatives.into_iter().enumerate() {
                if index != representative {
                    rows[index] = rows[representative].clone();
                }
            }
            Ok(rows)
        },
    )
}
