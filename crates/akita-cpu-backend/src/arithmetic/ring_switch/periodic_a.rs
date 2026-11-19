use akita_algebra::CyclotomicRing;
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

const MAX_PERIOD: usize = 8;
const REPEATS_PER_CHECK: usize = 64;

pub(super) fn checked_row_period<const D: usize>(rows: &[[i32; D]]) -> Option<usize> {
    let prefix = rows.get(..2 * MAX_PERIOD)?;
    let period = (1..=MAX_PERIOD).find(|&period| {
        rows.len().is_multiple_of(period) && prefix[period..] == prefix[..prefix.len() - period]
    })?;
    let template = &rows[..period];
    cfg_chunks!(rows, period * REPEATS_PER_CHECK)
        .all(|chunk| chunk.chunks_exact(period).all(|rows| rows == template))
        .then_some(period)
}

pub(super) fn fold_matrix_columns<F: Field, const D: usize>(
    matrix: &[CyclotomicRing<F, D>],
    rows: usize,
    width: usize,
    period: usize,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..rows * period)
        .map(|slot| {
            let row = slot / period;
            let first = slot % period;
            let mut sum = CyclotomicRing::<F, D>::zero();
            for source in matrix[row * width..(row + 1) * width]
                .iter()
                .skip(first)
                .step_by(period)
            {
                for (coefficient, &value) in
                    sum.coefficients_mut().iter_mut().zip(source.coefficients())
                {
                    *coefficient += value;
                }
            }
            sum
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linear::fused_split_eq_quotients_streamed_prover_bounds;
    use jolt_field::{Prime64Offset59 as F, Ring};

    include!("periodic_a/tests/periodic_quotient.rs");
}

#[cfg(test)]
mod tail_tests {
    use super::*;
    use crate::kernels::linear::small_field_matrix_a_quotients_with_tail;
    use jolt_field::{CanonicalEncoding, Prime64Offset59 as F, Ring};

    include!("periodic_a/tests/collapsed_tail.rs");
}
