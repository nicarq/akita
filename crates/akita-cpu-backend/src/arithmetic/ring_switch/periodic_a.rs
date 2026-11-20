#[cfg(test)]
use akita_algebra::CyclotomicRing;
use jolt_field::solinas::parallel::*;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arithmetic::matrix_columns::fold_matrix_columns;
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
