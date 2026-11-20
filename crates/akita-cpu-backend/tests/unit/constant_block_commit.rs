use super::*;
use crate::arithmetic::prepared_tests::{prepared, D, F};
use crate::FieldRunPoly;
use jolt_field::{Ring, Zero};

#[test]
fn constant_block_products_match_exact_ntt_with_padding_and_mixed_runs() {
    let prepared = prepared();
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let shape = DigitShape {
        log_basis: 13,
        num_digits: 5,
    };
    let plan = CommitInnerPlan {
        ring_dimension: D,
        n_a: 2,
        num_digits_inner: shape.num_digits,
        num_positions_per_block: 4,
        log_basis_inner: shape.log_basis,
        num_live_blocks: 4,
    };
    for lengths in [[4 * D, 8 * D, 4 * D], [3 * D, 8 * D + 1, 2 * D]] {
        let poly = FieldRunPoly::new(
            10,
            lengths
                .into_iter()
                .zip([F::from_u64(1 << 53), F::from_i64(-4096), F::zero()])
                .collect(),
        )
        .unwrap();
        let (ranges, width) =
            dense_commit_blocks(poly.logical_len() / D, poly.live_len().div_ceil(D), &plan)
                .unwrap();
        let digits = RunDigits::<F, D>::new(&poly, shape).unwrap();
        let reference = backend
            .commit_rows(
                &prepared,
                2,
                width,
                &digits,
                &ranges,
                shape,
                SignedDigitKernel::I16,
                false,
                I8CommitPolicy::Dense,
            )
            .unwrap();
        let output = backend
            .field_run_commit_rows::<D>(&prepared, &poly, plan)
            .unwrap();
        assert_eq!(output, reference);
    }
}
