#[test]
fn periodic_quotient_matches_full_matrix_and_checks_every_row() {
    const D: usize = 64;
    const PERIOD: usize = 5;
    const WIDTH: usize = 25;
    const RANK: usize = 3;

    let z: Vec<[i32; D]> = (0..WIDTH)
        .map(|j| std::array::from_fn(|i| ((i + (j % PERIOD) * 121) % 751) as i32 - 374))
        .collect();
    let matrix: Vec<CyclotomicRing<F, D>> = (0..RANK * WIDTH)
        .map(|j| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
                F::from_u64(((j * D + i) as u64).wrapping_mul(0xf3a5_c817_b283_9321))
            }))
        })
        .collect();
    assert_eq!(checked_row_period(&z), Some(PERIOD));
    let full =
        fused_split_eq_quotients_streamed_prover_bounds(&matrix, 0, RANK, &[], &z, 750, 4).unwrap();
    let collapsed = fold_matrix_columns(&matrix, RANK, WIDTH, PERIOD);
    let grouped = fused_split_eq_quotients_streamed_prover_bounds(
        &collapsed,
        0,
        RANK,
        &[],
        &z[..PERIOD],
        750,
        4,
    )
    .unwrap();
    assert_eq!(grouped.a_quotients, full.a_quotients);

    let mut invalid_period = z;
    invalid_period[WIDTH - 1][D - 1] ^= 1;
    assert_eq!(checked_row_period(&invalid_period), None);
}
