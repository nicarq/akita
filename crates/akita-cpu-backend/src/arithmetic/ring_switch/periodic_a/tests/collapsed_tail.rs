#[test]
fn collapsed_tail_matches_direct_integer_convolution() {
    const D: usize = 64;
    const PERIOD: usize = 5;
    const Q: i128 = (1i128 << 64) - 59;

    let matrix: Vec<CyclotomicRing<F, D>> = (0..PERIOD)
        .map(|row| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
                F::from_u64(((row * D + i) as u64).wrapping_mul(0xbc3c_748d_e2f0_d3eb))
            }))
        })
        .collect();
    let rhs: Vec<[i32; D]> = (0..PERIOD)
        .map(|row| std::array::from_fn(|i| ((i + row * 41) as i32 - 100) << 21))
        .collect();
    let mut coefficients = [0i128; D];
    for (a, z) in matrix.iter().zip(&rhs) {
        for (i, value) in a.coefficients().iter().enumerate() {
            let a = value.to_u128_checked().unwrap() as i128;
            for (output, z) in coefficients.iter_mut().zip(&z[D - i..]) {
                *output += a * i128::from(*z);
            }
        }
    }
    let expected = coefficients.map(|x| F::from_u64(x.rem_euclid(Q) as u64));
    let result = small_field_matrix_a_quotients_with_tail(&matrix, 1, &rhs, 0, 4)
        .unwrap()
        .unwrap();
    assert_eq!(result[0].coefficients(), &expected);
}
