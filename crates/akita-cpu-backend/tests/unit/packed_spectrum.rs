use super::*;

#[test]
fn signed_spectra_preserve_all_lanes_and_dot_products() {
    let boundary = [
        -(1 << 29),
        -(PRIME.p / 2),
        -2,
        -1,
        0,
        1,
        PRIME.p / 2,
        (1 << 29) - 1,
    ];
    let mut state = 17u64;
    let mut spectra = Vec::new();
    let mut weights = Vec::new();
    let mut sums = [0i128; STRIDE];
    for row in 0..97 {
        let mut packed = PackedSpectrum::default();
        let source: [i32; STRIDE] = std::array::from_fn(|lane| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            if row < boundary.len() {
                boundary[(lane + row) % boundary.len()]
            } else {
                (state as i32) >> 2
            }
        });
        let weight = boundary[row % boundary.len()] / 2;
        for lane in (0..STRIDE).rev() {
            packed.write(lane, source[lane]);
            sums[lane] += i128::from(weight) * i128::from(source[lane]);
        }
        assert_eq!(packed.unpack(), source);
        weights.push(weight);
        spectra.push(packed);
    }
    let expected = sums.map(|sum| (sum % i128::from(P)) as i32);
    for (actual, want) in dot_scalar(&spectra, &weights).into_iter().zip(expected) {
        assert_eq!(
            i64::from(actual).rem_euclid(P),
            i64::from(want).rem_euclid(P)
        );
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        for (actual, want) in dot_neon(&spectra, &weights).into_iter().zip(expected) {
            assert_eq!(
                i64::from(actual).rem_euclid(P),
                i64::from(want).rem_euclid(P)
            );
        }
    }
}
