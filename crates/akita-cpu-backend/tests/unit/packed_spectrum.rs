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
            packed::write(&mut packed, lane, source[lane]);
            sums[lane] += i128::from(weight) * i128::from(source[lane]);
        }
        assert_eq!(packed::unpack(&packed), source);
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
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        for (actual, want) in dot_neon(&spectra, &weights).into_iter().zip(expected) {
            assert_eq!(
                i64::from(actual).rem_euclid(P),
                i64::from(want).rem_euclid(P)
            );
        }
    }
}

#[test]
fn page_padding_preserves_metal_shapes_within_the_original_cache() {
    for page_words in [1024, 4096, 16384] {
        for positions in 1..=32 {
            for blocks in 1..=33 {
                let rows = positions * blocks * DIGITS * SUBRING_D;
                let words = packed::storage_words(rows, page_words).unwrap();
                assert!(words >= rows * 15 && words <= rows * STRIDE);
                if (rows * STRIDE).is_multiple_of(page_words) {
                    assert!(words.is_multiple_of(page_words));
                } else {
                    assert_eq!(words, rows * 15);
                }
            }
        }
    }
}
