use super::super::poly::DensePoly;
use super::{DIGITS, LOG_BASIS, RING_D};
use akita_challenges::SparseChallenge;
use jolt_field::Prime64Offset59;
use std::sync::Arc;

#[test]
fn large_cpu_source_folds_from_coefficients_like_a_cached_source() {
    let mut state = 9929u64;
    let prefix: Vec<_> = (0..8 * RING_D)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state % (u64::MAX - 58)
        })
        .collect();
    let mut large_words = vec![0; ((1 << 15) + 8) * RING_D];
    large_words[..prefix.len()].copy_from_slice(&prefix);
    *large_words.last_mut().unwrap() = 1;
    let source = |words| {
        DensePoly::<Prime64Offset59>::from_shared_canonical_u64_prefix(27, Arc::new(words))
            .unwrap()
            .with_fixed_fold_ntt_2048(2)
            .unwrap()
    };
    let cached = source(prefix);
    let large = source(large_words);
    assert!(cached.fixed_fold_ntt.as_ref().unwrap().spectra.is_some());
    assert!(large.fixed_fold_ntt.as_ref().unwrap().spectra.is_none());

    let challenges: Vec<_> = (0..4)
        .map(|i| SparseChallenge {
            positions: vec![0, 16, 16, 2032].into(),
            coeffs: vec![i, -5, 2, 1].into(),
        })
        .collect();
    let chunks = [0..4, 1..3, 2..2];
    let fold = |source: &DensePoly<Prime64Offset59>| {
        source
            .decompose_fold_chunked::<RING_D>(&challenges, &chunks, 2, DIGITS, LOG_BASIS, false)
            .unwrap()
            .into_iter()
            .map(|row| row.into_centered_coeffs_flat())
            .collect::<Vec<_>>()
    };
    assert!(fold(&large) == fold(&cached));
}
