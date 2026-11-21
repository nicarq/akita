use super::super::poly::DensePoly;
use super::{SparseChallenge, DIGITS, RING_D};
use akita_algebra::{Ring, Zero};
use jolt_field::Prime64Offset59 as F;
use std::ops::Range;
use std::sync::Arc;

#[test]
fn partial_shared_and_owned_keys_fold_on_metal_without_spectra() {
    const POSITIONS: usize = 64;
    const BLOCKS: usize = 513;
    let block = POSITIONS * RING_D;
    let challenge = |b: usize| SparseChallenge {
        positions: vec![0, 16, 16, 2032].into(),
        coeffs: vec![1, (b % 5) as i8 - 2, 2, -1].into(),
    };
    let challenges: Vec<_> = (0..BLOCKS).map(challenge).collect();
    let oracle_challenges = [challenge(0), challenge(BLOCKS - 1)];
    let chunks = [0..1, BLOCKS - 1..BLOCKS, 0..BLOCKS];
    let oracle_chunks = [0..1, 1..2, 0..2];

    let fold =
        |source: &DensePoly<F>, challenges: &[SparseChallenge], chunks: &[Range<usize>], metal| {
            let cache = source.fixed_fold_ntt.as_ref().unwrap();
            assert_eq!(cache.spectra.is_none(), metal);
            cache
                .fold::<RING_D>(challenges, chunks, metal, source.fold_words().as_ref())
                .unwrap()
                .expect("requested backend completes every chunk")
        };
    for owned in [false, true] {
        let missing = if owned { 0 } else { RING_D };
        let mut edge = vec![0; 2 * block];
        let mut state = 17u64;
        for word in &mut edge[..2 * block - missing] {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *word = state % (u64::MAX - 58);
        }
        let oracle = DensePoly::<F>::from_shared_canonical_u64_prefix(18, Arc::new(edge.clone()))
            .unwrap()
            .with_fixed_fold_ntt_2048(POSITIONS)
            .unwrap();
        let expected = fold(&oracle, &oracle_challenges, &oracle_chunks, false);
        let polys = if owned {
            let mut cells = vec![F::zero(); BLOCKS * block];
            for (dst, src) in cells[..block].iter_mut().zip(&edge[..block]) {
                *dst = F::from_u64(*src);
            }
            for (dst, src) in cells[(BLOCKS - 1) * block..].iter_mut().zip(&edge[block..]) {
                *dst = F::from_u64(*src);
            }
            DensePoly::from_field_prefix(27, cells).unwrap()
        } else {
            let mut cells = vec![0; BLOCKS * block];
            cells[..block].copy_from_slice(&edge[..block]);
            cells[(BLOCKS - 1) * block..].copy_from_slice(&edge[block..]);
            DensePoly::from_shared_canonical_u64_prefix(27, Arc::new(cells)).unwrap()
        };
        let source = polys.with_fixed_fold_ntt_2048(POSITIONS).unwrap();
        assert_eq!(
            source.nonzero_ring_coeffs::<RING_D>().unwrap().len(),
            BLOCKS * POSITIONS - missing / RING_D
        );
        let actual = fold(&source, &challenges, &chunks, true);
        assert!(actual == expected);
        assert_eq!(actual[1].len(), POSITIONS * DIGITS);
        if !owned {
            assert!(actual[1][(POSITIONS - 1) * DIGITS..]
                .iter()
                .flatten()
                .all(|&n| n == 0));
        }
    }
}
