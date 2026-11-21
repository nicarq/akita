use super::*;
use std::time::Instant;

#[test]
fn metal_subring_fold_matches_cpu() {
    let positions = 65;
    let blocks = 95;
    let mut state = 0x3141592u64;
    let mut next_word = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 31) as i64 % PRIME.p as i64 - PRIME.p as i64 / 2) as i32
    };
    let rows = positions * DIGITS * SUBRING_D * blocks;
    let mut spectra = vec![0; packed::storage_words(rows, packed::page_words()).unwrap()];
    for row in &mut spectra.as_chunks_mut::<15>().0[..rows] {
        for lane in 0..STRIDE {
            packed::write(row, lane, next_word());
        }
    }
    let cache = FixedFoldNtt {
        positions,
        blocks,
        live_rings: positions * blocks,
        spectra: Some(spectra),
        metal: std::sync::OnceLock::new(),
        ntt: NttKernelPlan::detect::<i32>(),
        twiddles: NttTwiddles::compute(PRIME),
        dot: dot_neon,
        #[cfg(any(akita_packing_sme, akita_fold_metal))]
        packing: None,
    };
    let weights: Vec<_> = (0..SUBRING_D * blocks).map(|_| next_word()).collect();
    for range in [0..blocks, 3..blocks - 5] {
        let mut cpu = vec![[0; RING_D]; positions * DIGITS];
        let mut actual = cpu.clone();
        for _ in 0..5 {
            let start = Instant::now();
            cache.fill_cpu(
                cache.spectra.as_deref().unwrap(),
                &range,
                &weights,
                &mut cpu,
            );
            let cpu_us = start.elapsed().as_micros();
            let start = Instant::now();
            assert!(cache.metal_fold(
                cache.spectra.as_deref(),
                &range,
                &weights,
                &mut actual,
                None
            ));
            let gpu_us = start.elapsed().as_micros();
            assert!(actual == cpu);
            eprintln!("range={range:?} cpu_us={cpu_us} metal_us={gpu_us}");
        }
    }
}

#[test]
fn raw_metal_subring_fold_matches_precomputed_spectra() {
    use super::super::poly::DensePoly;
    use jolt_field::Fp64;
    use std::slice::from_ref;
    use std::sync::Arc;

    const Q: u64 = u64::MAX - 58;
    let positions = 64;
    let blocks = 35;
    let threshold = 0x7fff7fff7fff7fffu64;
    let mut state = 718289112u64;
    let words = Arc::new(
        (0..positions * blocks * RING_D)
            .map(|i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                match i % 13 {
                    0 => threshold - 3 + (i % 7) as u64,
                    1 => Q - 1 - (i % 89) as u64,
                    _ => state % Q,
                }
            })
            .collect(),
    );
    let source = DensePoly::<Fp64<Q>>::from_shared_canonical_u64_prefix(23, words).unwrap();
    let words = source.fold_words().unwrap();
    let mut cache =
        FixedFoldNtt::new(source.nonzero_ring_coeffs::<RING_D>().unwrap(), positions).unwrap();
    let weights: Vec<_> = (0..SUBRING_D * blocks)
        .map(|i| (i as i32 % 43) - 21)
        .collect();
    for range in [0..blocks, 1..34, 7..8] {
        let mut cpu = vec![[0; RING_D]; positions * DIGITS];
        let mut actual = cpu.clone();
        cache.fill_cpu(
            cache.spectra.as_deref().unwrap(),
            &range,
            &weights,
            &mut cpu,
        );
        assert!(cache.metal_fold(None, &range, &weights, &mut actual, Some(&words)));
        assert!(actual == cpu);
    }

    let challenges = (0..blocks + 2)
        .map(|b| SparseChallenge {
            positions: vec![0, 16, 32, 32, 2032, ((b % 128) * 16) as u32].into(),
            coeffs: vec![-3, 2, -4, 1, 3, 0].into(),
        })
        .collect::<Vec<_>>();
    let chunks = [0..1, 1..34, 3..3, 34..37, 35..37, 5..15];
    let expected = cache
        .fold::<RING_D>(&challenges, &chunks, false, Some(&words))
        .unwrap()
        .unwrap();
    cache.spectra.take();
    let actual = cache
        .fold::<RING_D>(&challenges, &chunks, true, Some(&words))
        .unwrap()
        .unwrap();
    assert!(actual == expected);
    assert!(cache.spectra.is_none());
    assert!(cache
        .fold::<RING_D>(&challenges, &chunks, true, None)
        .unwrap()
        .is_none());
    assert!(cache
        .fold::<RING_D>(
            &challenges,
            from_ref(&(0..challenges.len() + 1)),
            true,
            Some(&words)
        )
        .unwrap()
        .is_none());
    let block = 0..1;
    let first_block = from_ref(&block);
    let unsupported = SparseChallenge {
        positions: vec![1].into(),
        coeffs: vec![1].into(),
    };
    assert!(cache
        .fold::<RING_D>(&[unsupported], first_block, true, Some(&words))
        .unwrap()
        .is_none());
    let excessive = SparseChallenge {
        positions: vec![0; 513].into(),
        coeffs: vec![-128; 513].into(),
    };
    assert!(cache
        .fold::<RING_D>(&[excessive], first_block, true, Some(&words))
        .unwrap()
        .is_none());
    let malformed = SparseChallenge {
        positions: vec![RING_D as u32].into(),
        coeffs: vec![1].into(),
    };
    assert!(cache
        .fold::<RING_D>(&[malformed], first_block, true, Some(&words))
        .is_err());
    assert!(cache.spectra.is_none());
    let mut source = source;
    source.fixed_fold_ntt = Some(Arc::new(cache));
    let cpu_rows = source
        .decompose_fold_chunked::<RING_D>(&challenges, &chunks, positions, DIGITS, LOG_BASIS, false)
        .unwrap();
    for (row, expected) in cpu_rows.into_iter().zip(expected) {
        assert!(row.into_centered_coeffs_flat() == expected.into_flattened());
    }
    assert!(source.fixed_fold_ntt.as_ref().unwrap().spectra.is_none());
}
