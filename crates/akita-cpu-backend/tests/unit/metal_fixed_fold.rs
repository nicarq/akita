use super::*;
use std::time::Instant;

#[test]
fn metal_subring_fold_matches_cpu() {
    let positions = 64;
    let blocks = 96;
    let mut state = 0x3141592u64;
    let mut next_word = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 31) as i64 % PRIME.p as i64 - PRIME.p as i64 / 2) as i32
    };
    let spectra = (0..positions * DIGITS * SUBRING_D * blocks)
        .map(|_| {
            let mut row = PackedSpectrum::default();
            for lane in 0..STRIDE {
                row.write(lane, next_word());
            }
            row
        })
        .collect::<Vec<_>>();
    let cache = FixedFoldNtt {
        positions,
        blocks,
        live_rings: positions * blocks,
        spectra,
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
            cache.fill_cpu(&range, &weights, &mut cpu);
            let cpu_us = start.elapsed().as_micros();
            let start = Instant::now();
            assert!(cache.metal_fold(&range, &weights, &mut actual));
            let gpu_us = start.elapsed().as_micros();
            assert!(actual == cpu);
            eprintln!("range={range:?} cpu_us={cpu_us} metal_us={gpu_us}");
        }
    }
}
