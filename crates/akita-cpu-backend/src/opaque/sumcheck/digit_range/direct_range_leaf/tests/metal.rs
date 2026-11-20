use super::*;
use jolt_field::{Ext2, Prime64Offset59};

type E = Ext2<Prime64Offset59>;

fn point(i: usize) -> E {
    E::new(
        Prime64Offset59::from_u64(7 * i as u64 + 3),
        Prime64Offset59::from_u64(11 * i as u64 + 17),
    )
}

#[test]
fn quartic_metal_recovery_preserves_odd_tail_and_later_rounds() {
    let count = (1usize << 26) + 3;
    let mut bytes: Vec<_> = (0..(3 * count).div_ceil(8))
        .map(|i| ((i * 91 + i / 7) >> 2) as u8)
        .collect();
    *bytes.last_mut().unwrap() &= 1;
    let packed = PackedSignedDigits::import_encoded(count, 3, bytes).unwrap();
    let bits = count.next_power_of_two().ilog2() as usize;
    let tau: Vec<_> = (0..bits).map(point).collect();
    let make = || {
        LowBasisRangeCheckProver::new(
            packed.clone(),
            &tau,
            DigitRangePlan::new(8).unwrap(),
            count,
            bits,
            0,
        )
        .unwrap()
    };
    let mut cpu = make();
    let mut gpu = make();
    gpu.metal_kernels = true;
    for round in 0..bits {
        let polynomial = cpu.compute_round_eq_factored(round, E::zero());
        assert_eq!(
            gpu.compute_round_eq_factored(round, E::zero()),
            polynomial,
            "round {round}"
        );
        let r = point(bits + round);
        cpu.ingest_challenge(round, r);
        gpu.ingest_challenge(round, r);
        if round == 6 {
            let state = std::mem::replace(
                &mut gpu.range_image,
                LowBasisRangeImageStorage::Materialized(Vec::new()),
            );
            if let LowBasisRangeImageStorage::Metal(device) = state {
                gpu.range_image = LowBasisRangeImageStorage::Materialized(device.recover());
                gpu.cached_round_poly = None;
            } else {
                gpu.range_image = state;
            }
        }
    }
    assert_eq!(
        gpu.final_range_image_eval().unwrap(),
        cpu.final_range_image_eval().unwrap()
    );
}
