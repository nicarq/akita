use super::{metal_quotient, CpuBackend};
use crate::opaque::RingSwitchRelationPlan;
use crate::AkitaProverSetup;
use akita_params::SetupMatrixCapacity;
use jolt_field::Prime64Offset59 as F;

#[test]
fn metal_quotient_matches_cpu_across_chunk_boundaries() {
    const D: usize = 2048;
    const WIDTH: usize = 1028;
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: D * WIDTH,
        },
    )
    .unwrap();
    let cpu = CpuBackend::<F, F>::with_ring_switch_cache_limit(setup.expanded, D * WIDTH).unwrap();
    let prepared = cpu.prepared().unwrap();
    let rhs = (0..WIDTH)
        .map(|col| {
            std::array::from_fn(|i| {
                let x = (col * D + i) as u32;
                (x.wrapping_mul(83_471_329).rotate_left(5) % 19_000_001) as i32 - 9_500_000
            })
        })
        .collect::<Vec<[i32; D]>>();
    let bound = rhs
        .as_flattened()
        .iter()
        .map(|x| x.unsigned_abs())
        .max()
        .unwrap();
    let plan = RingSwitchRelationPlan {
        n_d: 0,
        n_b: 0,
        n_a: 1,
        log_basis_open: 3,
        log_basis_outer: 3,
    };
    let expected = super::ring_switch::relation_b_a_rows(&cpu, prepared, &[], &rhs, bound, plan)
        .unwrap()
        .a_quotients;
    let direct = metal_quotient::quotient::<F, D>(
        prepared.expanded.shared_matrix().as_field_slice(),
        &rhs,
        u64::from(bound),
    )
    .unwrap();
    assert_eq!(direct, expected[0]);
    assert!(prepared.drop_built_ntt_slots().unwrap() > 0);
    let gpu = cpu.with_metal_kernels();
    let actual = super::ring_switch::relation_b_a_rows(&gpu, prepared, &[], &rhs, bound, plan)
        .unwrap()
        .a_quotients;
    assert_eq!(actual, expected);
    assert_eq!(prepared.shared_ntt_cache_bytes(), 0);
}
