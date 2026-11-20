use super::metal_packed;
use crate::opaque::{ComputeBackendSetup, CpuBackend};
use crate::sources::packed_digits::PackedSignedDigits;
use crate::AkitaProverSetup;
use akita_params::SetupMatrixCapacity;
use akita_types::{NttCacheKey, NttTransformDomain};
use jolt_field::Prime64Offset59 as F;

#[test]
fn metal_recursive_commit_matches_two_cpu_rows_and_pads_the_tail() {
    const D: usize = 512;
    const COLS: usize = 2048;
    const BLOCKS: usize = 17;
    let cpu = CpuBackend::<F, F>::for_arithmetic_tests();
    let gpu = cpu.with_metal_kernels();
    let setup = AkitaProverSetup::generate_with_capacity(
        8,
        1,
        SetupMatrixCapacity {
            num_field_elements: 2 * COLS * D,
        },
    )
    .unwrap();
    let prepared = cpu.prepare_setup(&setup).unwrap();
    let length = ((BLOCKS - 1) * COLS + 7) * D;
    let packed = PackedSignedDigits::from_i8_digits_auto(
        (0..length)
            .map(|i| ((i ^ (i >> 5) ^ (i >> 12)) & 7) as i8 - 4)
            .collect(),
    );
    let digits = packed.zero_padded(length + D).unwrap();
    let run = |backend: &CpuBackend<F, F>| {
        backend
            .recursive_packed_witness_commit_rows::<D>(&prepared, digits, 2, COLS, BLOCKS, 1, 3)
            .unwrap()
    };
    let expected = run(&cpu);
    assert_eq!(run(&gpu), expected);
    let key = NttCacheKey::from_matrix_shape(D, 2, COLS, NttTransformDomain::Negacyclic).unwrap();
    let direct = prepared
        .with_shared_ntt::<D, _>(key, |slot| {
            Ok(metal_packed::commit::<F, D>(slot, digits, COLS, BLOCKS))
        })
        .unwrap();
    assert_eq!(
        direct.expect("Metal dispatched the two-row commitment"),
        expected
    );
}
