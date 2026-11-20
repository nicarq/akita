use super::{run, workspace_fits, LookupShape};
use akita_params::FlatMatrix;
use akita_types::ntt_cache::{prepare_ntt_cache, NttCacheMode};
use jolt_field::{Prime64Offset59 as F, Ring};

#[test]
fn mixed_lookup_matches_cpu_for_partial_odd_and_empty_ranges() {
    const D: usize = 512;
    const WIDTH: usize = 160;
    #[repr(align(16384))]
    struct CodePage([u16; 8192]);
    let mut codes = Box::new(CodePage([0; 8192]));
    for (i, code) in codes.0.iter_mut().enumerate() {
        *code = [0, 1, 2, u16::MAX][(i ^ (i / 3)) % 4];
    }
    let lookup = [
        [0, 1, -1, 4095, -4096],
        [7, -9, 31, -1023, 4095],
        [2, 79, -35, 119, -4000],
        [0, -5, 1, 2304, 3899],
        [4, 17, -991, 777, 13],
        [-4049, -1, 0, 35, -11],
    ];
    let matrix = FlatMatrix::from_flat_data(
        (0..3 * WIDTH * D)
            .map(|i| F::from_u64((i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15)))
            .collect(),
    );
    let cache = prepare_ntt_cache(
        matrix.ring_view::<D>(3, WIDTH).unwrap(),
        NttCacheMode::ExactNegacyclic {
            width: WIDTH,
            rhs_abs_bound: 4096,
        },
    )
    .unwrap();
    let view = cache.i32_limb_view().expect("two-prime limb matrix");
    assert_eq!(view.limbs, 3);
    let ranges = [[0, 18], [26, 5], [32, 0]];
    let shape = LookupShape {
        blocks: 3,
        cols: WIDTH as u32,
        chunk: 256,
        table_len: 3,
        column_width: 8192,
        code_count: 8192,
        live_cells: 13827,
        primes: view.params.primes.map(|q| [q.p, q.pinv, q.mont, q.montsq]),
    };
    let got = run::<F, D>(view, &shape, &codes.0, &lookup, &ranges)
        .expect("mixed Metal lookup dispatched");
    for ([start, length], rows) in ranges.into_iter().zip(got) {
        let rhs = (0..WIDTH)
            .map(|col| {
                std::array::from_fn(|lane| {
                    let global = (start as usize + col / 5) * D + lane;
                    if col / 5 >= length as usize || global >= shape.live_cells as usize {
                        return 0;
                    }
                    let code = codes.0[global % 8192] as usize;
                    if code >= 3 {
                        0
                    } else {
                        lookup[(global / 8192) * 3 + code][col % 5]
                    }
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, cache.mat_vec_i16::<F>(13, 3, &rhs).unwrap());
    }
}

#[test]
fn lookup_workspace_includes_host_and_device_tables() {
    assert!(!workspace_fits::<512>(64, 5120, 4 * 8193, 10));
    assert!(!workspace_fits::<512>(64, 5120, 25_550, 10));
    assert!(workspace_fits::<512>(64, 5120, 4 * 8193, 11));
    assert!(workspace_fits::<512>(61, 5120, 4 * 5633, 10));
    assert!(!workspace_fits::<512>(61, 5120, 4 * 5633, 9));
}
