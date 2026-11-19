use super::common;
use akita_config::proof_optimized::fp128;
use akita_pcs::{BasisMode, CpuBackend, DensePoly, FieldRunPoly, GroupContext};
use akita_serialization::AkitaSerialize;
use jolt_field::{CanonicalEncoding, One, Ring, Zero};

type F = fp128::Field;
type Config = fp128::Dense;
const VARS: usize = 14;
const DOMAIN: &[u8] = b"akita/field-run-poly-parity";

#[test]
fn field_run_poly_commitment_and_proof_bytes_match_dense() {
    common::init_rayon_pool();
    common::run_on_large_stack(|| {
        let scheme = common::load_workspace_scheme::<Config>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap();
        let setup = scheme.setup_prover(VARS, 1).unwrap();
        let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
        let verifier = scheme
            .verifier(scheme.setup_verifier(&setup).unwrap().clone())
            .unwrap();
        let point = common::random_point(VARS, 0x51b1_0093);
        let positive = F::from_u128_reduced((1u128 << 110) + 7);
        let negative = -F::from_u128_reduced((1u128 << 122) - 5);
        let mut cases: Vec<_> = [
            vec![
                (1 << 12, positive),
                (1 << 12, negative),
                (1 << 13, positive),
            ],
            vec![(1 << 12, positive), (173, F::zero()), (4097, negative)],
        ]
        .into_iter()
        .map(|runs| {
            let coefficients = runs
                .iter()
                .flat_map(|&(len, value)| std::iter::repeat_n(value, len))
                .collect::<Vec<_>>();
            (FieldRunPoly::new(VARS, runs).unwrap(), coefficients)
        })
        .collect();
        let width = 1 << (VARS - 2);
        let indices: Vec<_> = (0..width - 19)
            .map(|i| match i % 5 {
                0 => u16::MAX,
                1 | 2 => 0,
                _ => 1,
            })
            .collect();
        let lookup = vec![
            vec![positive, negative],
            vec![negative, positive],
            vec![F::zero(), negative],
        ];
        let mut reference = Vec::new();
        for column in &lookup {
            let start = reference.len();
            reference.extend(indices.iter().map(|&code| {
                column
                    .get(usize::from(code))
                    .copied()
                    .unwrap_or_else(F::zero)
            }));
            reference.resize(start + width, F::zero());
        }
        cases.push((
            FieldRunPoly::from_lookup_columns(VARS, width, std::sync::Arc::new(indices), lookup)
                .unwrap(),
            reference,
        ));
        for (start, positive, negative, tail) in [
            (
                0,
                vec![1, 0x8020, u64::MAX],
                vec![1 << 63, 0x4000, 0],
                (277, positive),
            ),
            (
                83,
                vec![0, 0xff00ff00ff00ff00],
                vec![0, 0x00ff00ff00ff00ff],
                (543, F::zero()),
            ),
            (61, vec![], vec![], (92, negative)),
        ] {
            let mut reference = vec![F::zero(); tail.0 + 1];
            for (i, (&p, &n)) in positive.iter().zip(&negative).enumerate() {
                for bit in 0..64 {
                    reference[start + 64 * i + bit] = match ((p >> bit) & 1, (n >> bit) & 1) {
                        (1, 0) => F::one(),
                        (0, 1) => -F::one(),
                        _ => F::zero(),
                    };
                }
            }
            reference[tail.0] = tail.1;
            cases.push((
                FieldRunPoly::from_signed_span(VARS, start, positive, negative, tail).unwrap(),
                reference,
            ));
        }
        for (source, mut coefficients) in cases {
            coefficients.resize(1 << VARS, F::zero());
            let claim = [common::dense_opening_lagrange(&coefficients, &point)];
            let compact = backend.import_source(vec![source]).unwrap();
            let dense = backend
                .import_source(vec![
                    DensePoly::from_field_evals(VARS, coefficients).unwrap()
                ])
                .unwrap();
            let run = |source| {
                let output = backend
                    .commit(
                        scheme.schedules(),
                        &source,
                        GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .unwrap();
                drop(source);
                let mut commitment_bytes = Vec::new();
                output
                    .committed_group
                    .serialize_compressed(&mut commitment_bytes)
                    .unwrap();
                let opening = common::prove_input::<Config>(
                    &point,
                    &claim,
                    &output.committed_group,
                    output.private_handle,
                    scheme.schedules(),
                );
                let proof = scheme
                    .batched_prove(&setup, opening, &backend, DOMAIN, BasisMode::Lagrange)
                    .unwrap();
                verifier
                    .batched_verify(
                        &proof,
                        DOMAIN,
                        common::verify_input::<Config>(
                            &point,
                            &claim,
                            &output.committed_group,
                            scheme.schedules(),
                        ),
                        BasisMode::Lagrange,
                        akita_verifier::SetupPolicy::Check(None),
                    )
                    .unwrap();
                (commitment_bytes, proof)
            };
            assert_eq!(run(compact), run(dense));
        }
    });
}

#[test]
fn field_run_poly_checks_prefix_extents() {
    let value = F::from_u64(2);
    assert!(FieldRunPoly::new(3, vec![(0, value)]).is_err());
    assert!(FieldRunPoly::new(3, vec![(5, value), (4, value)]).is_err());
    assert!(FieldRunPoly::new(3, vec![(5, value), (usize::MAX, value)]).is_err());
    assert!(FieldRunPoly::<F>::new(usize::BITS as usize, vec![]).is_err());
    let signed = |start, p: Vec<u64>, n: Vec<u64>, tail| {
        FieldRunPoly::from_signed_span(8, start, p, n, (tail, value))
    };
    assert!(signed(0, vec![1], vec![1], 100).is_err());
    assert!(signed(0, vec![0], vec![], 100).is_err());
    assert!(signed(40, vec![1], vec![0], 99).is_err());
    assert!(signed(0, vec![], vec![], 256).is_err());
    let lookup = |width, codes: Vec<u16>, tables: Vec<Vec<F>>| {
        FieldRunPoly::from_lookup_columns(3, width, std::sync::Arc::new(codes), tables)
    };
    assert!(lookup(3, vec![0], vec![vec![value]]).is_err());
    assert!(lookup(2, vec![1], vec![vec![value]]).is_err());
    assert!(lookup(2, vec![0; 3], vec![vec![value]]).is_err());
    assert!(lookup(4, vec![], vec![vec![value]; 3]).is_err());
    assert!(lookup(4, vec![u16::MAX], vec![vec![value], vec![]]).is_err());
}
