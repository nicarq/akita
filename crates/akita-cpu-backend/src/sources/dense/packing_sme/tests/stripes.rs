use super::{available, PackingStripes, OUTPUT, Q, RING_D, STRIDE};
use crate::arithmetic::coefficient_packing::FusedPackingWeights;
use akita_algebra::CyclotomicRing;
use akita_params::{BasisMode, SubringCoefficientPackingGeometry};
use akita_types::{PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingTables};
use jolt_field::{Ext2, ExtField, Prime64Offset59, Ring};

type F = Prime64Offset59;
type E = Ext2<F>;

#[test]
fn fixed_packing_stripes_match_scalar_products_and_zero_suffix() {
    if !available() && !cfg!(akita_fold_metal) {
        return;
    }
    const VARS: usize = 21;
    const POSITIONS: usize = 64;
    let geometry = SubringCoefficientPackingGeometry::try_new(2, RING_D, 128).unwrap();
    let point = (0..VARS)
        .map(|j| {
            E::new(
                F::from_u64(Q - 5 - j as u64),
                F::from_u64(7 * j as u64 + 19),
            )
        })
        .collect::<Vec<_>>();
    let point = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        1024,
        POSITIONS,
        VARS,
        &point,
    )
    .unwrap();
    let tables = SubringCoefficientPackingTables::new(point).unwrap();
    let weights = FusedPackingWeights::new(&tables).unwrap();
    let rings = (0..8 * POSITIONS)
        .map(|ring| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|coordinate| {
                let samples = [0, Q - 1, 7, Q - 3, Q / 2, 1, Q - 16];
                F::from_u64(
                    samples
                        [(ring * 7 + coordinate * 5 + coordinate / 3 + ring / 2) % samples.len()],
                ) + F::from_u64((ring / POSITIONS) as u64)
            }))
        })
        .collect::<Vec<CyclotomicRing<F, RING_D>>>();
    let source: Vec<F> = rings
        .iter()
        .flat_map(|ring| ring.coefficients().iter().copied())
        .collect();
    let cache = PackingStripes::new(&rings, POSITIONS).unwrap();
    let mut expected = vec![F::from_u64(0); 16 * OUTPUT];
    for (block, rows) in rings.chunks_exact(POSITIONS).enumerate() {
        let mut sums = [E::new(F::from_u64(0), F::from_u64(0)); RING_D / STRIDE];
        for (position, row) in rows.iter().enumerate() {
            for (coordinate, &value) in row.coefficients().iter().enumerate() {
                sums[coordinate / STRIDE] +=
                    weights.values()[position * STRIDE + coordinate % STRIDE].mul_base(value);
            }
        }
        for (coordinate, value) in sums.into_iter().enumerate() {
            expected[block * OUTPUT + coordinate] = value.coeffs[0];
            expected[block * OUTPUT + coordinate + OUTPUT / 2] = value.coeffs[1];
        }
    }
    assert_ne!(expected[..OUTPUT], expected[7 * OUTPUT..8 * OUTPUT]);
    let cpu = cache
        .contract::<F, E, RING_D>(VARS, &source, &weights, None)
        .unwrap();
    match cpu {
        Some(partials) => assert_eq!(partials, expected),
        None => assert!(!available()),
    }
    #[cfg(akita_fold_metal)]
    {
        let gpu_cache = PackingStripes::new(&rings, POSITIONS).unwrap();
        let words = std::sync::Arc::new(
            jolt_field::CanonicalEncoding::canonical_u64_slice(&source)
                .unwrap()
                .to_vec(),
        );
        let poly = crate::DensePoly::from_shared_canonical_u64_prefix(VARS, words.clone()).unwrap();
        let gpu_result = gpu_cache
            .contract::<F, E, RING_D>(VARS, poly.field_coeffs(), &weights, Some(&words))
            .unwrap();
        assert_eq!(gpu_result, Some(expected));
        let handle = gpu_cache.metal.get().unwrap().as_ref().unwrap().handle;
        let owners = std::sync::Arc::strong_count(&words);
        let row_weights: [[u64; 2]; 8] = std::array::from_fn(|r| [Q - 1 - r as u64, r as u64 + 7]);
        let output = gpu_cache.weighted_rows(&words, &row_weights).unwrap();
        for (i, actual) in output.iter().enumerate() {
            let mut expected = [0u128; 2];
            for (row, weight) in words.chunks_exact(output.len()).zip(row_weights) {
                for (value, weight) in expected.iter_mut().zip(weight) {
                    *value = (*value + u128::from(row[i]) * u128::from(weight) % u128::from(Q))
                        % u128::from(Q);
                }
            }
            assert_eq!(*actual, expected.map(|x| x as u64));
        }
        assert_eq!(
            gpu_cache.metal.get().unwrap().as_ref().unwrap().handle,
            handle
        );
        assert_eq!(std::sync::Arc::strong_count(&words), owners);
        assert!(gpu_cache
            .weighted_rows(&std::sync::Arc::new((*words).clone()), &row_weights)
            .is_none());
        assert!(
            gpu_cache.metal.get().is_some_and(Option::is_some),
            "expected a resident Metal key"
        );
    }
    assert!(cache
        .contract::<F, E, RING_D>(VARS, &source[..source.len() - 1], &weights, None)
        .unwrap()
        .is_none());
    assert!(cache
        .contract::<F, E, RING_D>(VARS - 1, &source, &weights, None)
        .is_err());
    assert!(PackingStripes::new(&rings[..rings.len() - 1], POSITIONS).is_none());
}
