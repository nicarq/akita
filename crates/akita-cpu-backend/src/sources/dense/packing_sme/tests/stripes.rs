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
    if !available() {
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
    let partials = cache
        .contract::<F, E, RING_D>(VARS, &source, &weights)
        .unwrap()
        .unwrap();
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
    assert_eq!(partials, expected);
    assert!(cache
        .contract::<F, E, RING_D>(VARS, &source[..source.len() - 1], &weights)
        .unwrap()
        .is_none());
    assert!(cache
        .contract::<F, E, RING_D>(VARS - 1, &source, &weights)
        .is_err());
    assert!(PackingStripes::new(&rings[..rings.len() - 1], POSITIONS).is_none());
}
