#[test]
fn compact_full_field_runs_match_flat_packing_reference() {
    type Base = Prime64Offset59;
    type Extension = Ext2<Base>;
    const RING_D: usize = 512;
    const POSITIONS: usize = 32;
    const NUM_VARS: usize = 14;
    let geometry = SubringCoefficientPackingGeometry::try_new(2, RING_D, 128).unwrap();
    let point = (0..NUM_VARS)
        .map(|i| Extension::from_u64(i as u64 + 3))
        .collect::<Vec<_>>();
    let prepared = PreparedSubringCoefficientPackingPoint::new(
        geometry,
        BasisMode::Lagrange,
        POSITIONS,
        4,
        NUM_VARS,
        &point,
    )
    .unwrap();
    let tables = SubringCoefficientPackingTables::new(prepared).unwrap();
    let runs = vec![
        (4 * RING_D, Base::from_u64(u64::MAX - 65)),
        (5 * RING_D + 11, Base::from_u64((1 << 63) + 7)),
        (7 * RING_D, Base::zero()),
        (4 * RING_D, Base::from_u64((1 << 60) + 13)),
    ];
    let mut flat = runs
        .iter()
        .flat_map(|&(len, value)| std::iter::repeat_n(value, len))
        .collect::<Vec<_>>();
    flat.resize(POSITIONS * RING_D, Base::zero());
    let expected = coefficient_packing_partials::<Base, Extension>(
        geometry,
        POSITIONS,
        4,
        &flat,
        tables.position_weights(),
        tables.packing_weights(),
    )
    .unwrap();
    let poly = crate::FieldRunPoly::new(NUM_VARS, runs).unwrap();
    let polys = [&poly];
    let batch =
        <crate::FieldRunPoly<Base> as RootOpeningSource<Base, RING_D>>::opening_batch(&polys)
            .unwrap();
    let got = CpuBackend::<Base, Extension>::for_arithmetic_tests()
        .coefficient_packing_partials_batch(
            None,
            batch,
            SubringCoefficientPackingPlan { tables: &tables },
        )
        .unwrap();
    assert_eq!(got[0].coordinates(), expected);
}

#[test]
fn signed_packed_contraction_preserves_field_partials() {
    type Base = Prime64Offset59;
    type Extension = Ext2<Base>;
    const RING_D: usize = 512;
    const POSITIONS: usize = 35;
    const NUM_VARS: usize = 15;
    let source_len = POSITIONS * RING_D - 19;
    let digits = (0..source_len)
        .map(|index| {
            if (index / RING_D).is_multiple_of(4) {
                0
            } else {
                (index.wrapping_mul(21) as i8).wrapping_sub(83)
            }
        })
        .collect::<Vec<_>>();
    let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone())
        .align_for_commitment_ring_dim(RING_D)
        .unwrap();
    let mut source = digits.into_iter().map(Base::from_i8).collect::<Vec<_>>();
    source.resize(POSITIONS * RING_D, Base::zero());
    let point = (0..NUM_VARS)
        .map(|i| {
            Extension::new(
                Base::from_u64(0x8765_4567_319a_21ef_u64.wrapping_shl(i as u32)),
                Base::from_u64(0x7654_0b32_8943_72a0_u64.rotate_right(i as u32)),
            )
        })
        .collect::<Vec<_>>();
    for subring in [128, 256] {
        let geometry = SubringCoefficientPackingGeometry::try_new(2, RING_D, subring).unwrap();
        let tables = PreparedSubringCoefficientPackingPoint::new(
            geometry,
            BasisMode::Lagrange,
            POSITIONS,
            16,
            NUM_VARS,
            &point,
        )
        .and_then(SubringCoefficientPackingTables::new)
        .unwrap();
        let expected = coefficient_packing_partials::<Base, Extension>(
            geometry,
            POSITIONS,
            16,
            &source,
            tables.position_weights(),
            tables.packing_weights(),
        )
        .unwrap();
        let refs = [&witness];
        let batch = <RecursiveWitnessFlat as RootOpeningSource<Base, RING_D>>::opening_batch(&refs)
            .unwrap();
        let received = CpuBackend::<Base, Extension>::for_arithmetic_tests()
            .coefficient_packing_partials_batch(
                None,
                batch,
                SubringCoefficientPackingPlan { tables: &tables },
            )
            .unwrap();
        assert_eq!(received[0].coordinates(), expected);
    }
}
