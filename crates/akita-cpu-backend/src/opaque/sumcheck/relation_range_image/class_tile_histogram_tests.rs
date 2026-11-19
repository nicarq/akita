use super::*;
use jolt_field::{Ext2, Prime64Offset59};

#[test]
fn class_tiles_match_pair_histograms_for_repeated_and_mixed_blocks() {
    type E = Ext2<Prime64Offset59>;
    let points: Vec<E> = (0..19)
        .map(|i| {
            E::new(
                Prime64Offset59::from_u64(i * 15),
                Prime64Offset59::from_u64(i * 19),
            )
        })
        .collect();
    let eq = GruenSplitEq::new(&points).unwrap();
    let (low, high) = eq.remaining_eq_tables_after(2).unwrap();
    let (tiles, common_high) = eq
        .remaining_eq_tables_after(2 + HISTOGRAM_LOG_TILE)
        .unwrap();
    assert_eq!(high, common_high);
    assert_eq!(low.len(), tiles.len() * HISTOGRAM_TILE);
    for (segment, &sum) in low.chunks_exact(HISTOGRAM_TILE).zip(tiles) {
        assert_eq!(sum, segment.iter().copied().fold(E::zero(), |a, b| a + b));
    }

    let classes: Vec<[u16; 2]> = (0..HISTOGRAM_TILE * 7)
        .map(|i| match (i / HISTOGRAM_TILE) % 3 {
            0 => [8, 2],
            1 => [2, 9],
            _ => [(i % 8 + 1) as u16, (i * 3 % 9 + 1) as u16],
        })
        .collect();
    let offset = 18;
    let code = [0, 2, 4, 5, 6, 7, 8, 12, 15, 18];
    let first_pair = low.len() - HISTOGRAM_TILE;
    let mut actual = [
        vec![E::zero(); 10],
        vec![E::zero(); 10],
        vec![E::zero(); 2 * offset as usize + 1],
    ];
    let mut expected = actual.clone();
    let [even, odd, delta] = &mut actual;
    add_class_tiles(
        &classes,
        first_pair,
        [low, high, tiles],
        &code,
        offset,
        &mut [even, odd, delta],
    );
    for (i, &[a, b]) in classes.iter().enumerate() {
        let (a, b) = (usize::from(a), usize::from(b));
        let pos = first_pair + i;
        let weight = low[pos % low.len()] * high[pos / low.len()];
        expected[0][a] += weight;
        expected[1][b] += weight;
        expected[2][(offset + code[b] - code[a]) as usize] += weight;
    }
    assert_eq!(actual, expected);
}

#[test]
fn packed_zero_lanes_preserve_fold_inputs_and_relation_moments() {
    type E = Ext2<Prime64Offset59>;
    const LANES: usize = 784;
    const COEFF: usize = 128;
    const CLASS: usize = 4096;
    const DELTA: usize = 15usize.pow(4);
    let witness = PackedSignedDigits::from_i8_digits(
        (0..LANES * COEFF)
            .map(|i| {
                let lane = i / COEFF;
                if (5..261).contains(&lane) || (272..768).contains(&lane) && lane % 3 != 0 {
                    0
                } else {
                    ((i * 11 + i / 21) % 8) as i8 - 4
                }
            })
            .collect(),
        3,
    )
    .unwrap();
    let lane_weights = (0..LANES)
        .map(|i| {
            E::new(
                Prime64Offset59::from_u64(i as u64 * 71 + 1),
                Prime64Offset59::from_u64(i as u64 * 19),
            )
        })
        .collect::<Vec<_>>();
    let linear = PreparedProverLinearTerms::from_response_norm_factors(
        vec![E::from_u64(616); COEFF],
        lane_weights.clone(),
        LANES,
        COEFF,
    )
    .unwrap();
    let point: Vec<_> = (1..=23).map(|i| E::from_u64(i * 37)).collect();
    let eq = GruenSplitEq::new(&point).unwrap();
    let (eq_low, eq_high) = eq.remaining_eq_tables_after(2).unwrap();
    let (eq_tile, _) = eq
        .remaining_eq_tables_after(2 + HISTOGRAM_LOG_TILE)
        .unwrap();
    let delta_code: Vec<_> = (0..CLASS)
        .map(|i| {
            (0..4)
                .rev()
                .fold(0, |acc, d| acc * 15 + ((i >> (3 * d)) & 7) as i32)
        })
        .collect();
    let base = ScanLayout {
        witness: witness.view(),
        packed_lanes: false,
        lane_weights: &lane_weights,
        linear_terms: &linear,
        source_blocks: &[0],
        source_block_count: 1,
        eq_low,
        eq_high,
        eq_tile: Some(eq_tile),
        tau2: point[2],
        b: 8,
        coeff_count: COEFF,
        class_count: CLASS,
        delta_code: &delta_code,
        delta_offset: (DELTA / 2) as i32,
        delta_count: DELTA,
    };
    let mut expected_classes = vec![0; (LANES - 5) * COEFF / 4];
    let expected = scan_lanes::<E, 3>(&base, 5, &mut expected_classes);
    let packed = ScanLayout {
        packed_lanes: true,
        ..base
    };
    let mut classes = vec![0; expected_classes.len()];
    let actual = scan_lanes::<E, 3>(&packed, 5, &mut classes);
    assert_eq!(classes, expected_classes);
    assert_eq!(actual.alpha_mass, expected.alpha_mass);
    assert_eq!(actual.source_mass, expected.source_mass);
    assert_eq!(actual.even_histogram, expected.even_histogram);
    assert_eq!(actual.odd_histogram, expected.odd_histogram);
    assert_eq!(actual.delta_histogram, expected.delta_histogram);
}
