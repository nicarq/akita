use super::super::compact_digit_source::CompactDigitSource;
use super::*;
use akita_params::FlatBooleanDomain;
use jolt_field::{Ext2, Prime64Offset59};

type F = Ext2<Prime64Offset59>;

fn point(i: usize) -> F {
    F::new(
        Prime64Offset59::from_u64(7 * i as u64 + 3),
        Prime64Offset59::from_u64(11 * i as u64 + 17),
    )
}

fn check_pages(basis: usize, width: u8, n: usize, tail_zero: bool) {
    let bits = n.next_power_of_two().trailing_zeros() as usize;
    let last_page = (n - 1) / 4096;
    let digits = (0..n)
        .map(|i| {
            let page = i / 4096;
            let zero_page = if page == last_page {
                tail_zero
            } else {
                matches!(page % 13, 0 | 3 | 6..=12)
            };
            if zero_page {
                return -((i & 1) as i8);
            }
            if i % 4096 == 0 || i % 4096 == 4095 {
                return 1;
            }
            ((i * 67 + i / 7) % basis) as i8 - (basis / 2) as i8
        })
        .collect::<Vec<_>>();
    let expected_pages = digits
        .chunks(4096 * 8)
        .map(|tile| {
            tile.chunks(4096)
                .enumerate()
                .fold(0u8, |mask, (page, rows)| {
                    mask | (u8::from(rows.iter().all(|&x| x == 0 || x == -1)) << page)
                })
        })
        .collect::<Vec<_>>();
    let tau: Vec<_> = (0..bits).map(point).collect();
    let mut eq = GruenSplitEq::new(&tau).unwrap();
    let mut table = digits
        .iter()
        .map(|&d| F::from_i64(i64::from(d) * (i64::from(d) + 1)))
        .collect::<Vec<_>>();
    let mut prover = LowBasisRangeCheckProver::new(
        PackedSignedDigits::from_i8_digits(digits, width).unwrap(),
        &tau,
        DigitRangePlan::new(basis).unwrap(),
        n,
        bits,
        0,
    )
    .unwrap();
    assert!(matches!(
        &prover.range_image,
        LowBasisRangeImageStorage::OctetPrefix(_)
    ));
    for round in 0..bits {
        let expected =
            compute_range_round_polynomial_from_range_image(&eq, &prover.range_poly, |i| {
                (
                    table.get(2 * i).copied().unwrap_or_else(F::zero),
                    table.get(2 * i + 1).copied().unwrap_or_else(F::zero),
                )
            });
        assert_eq!(
            prover.compute_round_eq_factored(round, F::zero()),
            expected,
            "basis={basis} width={width} n={n} round={round}"
        );
        let r = point(round + bits);
        prover.ingest_challenge(round, r);
        eq.bind(r);
        table = table
            .chunks(2)
            .map(|v| v[0] + r * (v.get(1).copied().unwrap_or_else(F::zero) - v[0]))
            .collect();
        if round == 3 {
            assert_eq!(prover.zero_pages, expected_pages)
        }
        if let LowBasisRangeImageStorage::Materialized(materialized) = &prover.range_image {
            assert_eq!(
                materialized.as_slice(),
                table.as_slice(),
                "basis={basis} width={width} n={n} round={round}"
            );
        }
    }
    assert_eq!(prover.final_range_image_eval().unwrap(), table[0]);
}

#[test]
fn mixed_and_partial_zero_pages_match_dense_rounds() {
    for width in [2, 3, 4, 5, 6, 7, 8] {
        check_pages(4, width, 32_769, width % 2 == 0);
    }
    check_pages(8, 3, 262_145, true);
    check_pages(8, 5, 266_239, false);
}

#[test]
fn packed_class_pairs_match_signed_digits_and_odd_padding() {
    for basis in [4usize, 8, 16, 32, 64] {
        for width in 1..=8u8 {
            let half = (basis / 2).min(1usize << (width - 1));
            let mut digits = (0..2 * half)
                .flat_map(|a| {
                    (0..2 * half).flat_map(move |b| [a as i8 - half as i8, b as i8 - half as i8])
                })
                .collect::<Vec<_>>();
            for odd in [false, true] {
                if odd {
                    digits.push(-1)
                }
                let bits = digits.len().next_power_of_two().trailing_zeros() as usize;
                let mut source = CompactDigitSource::new(
                    PackedSignedDigits::from_i8_digits(digits.clone(), width).unwrap(),
                    FlatBooleanDomain::new(digits.len(), bits).unwrap(),
                    DigitRangePlan::new(basis).unwrap(),
                )
                .unwrap();
                source.prepare_class_indexed_leaf();
                assert_eq!(source.pair_count(), digits.len().div_ceil(2));
                let class = |d: i8| {
                    if d >= 0 {
                        d as usize
                    } else {
                        (-i16::from(d) - 1) as usize
                    }
                };
                for (i, pair) in digits.chunks(2).enumerate() {
                    let right = pair.get(1).copied().unwrap_or(0);
                    assert_eq!(
                        source.ordered_pair_index(i),
                        class(pair[0]) * (basis / 2) + class(right),
                        "basis={basis} width={width} odd={odd} pair={i}"
                    );
                }
            }
        }
    }
}
