use super::{lookup_pairs, lookup_pairs_by, PackedQuadPairs};
use crate::sources::packed_digits::PackedSignedDigits;
use jolt_field::{Ext2, ExtField, Prime64Offset59, Ring, Zero};

type F = Prime64Offset59;
pub(super) type E = Ext2<F>;

fn value(i: usize) -> E {
    E::from_base_fn(|j| {
        F::from_u64((i as u64 + 19).wrapping_mul(0x9e37_79b9_7f4a_7c15u64 ^ (j as u64)))
    })
}

pub(super) fn inputs<const G: usize>(pairs: usize) -> (Vec<E>, Vec<E>, Vec<E>) {
    let table: Vec<_> = (0..4096 * G).map(value).collect();
    let low: Vec<_> = (0..64).map(|i| value(31 + i)).collect();
    let high: Vec<_> = (0..pairs.div_ceil(low.len()))
        .map(|i| value(104 + i))
        .collect();
    (table, low, high)
}
fn check<const G: usize, const SKIP: bool>(witness: &PackedSignedDigits, classes: &[u16]) {
    let raw = PackedQuadPairs::new(witness.view()).unwrap();
    let pairs = classes.len() / (2 * G);
    let (table, low, high) = inputs::<G>(pairs);
    let mut a = vec![E::zero(); 2 * pairs];
    let mut b = a.clone();
    let want =
        lookup_pairs::<E, G, SKIP>(&table, 4096, classes, &low, &high, 0..pairs, Some(&mut a));
    let fold = |entry: usize| {
        (0..G)
            .map(|j| table[j * 4096 + usize::from(raw.at(entry * G + j))])
            .sum()
    };
    let got = lookup_pairs_by::<E, G, SKIP>(
        4096,
        fold,
        |s, e| raw.uniform::<G>(s, e),
        &low,
        &high,
        0..pairs,
        Some(&mut b),
    );
    assert_eq!(got, want);
    assert_eq!(b, a);
}

pub(super) fn witness_classes() -> (PackedSignedDigits, Vec<u16>) {
    let digits: Vec<i8> = (0..410 * 128)
        .map(|i| {
            if i / 512 % 3 == 0 {
                0
            } else {
                (((i * 29 + i / 31) % 8) as i8) - 4
            }
        })
        .collect();
    let classes: Vec<u16> = digits
        .chunks_exact(4)
        .map(|quad| {
            quad.iter()
                .enumerate()
                .fold(0, |c, (i, &x)| c | (((x + 4) as u16) << (3 * i)))
        })
        .collect();
    (PackedSignedDigits::from_i8_digits_auto(digits), classes)
}

#[test]
fn packed_cpu_fallback_matches_expanded_lookup_and_materialization() {
    let (witness, classes) = witness_classes();
    check::<2, true>(&witness, &classes);
    check::<2, false>(&witness, &classes);
    check::<4, true>(&witness, &classes);
    check::<4, false>(&witness, &classes);
    check::<8, true>(&witness, &classes);
    check::<8, false>(&witness, &classes);
}
