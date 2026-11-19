use super::*;
use crate::opaque::sumcheck::relation_range_image::quotient_prefix::lookup_pairs;
use jolt_field::{Ext2, One, Prime64Offset59, Ring, Zero};

#[test]
fn class_norms_equal_indexed_norms_for_mixed_pairs_and_partial_equality_blocks() {
    type F = Prime64Offset59;
    type E = Ext2<F>;
    let field = |a: u64, b: u64| E::new(F::from_u64(a), F::from_u64(b));
    let r = field(13, 31);
    let quad_factors = [r, r, field(7, 11), E::one() - r - r - field(7, 11)];
    let weights = [field(17, 22), E::one() - field(17, 22)];
    let lookups: Vec<E> = weights
        .into_iter()
        .flat_map(|weight| {
            (0..CLASS_COUNT).map(move |class| {
                weight
                    * quad_factors
                        .iter()
                        .enumerate()
                        .map(|(i, &factor)| {
                            factor * E::from_i64(((class >> (3 * i)) & 7) as i64 - 4)
                        })
                        .fold(E::zero(), |left, right| left + right)
            })
        })
        .collect();
    let classes: Vec<u16> = (0..4 * 1137usize)
        .map(|i| {
            let n = ((i * 73) ^ (i / 7)) as u16;
            if i / 4 % 13 == 0 {
                n & ((CLASS_COUNT - 1) as u16)
            } else {
                (n & 7) * 0x249
            }
        })
        .collect();
    let eq_low: Vec<_> = (0..512).map(|i| field(i + 1, 7 * i + 9)).collect();
    let eq_high = [field(1, 4), field(3, 8), field(17, 7)];
    let pairs = 39..1137;
    assert!(ConstantQuadNorms::<E>::supports(&classes, &pairs));
    let expected = lookup_pairs::<E, 2, true>(
        &lookups,
        CLASS_COUNT,
        &classes,
        &eq_low,
        &eq_high,
        pairs.clone(),
        None,
    );
    let actual = ConstantQuadNorms::new(weights).sum(&lookups, &classes, &eq_low, &eq_high, pairs);
    assert_eq!(actual, expected);
}
