use super::*;
use crate::sources::packed_digits::PackedSignedDigits;
use akita_params::{DigitRangePlan, FlatBooleanDomain};
use jolt_field::{Ext2, Fold, Prime64Offset59, Ring, Zero};

#[test]
fn factored_quartets_match_direct_folds_with_padded_digits() {
    type F = Prime64Offset59;
    type E = Ext2<F>;
    for basis in [16usize, 32, 64] {
        let classes = basis / 2;
        let live = 4 * MIN_ROWS_PER_PAIR_CLASS * classes * classes + 1;
        let log_domain = live.next_power_of_two().trailing_zeros() as usize;
        let digits = (0..live)
            .map(|i| (((i * 13) ^ (i >> 4)) & (basis - 1)) as i8 - classes as i8)
            .collect();
        let source = CompactDigitSource::new(
            PackedSignedDigits::from_i8_digits_auto(digits),
            FlatBooleanDomain::new(live, log_domain).unwrap(),
            DigitRangePlan::new(basis).unwrap(),
        )
        .unwrap();
        let row = |i| {
            [
                E::from_u64(i as u64 + 3),
                E::new(F::from_u64(i as u64 + 7), F::from_u64(i as u64 + 11)),
            ]
        };
        let r = E::new(F::from_u64(17), F::from_u64(19));
        let context = E::precompute(r);
        let fold =
            |a: [E; 2], b: [E; 2]| std::array::from_fn(|i| E::fold_one(&context, a[i], b[i]));
        let table = FoldedQuartetRows::new(&source, [E::zero(); 2], row, fold).unwrap();
        for i in 0..source.quartet_count() {
            let (a, b) = source.ordered_pair_indices_for_quartet(i);
            assert_eq!(
                table.row(&source, i, |a, b| [a[0] + b[0], a[1] + b[1]]),
                fold(row(a), row(b)),
            );
        }
    }
}
