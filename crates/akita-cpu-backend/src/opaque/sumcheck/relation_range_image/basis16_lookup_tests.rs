use super::append_basis16_table;
use jolt_field::{Ext2, Prime64Offset59, Ring, Zero};

#[test]
fn separable_tables_match_every_balanced_digit_class() {
    type F = Prime64Offset59;
    type E = Ext2<F>;
    let x = |a: u64, b: u64| E::new(F::from_u64(a), F::from_u64(b));
    let weights = [x(9, 52), -x(4, 3), x(5, 123), -x(50, 18)];
    let scales = [E::zero(), x(1, 0), x(15, 10), -x(59, 201)];
    let mut table = Vec::new();
    for scale in scales {
        append_basis16_table(&mut table, weights.map(|weight| scale * weight));
    }
    assert_eq!(table.len(), scales.len() << 16);
    for (&scale, row) in scales.iter().zip(table.chunks_exact(1 << 16)) {
        for (class, &value) in row.iter().enumerate() {
            let direct = weights
                .iter()
                .enumerate()
                .fold(E::zero(), |sum, (i, &weight)| {
                    sum + weight * E::from_i64(((class >> (4 * i)) & 15) as i64 - 8)
                });
            assert_eq!(value, scale * direct);
        }
    }
}
