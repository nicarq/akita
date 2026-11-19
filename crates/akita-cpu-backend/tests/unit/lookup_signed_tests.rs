use super::{DigitShape, FieldRunPoly, RunDigits};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use jolt_field::{Prime64Offset59 as F, Ring, Zero};
use std::sync::Arc;

fn check<const D: usize>(
    width: usize,
    codes: Vec<u16>,
    tables: Vec<Vec<F>>,
    count: usize,
    basis: u32,
) {
    let size = 256usize;
    let mut cells = vec![F::zero(); size];
    for (column, table) in tables.iter().enumerate() {
        for (offset, &code) in codes.iter().enumerate() {
            cells[column * width + offset] = table
                .get(usize::from(code))
                .copied()
                .unwrap_or_else(F::zero);
        }
    }
    let source = FieldRunPoly::from_lookup_columns(8, width, Arc::new(codes), tables).unwrap();
    let parameters = BalancedDecomposePow2Params::new(count, basis);
    let digits = RunDigits::<F, D>::new(
        &source,
        DigitShape {
            num_digits: count,
            log_basis: basis,
        },
    )
    .unwrap();
    let mut expected = vec![[0i16; D]; count];
    let mut projected = vec![[i16::MIN; D]; count];
    for ring in 0..size.div_ceil(D) {
        let words =
            std::array::from_fn(|i| cells.get(ring * D + i).copied().unwrap_or_else(F::zero));
        CyclotomicRing::from_coefficients(words)
            .balanced_decompose_pow2_i16_into(&mut expected, &parameters);
        projected.fill([i16::MIN; D]);
        digits.decompose_i16(ring..ring + 1, count, &mut projected);
        assert_eq!(projected, expected, "ring {ring}");
    }
}

#[test]
fn inverse_codes_match_dense_digits_across_columns_and_padding() {
    let numbers = [0u64, 32768, 65535, 65536, 1 << 63, u64::MAX - 60];
    let tables = (0..5)
        .map(|i| {
            numbers
                .iter()
                .cycle()
                .skip(i)
                .take(numbers.len())
                .map(|&n| F::from_u64(n))
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let codes = vec![u16::MAX, 0, 1, 2, 3, 4, 5, 4, 1, u16::MAX, 3];
    check::<8>(32, codes.clone(), tables.clone(), 4, 16);
    check::<64>(32, codes.clone(), tables.clone(), 4, 16);
    check::<8>(32, codes.clone(), tables.clone(), 8, 8);
    check::<64>(32, Vec::new(), tables, 4, 16);
}
