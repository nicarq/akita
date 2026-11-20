use super::CpuPreparedSetup;
use crate::sources::block_digits::DigitShape;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};

pub(super) fn fold_matrix_columns<F: Field, const D: usize>(
    matrix: &[CyclotomicRing<F, D>],
    rows: usize,
    width: usize,
    period: usize,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..rows * period)
        .map(|slot| {
            let row = slot / period;
            let first = slot % period;
            let mut sum = CyclotomicRing::<F, D>::zero();
            for source in matrix[row * width..(row + 1) * width]
                .iter()
                .skip(first)
                .step_by(period)
            {
                for (coefficient, &value) in
                    sum.coefficients_mut().iter_mut().zip(source.coefficients())
                {
                    *coefficient += value;
                }
            }
            sum
        })
        .collect()
}

pub(super) fn commit_constant_blocks<F: Field + CanonicalEncoding, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    n_a: usize,
    row_width: usize,
    shape: DigitShape,
    values: &[F],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    if values.iter().all(|value| value.is_zero()) {
        return Ok(vec![vec![CyclotomicRing::zero(); n_a]; values.len()]);
    }
    let matrix = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D>(n_a, row_width)?;
    let mut columns = fold_matrix_columns(matrix.as_slice(), n_a, row_width, shape.num_digits);
    for column in &mut columns {
        let sum = column.coefficients().iter().copied().sum::<F>();
        let mut prefix = F::zero();
        for coefficient in column.coefficients_mut() {
            prefix += *coefficient;
            *coefficient = prefix + prefix - sum;
        }
    }
    let params = BalancedDecomposePow2Params::new(shape.num_digits, shape.log_basis);
    let mut digits = vec![[0i16; 1]; shape.num_digits];
    let mut blocks = Vec::with_capacity(values.len());
    for &value in values {
        let mut rows = vec![CyclotomicRing::zero(); n_a];
        if !value.is_zero() {
            CyclotomicRing::from_coefficients([value])
                .balanced_decompose_pow2_i16_into(&mut digits, &params);
            for (row, group) in rows.iter_mut().zip(columns.chunks_exact(shape.num_digits)) {
                for (column, &[digit]) in group.iter().zip(&digits) {
                    column.scale_accumulate_into(row, F::from_i64(i64::from(digit)));
                }
            }
        }
        blocks.push(rows);
    }
    Ok(blocks)
}
