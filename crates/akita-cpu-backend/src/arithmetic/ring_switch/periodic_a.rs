use akita_algebra::CyclotomicRing;
use jolt_field::solinas::parallel::*;
use jolt_field::Field;

const MAX_PERIOD: usize = 8;
const REPEATS_PER_CHECK: usize = 64;

pub(super) fn checked_row_period<const D: usize>(rows: &[[i32; D]]) -> Option<usize> {
    let prefix = rows.get(..2 * MAX_PERIOD)?;
    let period = (1..=MAX_PERIOD).find(|&period| {
        rows.len().is_multiple_of(period) && prefix[period..] == prefix[..prefix.len() - period]
    })?;
    let template = &rows[..period];
    cfg_chunks!(rows, period * REPEATS_PER_CHECK)
        .all(|chunk| chunk.chunks_exact(period).all(|rows| rows == template))
        .then_some(period)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::linear::fused_split_eq_quotients_streamed_prover_bounds;
    use jolt_field::{Prime64Offset59 as F, Ring};

    #[test]
    fn periodic_quotient_matches_full_matrix_and_checks_every_row() {
        const D: usize = 64;
        const PERIOD: usize = 5;
        const WIDTH: usize = 25;
        const RANK: usize = 3;

        let z: Vec<[i32; D]> = (0..WIDTH)
            .map(|j| std::array::from_fn(|i| ((i + (j % PERIOD) * 121) % 751) as i32 - 374))
            .collect();
        let matrix: Vec<CyclotomicRing<F, D>> = (0..RANK * WIDTH)
            .map(|j| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
                    F::from_u64(((j * D + i) as u64).wrapping_mul(0xf3a5_c817_b283_9321))
                }))
            })
            .collect();
        assert_eq!(checked_row_period(&z), Some(PERIOD));
        let full =
            fused_split_eq_quotients_streamed_prover_bounds(&matrix, 0, RANK, &[], &z, 750, 4)
                .unwrap();
        let collapsed = fold_matrix_columns(&matrix, RANK, WIDTH, PERIOD);
        let grouped = fused_split_eq_quotients_streamed_prover_bounds(
            &collapsed,
            0,
            RANK,
            &[],
            &z[..PERIOD],
            750,
            4,
        )
        .unwrap();
        assert_eq!(grouped.a_quotients, full.a_quotients);

        let mut invalid_period = z;
        invalid_period[WIDTH - 1][D - 1] ^= 1;
        assert_eq!(checked_row_period(&invalid_period), None);
    }
}

#[cfg(test)]
mod tail_tests {
    use super::*;
    use crate::kernels::linear::small_field_matrix_a_quotients_with_tail;
    use jolt_field::{CanonicalEncoding, Prime64Offset59 as F, Ring};

    #[test]
    fn collapsed_tail_matches_direct_integer_convolution() {
        const D: usize = 64;
        const PERIOD: usize = 5;
        const Q: i128 = (1i128 << 64) - 59;

        let matrix: Vec<CyclotomicRing<F, D>> = (0..PERIOD)
            .map(|row| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
                    F::from_u64(((row * D + i) as u64).wrapping_mul(0xbc3c_748d_e2f0_d3eb))
                }))
            })
            .collect();
        let rhs: Vec<[i32; D]> = (0..PERIOD)
            .map(|row| std::array::from_fn(|i| ((i + row * 41) as i32 - 100) << 21))
            .collect();
        let mut coefficients = [0i128; D];
        for (a, z) in matrix.iter().zip(&rhs) {
            for (i, value) in a.coefficients().iter().enumerate() {
                let a = value.to_u128_checked().unwrap() as i128;
                for (output, z) in coefficients.iter_mut().zip(&z[D - i..]) {
                    *output += a * i128::from(*z);
                }
            }
        }
        let expected = coefficients.map(|x| F::from_u64(x.rem_euclid(Q) as u64));
        let result = small_field_matrix_a_quotients_with_tail(&matrix, 1, &rhs, 0, 4)
            .unwrap()
            .unwrap();
        assert_eq!(result[0].coefficients(), &expected);
    }
}
