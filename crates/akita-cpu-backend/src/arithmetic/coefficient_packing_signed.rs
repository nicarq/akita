use super::coefficient_packing::zero_vec;
use crate::sources::packed_digits::PackedSignedDigitView;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, Prime64Offset59};
use std::any::TypeId;

#[derive(Clone, Copy)]
pub(crate) struct PackedCoefficientSource<'a> {
    pub(crate) digits: PackedSignedDigitView<'a>,
    pub(crate) live_coeff_len: usize,
}

impl PackedCoefficientSource<'_> {
    pub(super) fn contract<F, E, const D: usize>(
        self,
        first: usize,
        positions: usize,
        stride: usize,
        weights: &[E],
    ) -> Result<Option<Vec<E>>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        if TypeId::of::<F>() != TypeId::of::<Prime64Offset59>()
            || E::DEGREE != 2
            || !D.is_multiple_of(16)
        {
            return Ok(None);
        }
        let bounds = self.digits.bounds();
        let magnitude = usize::from(bounds.negative_abs_max().max(bounds.positive_max()));
        if positions.saturating_mul(stride).saturating_mul(magnitude) > u16::MAX.into() {
            return Ok(None);
        }
        match stride {
            2 => self
                .contract_stride::<F, E, D, 2>(first, positions, weights)
                .map(Some),
            4 => self
                .contract_stride::<F, E, D, 4>(first, positions, weights)
                .map(Some),
            _ => Ok(None),
        }
    }

    fn contract_stride<F, E, const D: usize, const STRIDE: usize>(
        self,
        first: usize,
        positions: usize,
        weights: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        F: Field + CanonicalEncoding,
        E: ExtField<F>,
    {
        let mut sums = Vec::new();
        sums.try_reserve_exact(D / STRIDE).map_err(|_| {
            AkitaError::InvalidInput("coefficient-packing signed product allocation failed".into())
        })?;
        sums.resize(D / STRIDE, [0i32; 8]);
        let mut ring = [0i8; D];
        for (i, row) in weights
            .as_chunks::<STRIDE>()
            .0
            .iter()
            .take(positions)
            .enumerate()
        {
            let start = (first + i) * D;
            self.digits.decode_range(start, &mut ring)?;
            ring[(self.live_coeff_len - start).min(D)..].fill(0);
            if ring
                .as_chunks::<16>()
                .0
                .iter()
                .all(|&chunk| chunk == [0; 16])
            {
                continue;
            }
            let weights = row.each_ref().map(|value| {
                let [a, b] = [0, 1].map(|coordinate| {
                    let canonical = value.base_coefficient(coordinate).to_u64_checked().unwrap();
                    balanced_words(canonical)
                });
                [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
            });
            for (total, source) in sums.iter_mut().zip(ring.as_chunks::<STRIDE>().0) {
                for (weight, &digit) in weights.iter().zip(source) {
                    for (output, &word) in total.iter_mut().zip(weight) {
                        *output += i32::from(digit) * i32::from(word);
                    }
                }
            }
        }
        let mut result = zero_vec::<E>(sums.len())?;
        for (packed, words) in result.iter_mut().zip(sums) {
            let coordinates = words.as_chunks::<4>().0;
            *packed =
                E::from_base_fn(|coordinate| F::from_i128(reduce_words(&coordinates[coordinate])));
        }
        Ok(result)
    }
}

fn balanced_words(value: u64) -> [i16; 4] {
    const PRIME: i128 = Prime64Offset59::CHALLENGE_SET_LIMBS[0] as i128;
    const MAX_POSITIVE: u64 = 0x7fff_7fff_7fff_7fff;
    let mut rest = i128::from(value);
    if value > MAX_POSITIVE {
        rest -= PRIME;
    }
    std::array::from_fn(|_| {
        let word = rest as i16;
        rest = (rest - i128::from(word)) >> 16;
        word
    })
}

fn reduce_words(words: &[i32; 4]) -> i128 {
    words
        .iter()
        .rev()
        .fold(0i128, |acc, &word| (acc << 16) + i128::from(word))
}
