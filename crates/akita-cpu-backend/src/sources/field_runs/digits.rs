use super::FieldRunPoly;
use crate::sources::block_digits::{
    BlockDigits, DigitPlanes, DigitShape, I16Rows, I8Rows, LiveDigitScratch, RingValueSource,
};
use crate::sources::poly_helpers::{I8DigitSpan, SignedDigitBasis};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_params::SignedDigitKernel;
use jolt_field::{CanonicalEncoding, Field};
use std::ops::Range;

pub(crate) struct RunRings<'a, F: Field, const D: usize>(pub(super) &'a FieldRunPoly<F>);

impl<F: Field, const D: usize> RunRings<'_, F, D> {
    pub(crate) fn ring(&self, index: usize) -> CyclotomicRing<F, D> {
        let mut ring = [F::zero(); D];
        self.0.read(index * D, &mut ring);
        CyclotomicRing::from_coefficients(ring)
    }

    pub(crate) fn constant_value(&self, rings: Range<usize>) -> Option<F> {
        self.0.constant_value(rings.start * D..rings.end * D)
    }
}

impl<F: Field, const D: usize> RingValueSource<F, D> for RunRings<'_, F, D> {
    type Values<'a>
        = CyclotomicRing<F, D>
    where
        Self: 'a;

    fn num_rings(&self) -> usize {
        self.0.live_len().div_ceil(D)
    }

    fn values(&self, ring: usize) -> Result<Self::Values<'_>, AkitaError> {
        if ring >= self.num_rings() {
            return Err(AkitaError::InvalidInput(
                "field run ring index is outside the source".into(),
            ));
        }
        Ok(self.ring(ring))
    }
}

pub(crate) struct RunDigits<'a, F: Field + CanonicalEncoding, const D: usize> {
    rings: RunRings<'a, F, D>,
    params: BalancedDecomposePow2Params<F>,
    basis: SignedDigitBasis,
    inverse_digits: Option<Vec<Vec<[i16; 4]>>>,
}

impl<'a, F: Field + CanonicalEncoding, const D: usize> RunDigits<'a, F, D> {
    pub(crate) fn new(poly: &'a FieldRunPoly<F>, shape: DigitShape) -> Result<Self, AkitaError> {
        let basis = SignedDigitBasis::new(shape.log_basis)?;
        let params = BalancedDecomposePow2Params::new(shape.num_digits, shape.log_basis);
        let inverse_digits = match &poly.coefficients {
            super::Coefficients::Columns(codes)
                if shape.num_digits == 4 && shape.log_basis == 16 =>
            {
                codes.digit_tables(|value| {
                    let mut digits = [[0i16; 1]; 4];
                    CyclotomicRing::from_coefficients([value])
                        .balanced_decompose_pow2_i16_into(&mut digits, &params);
                    digits.map(|[digit]| digit)
                })
            }
            _ => None,
        };
        Ok(Self {
            inverse_digits,
            rings: RunRings(poly),
            params,
            basis,
        })
    }

    fn decompose<T: Copy>(
        &self,
        rings: Range<usize>,
        num_digits: usize,
        rows: &mut [[T; D]],
        decompose: impl Fn(&CyclotomicRing<F, D>, &mut [[T; D]], &BalancedDecomposePow2Params<F>),
    ) {
        if let Some(value) = self.rings.constant_value(rings.clone()) {
            let (first, rest) = rows.split_at_mut(num_digits);
            decompose(
                &CyclotomicRing::from_coefficients([value; D]),
                first,
                &self.params,
            );
            for ring in rest.chunks_exact_mut(num_digits) {
                ring.copy_from_slice(first);
            }
        } else {
            for (index, row) in rings.zip(rows.chunks_exact_mut(num_digits)) {
                decompose(&self.rings.ring(index), row, &self.params);
            }
        }
    }
    fn decompose_i16(&self, rings: Range<usize>, num_digits: usize, rows: &mut [[i16; D]]) {
        if let (Some(tables), super::Coefficients::Columns(codes)) =
            (&self.inverse_digits, &self.rings.0.coefficients)
        {
            for (ring, out) in rings.zip(rows.chunks_exact_mut(num_digits)) {
                codes.write_digits(ring * D, tables, out);
            }
            return;
        }
        if let super::Coefficients::Signed(source) = &self.rings.0.coefficients {
            if self.params.log_basis() >= 2 {
                for (index, digits) in rings.zip(rows.chunks_exact_mut(num_digits)) {
                    let (first, rest) = digits.split_first_mut().expect("a digit row is nonempty");
                    if source.read_unit_digits(index * D, first) {
                        rest.fill([0; D]);
                    } else {
                        self.rings
                            .ring(index)
                            .balanced_decompose_pow2_i16_into(digits, &self.params);
                    }
                }
                return;
            }
        }
        self.decompose(
            rings,
            num_digits,
            rows,
            CyclotomicRing::balanced_decompose_pow2_i16_into,
        );
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> BlockDigits<D> for RunDigits<'_, F, D> {
    type Scratch = LiveDigitScratch<D>;
    type Planes<'a>
        = DigitPlanes<'a, D>
    where
        Self: 'a;

    fn num_rings(&self) -> usize {
        self.rings.num_rings()
    }

    fn rings_repeat_within_full_blocks(&self, rings: Range<usize>, block_width: usize) -> bool {
        block_width > 1
            && !rings.is_empty()
            && rings.len().is_multiple_of(block_width)
            && (rings.start..rings.end).step_by(block_width).all(|start| {
                self.rings
                    .constant_value(start..start + block_width)
                    .is_some()
            })
    }

    fn digit_abs_bound(&self, _shape: DigitShape) -> u64 {
        self.basis.abs_bound
    }

    fn i8_digit_span(&self, _shape: DigitShape) -> Option<I8DigitSpan> {
        matches!(self.basis.kernel, SignedDigitKernel::I8).then_some(
            if self.basis.abs_bound <= 64 {
                I8DigitSpan::Half
            } else {
                I8DigitSpan::Full
            },
        )
    }

    fn scratch(&self, shape: DigitShape, max_rings: usize) -> Self::Scratch {
        match self.basis.kernel {
            SignedDigitKernel::I8 => LiveDigitScratch::I8(self.i8_scratch(shape, max_rings)),
            SignedDigitKernel::I16 => LiveDigitScratch::I16(self.i16_scratch(shape, max_rings)),
        }
    }

    fn digit_planes<'a>(
        &'a self,
        ring: usize,
        shape: DigitShape,
        scratch: &'a mut Self::Scratch,
    ) -> Result<Self::Planes<'a>, AkitaError> {
        Ok(match scratch {
            LiveDigitScratch::I8(buffer) => {
                let planes = &mut buffer[..shape.num_digits];
                self.rings
                    .ring(ring)
                    .balanced_decompose_pow2_i8_into_with_params(planes, &self.params);
                DigitPlanes::I8(planes)
            }
            LiveDigitScratch::I16(buffer) => {
                let planes = &mut buffer[..shape.num_digits];
                self.decompose_i16(ring..ring + 1, shape.num_digits, planes);
                DigitPlanes::I16(planes)
            }
        })
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> I8Rows<D> for RunDigits<'_, F, D> {
    type I8Scratch = Vec<[i8; D]>;

    fn i8_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I8Scratch {
        vec![[0; D]; shape.num_digits * max_rings]
    }

    fn digit_abs_bound(&self, _shape: DigitShape) -> u64 {
        self.basis.abs_bound
    }

    fn i8_rows<'a>(
        &'a self,
        _block_index: usize,
        block_start: usize,
        rings: Range<usize>,
        shape: DigitShape,
        scratch: &'a mut Self::I8Scratch,
    ) -> Result<&'a [[i8; D]], AkitaError> {
        let rows = &mut scratch[..rings.len() * shape.num_digits];
        if !rows.is_empty() {
            self.decompose(
                block_start + rings.start..block_start + rings.end,
                shape.num_digits,
                rows,
                CyclotomicRing::balanced_decompose_pow2_i8_into_with_params,
            );
        }
        Ok(rows)
    }
}

impl<F: Field + CanonicalEncoding, const D: usize> I16Rows<D> for RunDigits<'_, F, D> {
    type I16Scratch = Vec<[i16; D]>;

    fn i16_scratch(&self, shape: DigitShape, max_rings: usize) -> Self::I16Scratch {
        vec![[0; D]; shape.num_digits * max_rings]
    }

    fn i16_rows<'a>(
        &'a self,
        rings: Range<usize>,
        shape: DigitShape,
        scratch: &'a mut Self::I16Scratch,
    ) -> Result<&'a [[i16; D]], AkitaError> {
        let (rows, padding) = scratch.split_at_mut(rings.len() * shape.num_digits);
        if !rows.is_empty() {
            self.decompose_i16(rings, shape.num_digits, rows);
        }
        padding.fill([0; D]);
        Ok(scratch)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/lookup_signed_tests.rs"]
mod checked_lookup_tests;
