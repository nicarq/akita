//! Owned run-length encoded field coefficients.

use crate::commitment::{
    AvailablePolynomialTypes, CommitSourceClass, CommitSourceDescriptor, CommitmentSource,
    DenseRepresentation, DenseType, PolynomialRepresentation, PolynomialType,
    PolynomialTypeSelection,
};
use crate::opaque::{CommitInnerPlan, RootPolyMeta};
use crate::sources::poly::SourceCoefficients;
use akita_error::{checked, AkitaError};
use akita_params::SUPPORTED_COMMITMENT_RING_DIMS;
use jolt_field::{CanonicalEncoding, Field};
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

mod columns;
mod digits;
mod kernels;
mod signed_span;
pub(crate) use digits::RunDigits;

/// A multilinear polynomial stored as consecutive runs of field coefficients.
///
/// The run values follow the same coefficient order as [`crate::DensePoly`].
/// Coefficients after the last run are zero. The standard commitment and root
/// opening kernels consume this representation without expanding its domain.
/// Tensor openings and transfers to other backends can materialize up to
/// 16 MiB of coefficients.
#[derive(Debug)]
pub struct FieldRunPoly<F: Field> {
    num_vars: usize,
    coefficients: Coefficients<F>,
}

#[derive(Debug)]
enum Coefficients<F> {
    Runs(Vec<(usize, F)>),
    Columns(columns::CodeColumns<F>),
    Signed(signed_span::SignedSpan<F>),
}

impl<F: Field> FieldRunPoly<F> {
    /// Own `(length, value)` runs covering a prefix of the Boolean domain.
    pub fn new(num_vars: usize, mut runs: Vec<(usize, F)>) -> Result<Self, AkitaError> {
        let domain = checked::pow2(num_vars).ok_or_else(|| {
            AkitaError::InvalidInput("field run domain size overflows usize".into())
        })?;
        let mut end = 0usize;
        for (len, _) in &mut runs {
            if *len == 0 {
                return Err(AkitaError::InvalidInput("field run is empty".into()));
            }
            end = end
                .checked_add(*len)
                .filter(|&next| next <= domain)
                .ok_or_else(|| AkitaError::InvalidInput("field runs exceed their domain".into()))?;
            *len = end;
        }
        runs.truncate(
            runs.iter()
                .rposition(|(_, value)| !value.is_zero())
                .map_or(0, |i| i + 1),
        );
        Ok(Self {
            num_vars,
            coefficients: Coefficients::Runs(runs),
        })
    }

    pub fn from_lookup_columns(
        num_vars: usize,
        column_width: usize,
        codes: Arc<Vec<u16>>,
        tables: Vec<Vec<F>>,
    ) -> Result<Self, AkitaError> {
        let domain = checked::pow2(num_vars).ok_or_else(|| {
            AkitaError::InvalidInput("field lookup domain size overflows usize".into())
        })?;
        Ok(Self {
            num_vars,
            coefficients: Coefficients::Columns(columns::CodeColumns::new(
                domain,
                column_width,
                codes,
                tables,
            )?),
        })
    }

    pub fn from_signed_span(
        num_vars: usize,
        start: usize,
        positive: Vec<u64>,
        negative: Vec<u64>,
        tail: (usize, F),
    ) -> Result<Self, AkitaError> {
        let domain = checked::pow2(num_vars).ok_or_else(|| {
            AkitaError::InvalidInput("signed coefficient domain exceeds usize".into())
        })?;
        Ok(Self {
            num_vars,
            coefficients: Coefficients::Signed(signed_span::SignedSpan::new(
                domain, start, positive, negative, tail,
            )?),
        })
    }

    pub(crate) const fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub(crate) fn live_len(&self) -> usize {
        match &self.coefficients {
            Coefficients::Runs(ends) => ends.last().map_or(0, |&(end, _)| end),
            Coefficients::Columns(columns) => columns.live_len(),
            Coefficients::Signed(span) => span.live_len(),
        }
    }

    pub(crate) fn logical_len(&self) -> usize {
        1 << self.num_vars
    }

    pub(crate) fn read(&self, start: usize, out: &mut [F]) {
        let ends = match &self.coefficients {
            Coefficients::Runs(ends) => ends,
            Coefficients::Columns(columns) => return columns.read(start, out),
            Coefficients::Signed(span) => return span.read(start, out),
        };
        let mut cursor = start;
        let mut remaining = out;
        let first = ends.partition_point(|&(end, _)| end <= start);
        for &(end, value) in &ends[first..] {
            let count = (end - cursor).min(remaining.len());
            let (run, rest) = remaining.split_at_mut(count);
            run.fill(value);
            remaining = rest;
            if remaining.is_empty() {
                return;
            }
            cursor += count;
        }
        remaining.fill(F::zero());
    }

    #[cfg(akita_fold_metal)]
    pub(crate) fn lookup_columns(&self) -> Option<columns::LookupColumns<'_, F>> {
        match &self.coefficients {
            Coefficients::Columns(columns) => Some(columns.view()),
            _ => None,
        }
    }

    pub(crate) fn constant_value(&self, range: Range<usize>) -> Option<F> {
        let ends = match &self.coefficients {
            Coefficients::Runs(ends) => ends,
            Coefficients::Columns(columns) => return columns.constant(range),
            Coefficients::Signed(span) => return span.constant(range),
        };
        let first = ends.partition_point(|&(end, _)| end <= range.start);
        match ends.get(first) {
            None => Some(F::zero()),
            Some(&(end, value)) if range.end <= end => Some(value),
            _ => None,
        }
    }
}

impl<F: Field> RootPolyMeta<F> for FieldRunPoly<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Field> CommitmentSource<F> for FieldRunPoly<F> {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        let logical = self.logical_len();
        let minimum = SUPPORTED_COMMITMENT_RING_DIMS[SUPPORTED_COMMITMENT_RING_DIMS.len() - 1];
        CommitSourceDescriptor::new(
            self.num_vars,
            logical.max(minimum),
            logical,
            CommitSourceClass::Dense,
            "akita_field_runs",
        )
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding,
    {
        let mut reach = (0, 0);
        let mut include = |value: F| -> Result<(), AkitaError> {
            let canonical = value.to_u128_checked().ok_or_else(|| {
                AkitaError::InvalidInput("field run value must fit in u128".into())
            })?;
            if canonical <= centering_threshold {
                reach.1 = reach.1.max(canonical);
            } else {
                let negative = modulus.checked_sub(canonical).ok_or_else(|| {
                    AkitaError::InvalidInput("field run value exceeds the modulus".into())
                })?;
                reach.0 = reach.0.max(negative);
            }
            Ok(())
        };
        match &self.coefficients {
            Coefficients::Runs(ends) => ends.iter().try_for_each(|&(_, value)| include(value))?,
            Coefficients::Columns(columns) => columns.used_values().try_for_each(include)?,
            Coefficients::Signed(span) => span.used_values().try_for_each(include)?,
        }
        Ok(reach)
    }

    fn available_polynomial_types(
        &self,
        _plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        AvailablePolynomialTypes::new(vec![PolynomialType::Dense(DenseType::Coefficients)])
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        if selected.polynomial_type() != PolynomialType::Dense(DenseType::Coefficients) {
            return Err(AkitaError::InvalidInput(
                "field runs received an unadvertised representation selection".into(),
            ));
        }
        Ok(PolynomialRepresentation::Dense(
            DenseRepresentation::FieldRuns(self),
        ))
    }
}

impl<F: Field> SourceCoefficients<F> for FieldRunPoly<F> {
    fn source_coefficients(&self) -> Result<Cow<'_, [F]>, AkitaError> {
        const MAX_MATERIALIZED_BYTES: usize = 16 * 1024 * 1024;
        let len = self.logical_len();
        if len > MAX_MATERIALIZED_BYTES / std::mem::size_of::<F>().max(1) {
            return Err(AkitaError::InvalidInput(
                "large field runs require a streaming opening schedule".into(),
            ));
        }
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(len).map_err(|_| {
            AkitaError::InvalidInput("field run coefficient allocation failed".into())
        })?;
        coefficients.resize(len, F::zero());
        self.read(0, &mut coefficients);
        Ok(Cow::Owned(coefficients))
    }
}
