use super::{digits::RunRings, FieldRunPoly, RunDigits};
use crate::arithmetic::coefficient_packing::{
    coefficient_packing_partials_from_position_source, FusedPackingWeights,
};
use crate::opaque::{
    aggregate_decompose_fold_witnesses, CpuBackend, CpuFoldResponses, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, DecomposeFoldWitness, OpeningBatchKernel, OpeningFoldKernel,
    OpeningFoldOutput, OpeningFoldPlan, RootOpeningSource, RootPolyShape,
    SubringCoefficientPackingBatchKernel, SubringCoefficientPackingPartials,
    SubringCoefficientPackingPlan,
};
use crate::sources::block_digits::evaluate::{evaluate_and_fold_base, evaluate_and_fold_ring};
use crate::sources::block_digits::{DigitShape, RingValueSource};
use crate::sources::poly_helpers::element_partitioned_decompose_fold;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};
use std::ops::Range;

pub struct RunBatchView<'a, F: Field, const D: usize>(&'a [&'a FieldRunPoly<F>]);

impl<F: Field, const D: usize> RootPolyShape<F, D> for FieldRunPoly<F> {
    fn num_ring_elems(&self) -> usize {
        self.logical_len().div_ceil(D)
    }

    fn num_vars(&self) -> usize {
        self.num_vars()
    }
}

impl<F: Field, const D: usize> RootOpeningSource<F, D> for FieldRunPoly<F> {
    type OpeningView<'a>
        = &'a Self
    where
        Self: 'a;
    type OpeningBatchView<'a>
        = RunBatchView<'a, F, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        if D == 0 || !D.is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "field run ring dimension must be a power of two".into(),
            ));
        }
        Ok(self)
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        for poly in polys {
            <Self as RootOpeningSource<F, D>>::opening_view(poly)?;
        }
        Ok(RunBatchView(polys))
    }
}

fn fold<F: Field + CanonicalEncoding, const D: usize>(
    source: &FieldRunPoly<F>,
    challenges: &[akita_challenges::SparseChallenge],
    ranges: &[Range<usize>],
    positions: usize,
    shape: DigitShape,
) -> Result<Vec<DecomposeFoldWitness>, AkitaError> {
    if shape.num_digits == 0 {
        return Err(AkitaError::InvalidInput(
            "field run fold digit count must be nonzero".into(),
        ));
    }
    let digits = RunDigits::<F, D>::new(source, shape)?;
    Ok(
        element_partitioned_decompose_fold(&digits, challenges, ranges, positions, shape)?
            .into_iter()
            .map(DecomposeFoldWitness::from_centered_rows)
            .collect(),
    )
}

impl<F, E, const D: usize> OpeningFoldKernel<&FieldRunPoly<F>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: &FieldRunPoly<F>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let positions = plan.num_positions_per_block();
        if positions == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".into(),
            ));
        }
        let blocks =
            <FieldRunPoly<F> as RootPolyShape<F, D>>::num_ring_elems(source).div_ceil(positions);
        plan.validate::<D>(blocks)?;
        let rings = RunRings::<F, D>(source);
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                live_block_weights,
                position_weights,
                ..
            } => evaluate_and_fold_base(
                &rings,
                live_block_weights,
                position_weights,
                positions,
                blocks,
            )?,
            OpeningFoldPlan::Subfield { multipliers, .. } => evaluate_and_fold_ring(
                &rings,
                &multipliers.materialize_fold_rings::<D>()?,
                &multipliers.materialize_position_rings::<D>()?,
                positions,
                blocks,
            )?,
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: &FieldRunPoly<F>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness, AkitaError> {
        let positions = plan.num_positions_per_block;
        if positions == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".into(),
            ));
        }
        let blocks =
            <FieldRunPoly<F> as RootPolyShape<F, D>>::num_ring_elems(source).div_ceil(positions);
        if plan.challenges.len() != blocks {
            return Err(AkitaError::InvalidSize {
                expected: blocks,
                actual: plan.challenges.len(),
            });
        }
        let shape = DigitShape {
            num_digits: plan.num_digits,
            log_basis: plan.log_basis,
        };
        fold::<F, D>(
            source,
            plan.challenges,
            std::slice::from_ref(&(0..blocks)),
            positions,
            shape,
        )?
        .pop()
        .ok_or_else(|| AkitaError::InvalidInput("field run fold produced no chunk".into()))
    }
}

impl<F, E, const D: usize> OpeningBatchKernel<RunBatchView<'_, F, D>, F, D> for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold_batch(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source: RunBatchView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<Vec<OpeningFoldOutput<F, D>>, AkitaError> {
        source
            .0
            .iter()
            .map(|poly| self.evaluate_and_fold(prepared, *poly, plan))
            .collect()
    }

    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: RunBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<CpuFoldResponses, AkitaError> {
        let (positions, num_digits, log_basis) = plan.scalar_params();
        let shape = DigitShape {
            num_digits,
            log_basis,
        };
        let blocks = plan.validate_uniform_batch(source.0.iter().map(|poly| {
            <FieldRunPoly<F> as RootPolyShape<F, D>>::num_ring_elems(poly).div_ceil(positions)
        }))?;
        let full = 0..blocks;
        let (challenges, ranges) = match &plan {
            DecomposeFoldBatchPlan::Sparse { challenges, .. } => {
                (*challenges, std::slice::from_ref(&full))
            }
            DecomposeFoldBatchPlan::SparseChunked {
                challenges,
                chunk_ranges,
                ..
            } => (challenges.as_slice(), *chunk_ranges),
        };
        let mut by_chunk = (0..ranges.len())
            .map(|_| Vec::with_capacity(source.0.len()))
            .collect::<Vec<_>>();
        for (poly, poly_challenges) in source.0.iter().zip(challenges.chunks_exact(blocks)) {
            for (chunk, witness) in by_chunk.iter_mut().zip(fold::<F, D>(
                poly,
                poly_challenges,
                ranges,
                positions,
                shape,
            )?) {
                chunk.push(Ok(witness));
            }
        }
        let mut folded = by_chunk
            .into_iter()
            .map(aggregate_decompose_fold_witnesses::<D>)
            .collect::<Result<Vec<_>, _>>()?;
        match plan {
            DecomposeFoldBatchPlan::Sparse { .. } => {
                Ok(CpuFoldResponses::sparse(folded.pop().ok_or_else(|| {
                    AkitaError::InvalidInput("field run fold produced no chunk".into())
                })?))
            }
            DecomposeFoldBatchPlan::SparseChunked { .. } => CpuFoldResponses::chunked::<D>(folded),
        }
    }
}

impl<F, E, const D: usize> SubringCoefficientPackingBatchKernel<RunBatchView<'_, F, D>, F, E, D>
    for CpuBackend<F, E>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: RunBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let point = plan.tables.point();
        let weights = FusedPackingWeights::new(plan.tables)?;
        source
            .0
            .iter()
            .map(|poly| {
                let logical_rings = <FieldRunPoly<F> as RootPolyShape<F, D>>::num_ring_elems(poly);
                if logical_rings != point.num_live_positions() {
                    return Err(AkitaError::InvalidSize {
                        expected: point.num_live_positions(),
                        actual: logical_rings,
                    });
                }
                let rings = RunRings::<F, D>(poly);
                let coordinates = coefficient_packing_partials_from_position_source::<F, E, _, D>(
                    &weights,
                    FieldRunPoly::<F>::num_vars(poly),
                    rings.num_rings(),
                    None,
                    |block| rings.constant_value(block),
                    |position| rings.values(position),
                    |_, coefficient, value| value.coefficients()[coefficient],
                )?;
                SubringCoefficientPackingPartials::new(
                    point.geometry(),
                    point.num_live_blocks(),
                    coordinates,
                )
            })
            .collect()
    }
}
