use super::exact_prefix::SplitEqualitySuffixMass;
use jolt_field::solinas::parallel::*;
use jolt_field::Unreduced;
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Field, Prime64Offset59, Zero};
use std::any::{Any, TypeId};

/// Sum `N` products over the explicit pairs of a split-equality round.
///
/// This is the only traversal of the split-equality blocks: exact-prefix
/// reduction, canonical or delayed accumulation, outer weighting and the
/// parallel reduction all live here.
///
/// `pair_terms(pair_index, inner_weight)` returns two factor arrays whose
/// lane-wise products are the pair's terms, already scaled by its inner
/// equality weight. Each block of the inner table is summed first, then scaled
/// by its outer weight. Implicit pairs contribute nothing here, so callers use
/// this directly only for terms that vanish on a constant pair; see
/// [`accumulate_equality_weighted_values`] for terms that do not.
pub(super) fn accumulate_equality_weighted_pair_terms<E: Field + Unreduced, const N: usize>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    pair_terms: impl Fn(usize, E) -> ([E; N], [E; N]) + Sync,
) -> [E; N] {
    debug_assert!(explicit_pair_count <= first.len() * second.len());
    let explicit_block_count = explicit_pair_count.div_ceil(first.len());
    if E::SUM_IS_EXACT {
        cfg_fold_reduce!(
            0..explicit_block_count,
            || [E::Product::zero(); N],
            |mut outer, second_index| {
                let block_start = second_index * first.len();
                let block_end = explicit_pair_count.min(block_start + first.len());
                let mut inner = [E::Product::zero(); N];
                for pair_index in block_start..block_end {
                    let (factors, weighted) =
                        pair_terms(pair_index, first[pair_index - block_start]);
                    for ((destination, factor), weighted) in
                        inner.iter_mut().zip(factors).zip(weighted)
                    {
                        *destination += factor.mul_unreduced(weighted);
                    }
                }
                let second_weight = second[second_index];
                for (destination, inner) in outer.iter_mut().zip(inner) {
                    *destination += second_weight.mul_unreduced(E::reduce_product(inner));
                }
                outer
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            }
        )
        .map(E::reduce_product)
    } else {
        cfg_fold_reduce!(
            0..explicit_block_count,
            || [E::zero(); N],
            |mut outer, second_index| {
                let block_start = second_index * first.len();
                let block_end = explicit_pair_count.min(block_start + first.len());
                let mut inner = [E::zero(); N];
                for pair_index in block_start..block_end {
                    let (factors, weighted) =
                        pair_terms(pair_index, first[pair_index - block_start]);
                    for ((destination, factor), weighted) in
                        inner.iter_mut().zip(factors).zip(weighted)
                    {
                        *destination += factor * weighted;
                    }
                }
                let second_weight = second[second_index];
                for (destination, inner) in outer.iter_mut().zip(inner) {
                    *destination += second_weight * inner;
                }
                outer
            },
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            }
        )
    }
}

/// Sum per-pair values over a split-equality round.
///
/// Explicit pairs contribute `values_at(pair_index)` weighted by their
/// equality weight; every implicit pair contributes `default_values`.
///
/// The explicit part is [`accumulate_equality_weighted_pair_terms`] with the
/// pair's values as the first factor and its inner equality weight as the
/// second. That function counts implicit pairs as zero, so the total equality
/// mass of the implicit suffix times `default_values` is added on top here.
pub(super) fn accumulate_equality_weighted_values<E: Field + Unreduced, const N: usize>(
    first: &[E],
    second: &[E],
    explicit_pair_count: usize,
    values_at: impl Fn(usize) -> [E; N] + Sync,
    default_values: [E; N],
) -> [E; N] {
    let mut values = accumulate_equality_weighted_pair_terms(
        first,
        second,
        explicit_pair_count,
        |pair_index, weight| (values_at(pair_index), [weight; N]),
    );
    let suffix_weight = SplitEqualitySuffixMass::new(first, second)
        .and_then(|suffix| suffix.weight_from(explicit_pair_count))
        .expect("split equality and exact prefix were validated at construction");
    for (value, default) in values.iter_mut().zip(default_values) {
        *value += suffix_weight * default;
    }
    values
}

/// Explicit entries per class below which a block-local histogram would spend
/// more on scaling its classes than on multiplying each entry's weight.
const MIN_BLOCK_ENTRIES_PER_CLASS: usize = 8;

/// Sum class-indexed values over a split-equality round.
///
/// Equivalent to [`accumulate_equality_weighted_values`] with
/// `values_at(index) = class_values(class_at(index))`, for values that depend
/// on an entry only through a class below `class_count`. The equality weight is
/// first totalled per class and each class's values are multiplied once, so the
/// traversal costs at most one multiplication per entry instead of `N`. When
/// blocks are long relative to `class_count` the per-entry work is additions
/// only: a block's inner weights are binned by class and the bins are scaled by
/// the block's outer weight.
pub(super) fn accumulate_equality_weighted_class_values<E: Field + 'static, const N: usize>(
    first: &[E],
    second: &[E],
    explicit_count: usize,
    class_count: usize,
    class_at: impl Fn(usize) -> usize + Sync,
    class_values: impl Fn(usize) -> [E; N],
    default_values: [E; N],
) -> [E; N] {
    debug_assert!(explicit_count <= first.len() * second.len());
    let block_count = explicit_count.div_ceil(first.len());
    let task_count = super::super::parallel_tasks(2).min(block_count).max(1);
    let blocks_per_task = block_count.div_ceil(task_count).max(1);
    let bin_blocks = first.len() >= class_count * MIN_BLOCK_ENTRIES_PER_CLASS;
    let wide_bins = bin_blocks
        && class_count <= 64
        && TypeId::of::<E>() == TypeId::of::<Ext2<Prime64Offset59>>();
    let class_weights = cfg_fold_reduce!(
        0..task_count,
        || vec![E::zero(); class_count],
        |mut weights: Vec<E>, task_index| {
            let blocks =
                task_index * blocks_per_task..block_count.min((task_index + 1) * blocks_per_task);
            let mut block_weights = vec![
                E::zero();
                if bin_blocks && !wide_bins {
                    class_count
                } else {
                    0
                }
            ];
            let mut wide_weights = vec![[0u128; 2]; if wide_bins { class_count } else { 0 }];
            for second_index in blocks {
                let block_start = second_index * first.len();
                let block_end = explicit_count.min(block_start + first.len());
                let second_weight = second[second_index];
                if wide_bins {
                    accumulate_p64_class_block(
                        &first[..block_end - block_start],
                        second_weight,
                        &mut weights,
                        &mut wide_weights,
                        |index| class_at(block_start + index),
                    );
                } else if bin_blocks {
                    block_weights.fill(E::zero());
                    for index in block_start..block_end {
                        block_weights[class_at(index)] += first[index - block_start];
                    }
                    for (weight, block_weight) in weights.iter_mut().zip(&block_weights) {
                        *weight += second_weight * *block_weight;
                    }
                } else {
                    for index in block_start..block_end {
                        weights[class_at(index)] += second_weight * first[index - block_start];
                    }
                }
            }
            weights
        },
        |mut left: Vec<E>, right: Vec<E>| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        }
    );
    let suffix_weight = SplitEqualitySuffixMass::new(first, second)
        .and_then(|suffix| suffix.weight_from(explicit_count))
        .expect("split equality and exact prefix were validated at construction");
    let mut values = default_values.map(|default| suffix_weight * default);
    for (class, weight) in class_weights.into_iter().enumerate() {
        if weight.is_zero() {
            continue;
        }
        for (value, class_value) in values.iter_mut().zip(class_values(class)) {
            *value += weight * class_value;
        }
    }
    values
}

fn accumulate_p64_class_block<E: Field + 'static>(
    first: &[E],
    second: E,
    weights: &mut [E],
    wide_weights: &mut [[u128; 2]],
    class_at: impl Fn(usize) -> usize,
) {
    type Base = Prime64Offset59;
    type WideField = Ext2<Base>;
    wide_weights.fill([0, 0]);
    for (index, value) in first.iter().enumerate() {
        let value = (value as &dyn Any)
            .downcast_ref::<WideField>()
            .expect("caller selected the P64 quadratic extension");
        let sum = &mut wide_weights[class_at(index)];
        for (coordinate, slot) in sum.iter_mut().enumerate() {
            *slot += value
                .base_coefficient(coordinate)
                .to_u128_checked()
                .unwrap();
        }
    }
    for (weight, &[re, im]) in weights.iter_mut().zip(wide_weights.iter()) {
        let reduced = WideField::new(Base::solinas_reduce(re), Base::solinas_reduce(im));
        let reduced = *(&reduced as &dyn Any)
            .downcast_ref::<E>()
            .expect("caller selected the P64 quadratic extension");
        *weight += second * reduced;
    }
}

#[cfg(test)]
mod tests {
    include!("round_accumulation/tests/wide_bins.rs");
    use super::super::MAX_TREE_STAGE_Q_DEGREE;
    use super::*;
    use jolt_field::{FpExt4, Prime128Offset275, Prime32Offset99, Ring};

    fn check_blocked_accumulation<E: Field + Ring + Unreduced + 'static>() {
        // Inner tables below, at and above the block-binning threshold for
        // four classes.
        for first_len in [4, 16, 32, 64] {
            check_blocked_accumulation_with_inner_len::<E>(first_len);
        }
    }

    fn check_blocked_accumulation_with_inner_len<E: Field + Ring + Unreduced + 'static>(
        first_len: usize,
    ) {
        // A zero weight and a cancelling pair sit among generic weights.
        let mut first = (0..first_len)
            .map(|index| E::from_u64((index * index + 3 * index + 2) as u64))
            .collect::<Vec<_>>();
        first[1] = E::zero();
        first[3] = E::zero() - first[2];
        let second = [11, 13, 17, 19].map(E::from_u64);
        let rows = (0..first.len() * second.len())
            .map(|row| {
                std::array::from_fn(|coefficient| {
                    E::from_u64((row * 7 + coefficient * 3 + 1) as u64)
                })
            })
            .collect::<Vec<_>>();
        let default = [23, 29, 31, 37, 41].map(E::from_u64);

        for explicit_pair_count in 0..=rows.len() {
            let actual = accumulate_equality_weighted_values(
                &first,
                &second,
                explicit_pair_count,
                |pair_index| rows[pair_index],
                default,
            );
            let mut expected = [E::zero(); MAX_TREE_STAGE_Q_DEGREE + 1];
            for pair_index in 0..rows.len() {
                let row = if pair_index < explicit_pair_count {
                    rows[pair_index]
                } else {
                    default
                };
                let weight = first[pair_index % first.len()] * second[pair_index / first.len()];
                for (expected, row) in expected.iter_mut().zip(row) {
                    *expected += weight * row;
                }
            }
            assert_eq!(actual, expected);

            // Few classes bin each block; many classes weight each entry.
            for class_count in [1, 3, 4, rows.len()] {
                let class_rows = &rows[..class_count];
                let class_at = |index: usize| (index * 5 + 2) % class_count;
                assert_eq!(
                    accumulate_equality_weighted_class_values(
                        &first,
                        &second,
                        explicit_pair_count,
                        class_count,
                        class_at,
                        |class| class_rows[class],
                        default,
                    ),
                    accumulate_equality_weighted_values(
                        &first,
                        &second,
                        explicit_pair_count,
                        |index| class_rows[class_at(index)],
                        default,
                    )
                );
            }
        }
    }

    #[test]
    fn blocked_accumulation_matches_dense_for_canonical_fallback() {
        check_blocked_accumulation::<Prime128Offset275>();
    }

    #[test]
    fn blocked_accumulation_matches_dense_with_delayed_reduction() {
        type E = FpExt4<Prime32Offset99>;
        check_blocked_accumulation::<E>();
    }
}
