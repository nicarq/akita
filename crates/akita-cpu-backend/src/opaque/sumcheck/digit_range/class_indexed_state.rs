use super::compact_digit_source::CompactDigitSource;
use super::exact_prefix::ExactPrefixTable;
const MIN_ROWS_PER_PAIR_CLASS: usize = 64;

pub(super) struct FoldedQuartetRows<Row> {
    half_len: usize,
    values: Vec<Row>,
}

impl<Row: Copy> FoldedQuartetRows<Row> {
    pub(super) fn new(
        source: &CompactDigitSource,
        zero: Row,
        pair_row: impl Fn(usize) -> Row,
        fold: impl Fn(Row, Row) -> Row,
    ) -> Option<Self> {
        let half_len = source.class_count().pow(2);
        if source.quartet_count() < MIN_ROWS_PER_PAIR_CLASS * half_len {
            return None;
        }
        let mut values = Vec::with_capacity(2 * half_len);
        values.extend((0..half_len).map(|i| fold(pair_row(i), zero)));
        values.extend((0..half_len).map(|i| fold(zero, pair_row(i))));
        Some(Self { half_len, values })
    }

    #[inline(always)]
    pub(super) fn row(
        &self,
        source: &CompactDigitSource,
        quartet: usize,
        add: impl Fn(Row, Row) -> Row,
    ) -> Row {
        let (left, right) = source.ordered_pair_indices_for_quartet(quartet);
        add(self.values[left], self.values[self.half_len + right])
    }
}

/// Shared storage lifecycle for class-indexed early rounds and exact-prefix later rounds.
pub(super) enum ClassIndexedTableState<Compact, FirstChallengeFolded, Row: Copy> {
    Compact(Compact),
    FirstChallengeFolded(FirstChallengeFolded),
    Materialized(ExactPrefixTable<Row>),
}

impl<Compact, FirstChallengeFolded, Row: Copy>
    ClassIndexedTableState<Compact, FirstChallengeFolded, Row>
{
    pub(super) fn final_value(&self) -> Option<Row> {
        match self {
            Self::Materialized(table) => table.final_value(),
            Self::Compact(_) | Self::FirstChallengeFolded(_) => None,
        }
    }
}

#[cfg(test)]
#[path = "class_indexed_state_tests.rs"]
mod tests;
