use super::{Field, ProductSum, Unreduced};
use std::ops::Range;

const CLASS_COUNT: usize = 1 << 12;
const BIASED_ZERO_PAIR: usize = 4 | (4 << 3);

pub(super) struct ConstantQuadNorms<E> {
    left: [E; 64],
    delta: [E; 225],
}

impl<E: Field + Unreduced> ConstantQuadNorms<E> {
    pub(super) fn new([a, b]: [E; 2]) -> Self {
        let left = std::array::from_fn(|i| {
            let value = a * E::from_i64((i % 8) as i64 - 4) + b * E::from_i64((i / 8) as i64 - 4);
            value.square() + value
        });
        let delta = std::array::from_fn(|i| {
            (a * E::from_i64((i % 15) as i64 - 7) + b * E::from_i64((i / 15) as i64 - 7)).square()
        });
        Self { left, delta }
    }

    pub(super) fn supports(classes: &[u16], pairs: &Range<usize>) -> bool {
        pairs.len() >= 512
            && (0..16)
                .filter(|i| {
                    let pair = pairs.start + pairs.len() * i / 16;
                    let row = &classes[4 * pair..][..4];
                    constant_pair_code(row).is_some() && constant_pair_code(&row[2..]).is_some()
                })
                .count()
                >= 8
    }

    pub(super) fn sum(
        &self,
        lookups: &[E],
        classes: &[u16],
        eq_low: &[E],
        eq_high: &[E],
        pairs: Range<usize>,
    ) -> [E; 3] {
        let (left_values, right_values) = lookups.split_at(CLASS_COUNT);
        let low_mask = eq_low.len() - 1;
        let low_bits = eq_low.len().trailing_zeros();
        let mut totals = [E::zero(); 3];
        let mut start = pairs.start;
        while start < pairs.end {
            let high = start >> low_bits;
            let end = ((high + 1) << low_bits).min(pairs.end);
            let local = start & low_mask;
            let mut left_bins = [E::zero(); 64];
            let mut delta_bins = [E::zero(); 225];
            let [mut left, mut delta] = [ProductSum::<E>::zero(); 2];
            for (row, &weight) in classes[4 * start..4 * end]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(&eq_low[local..local + end - start])
            {
                if let (Some(l), Some(r)) = (constant_pair_code(row), constant_pair_code(&row[2..]))
                {
                    if l != BIASED_ZERO_PAIR {
                        left_bins[l] += weight;
                    }
                    if l != r {
                        let lo = (r & 7) + 7 - (l & 7);
                        let hi = (r >> 3) + 7 - (l >> 3);
                        delta_bins[lo + 15 * hi] += weight;
                    }
                } else {
                    let [a, b, c, d] = row.map(usize::from);
                    let w0 = left_values[a] + right_values[b];
                    let w1 = left_values[c] + right_values[d];
                    left.add(weight, w0.square() + w0);
                    delta.add(weight, (w1 - w0).square());
                }
            }
            for (&norm, weight) in self.left.iter().zip(left_bins) {
                if !weight.is_zero() {
                    left.add(weight, norm);
                }
            }
            for (&norm, weight) in self.delta.iter().zip(delta_bins) {
                if !weight.is_zero() {
                    delta.add(weight, norm);
                }
            }
            totals[0] += eq_high[high] * left.finish();
            totals[2] += eq_high[high] * delta.finish();
            start = end;
        }
        totals
    }
}

#[inline(always)]
fn constant_pair_code(classes: &[u16]) -> Option<usize> {
    let packed = u32::from(classes[0]) | (u32::from(classes[1]) << 16);
    let digits = packed & 0x0007_0007;
    (digits * 0x249 == packed).then_some(((digits | (digits >> 13)) & 63) as usize)
}

#[cfg(test)]
#[path = "constant_quad_norms/tests.rs"]
mod tests;
