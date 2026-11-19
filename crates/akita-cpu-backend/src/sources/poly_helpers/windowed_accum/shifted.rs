use std::arch::aarch64::*;

const TILE: usize = 32;

pub(super) struct ShiftedWindows {
    dimension: usize,
    pairs: Vec<[u16; 2]>,
    last: Option<u16>,
}

impl ShiftedWindows {
    pub(super) fn new<const D: usize>(
        positions: &[u32],
        coefficients: &[i8],
        digit_abs_bound: u64,
    ) -> Option<Self> {
        if !matches!(D, 128 | 512)
            || digit_abs_bound > 63
            || !(16..=41).contains(&positions.len())
            || positions.len() != coefficients.len()
            || positions.iter().any(|&position| position >= D as u32)
            || coefficients
                .iter()
                .any(|&coefficient| !matches!(coefficient, -1 | 1))
        {
            return None;
        }
        let offset = |position: u32, coefficient: i8| {
            ((1 + usize::from(coefficient < 0)) * D - position as usize) as u16
        };
        let pairs = positions
            .chunks_exact(2)
            .zip(coefficients.chunks_exact(2))
            .map(|(position, coefficient)| {
                [
                    offset(position[0], coefficient[0]),
                    offset(position[1], coefficient[1]),
                ]
            })
            .collect();
        let last = (!positions.len().is_multiple_of(2)).then(|| {
            offset(
                positions[positions.len() - 1],
                coefficients[coefficients.len() - 1],
            )
        });
        Some(Self {
            dimension: D,
            pairs,
            last,
        })
    }

    #[inline]
    pub(super) fn add<const D: usize>(&self, plane: &[i8; D], acc: &mut [i16; D]) {
        assert_eq!(D, self.dimension);
        if plane
            .as_chunks::<16>()
            .0
            .iter()
            .all(|window| *window == [0; 16])
        {
            return;
        }
        let negated = plane.map(i8::wrapping_neg);
        let signed = [negated, *plane, negated];
        let mut tiles = acc.chunks_exact_mut(TILE);
        for (tile_index, tile) in tiles.by_ref().enumerate() {
            // SAFETY: The constructor caps offsets at 2D. The tile ends at
            // or before D, all source loads fit, and tile has 32 output cells.
            unsafe {
                let source = signed.as_flattened().as_ptr().add(tile_index * TILE);
                self.add_tile(source, tile.as_mut_ptr());
            }
        }
        debug_assert!(tiles.into_remainder().is_empty());
    }

    #[inline(always)]
    unsafe fn add_tile(&self, source: *const i8, output: *mut i16) {
        unsafe {
            let mut acc0 = vld1q_s16(output);
            let mut acc1 = vld1q_s16(output.add(8));
            let mut acc2 = vld1q_s16(output.add(16));
            let mut acc3 = vld1q_s16(output.add(24));
            for &[left, right] in &self.pairs {
                let left = source.add(usize::from(left));
                let right = source.add(usize::from(right));
                let first = vaddq_s8(vld1q_s8(left), vld1q_s8(right));
                let second = vaddq_s8(vld1q_s8(left.add(16)), vld1q_s8(right.add(16)));
                acc0 = vaddw_s8(acc0, vget_low_s8(first));
                acc1 = vaddw_high_s8(acc1, first);
                acc2 = vaddw_s8(acc2, vget_low_s8(second));
                acc3 = vaddw_high_s8(acc3, second);
            }
            if let Some(last) = self.last {
                let last = source.add(usize::from(last));
                let first = vld1q_s8(last);
                let second = vld1q_s8(last.add(16));
                acc0 = vaddw_s8(acc0, vget_low_s8(first));
                acc1 = vaddw_high_s8(acc1, first);
                acc2 = vaddw_s8(acc2, vget_low_s8(second));
                acc3 = vaddw_high_s8(acc3, second);
            }
            vst1q_s16(output, acc0);
            vst1q_s16(output.add(8), acc1);
            vst1q_s16(output.add(16), acc2);
            vst1q_s16(output.add(24), acc3);
        }
    }
}
