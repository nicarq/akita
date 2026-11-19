use jolt_field::{CanonicalEncoding, Ext2, ExtField, Field, Prime64Offset59, Ring, Zero};
use std::any::Any;
use std::arch::aarch64::*;

pub(super) type Weight = Ext2<Prime64Offset59>;
type Base = Prime64Offset59;
const TILE: usize = 8;
const ROW_BATCH: usize = 256;
const MIN_ROWS_FOR_NEON: usize = 16;

pub(super) fn add_rows<E: Field + 'static>(
    masses: &mut [Weight],
    half: usize,
    rows: &[(E, u32)],
    digits: &[i8],
    use_neon: bool,
) {
    if !use_neon || rows.len() < MIN_ROWS_FOR_NEON || !masses.len().is_multiple_of(TILE) {
        if half <= 4 {
            add_table_rows::<E, true>(masses, half, rows, digits);
        } else {
            add_table_rows::<E, false>(masses, half, rows, digits);
        }
        return;
    }
    let width = masses.len();
    let mut factors = [[0i16; 8]; ROW_BATCH];
    let mut offsets = [0usize; ROW_BATCH];
    for group in rows.chunks(ROW_BATCH) {
        for ((factor, offset), &(value, row)) in factors.iter_mut().zip(&mut offsets).zip(group) {
            let value = weight(&value);
            let [first, second] = [0, 1].map(|i| limbs(value.base_coefficient(i)));
            let (left, right) = factor.split_at_mut(4);
            left.copy_from_slice(&first);
            right.copy_from_slice(&second);
            *offset = row as usize * width;
            assert!(*offset + width <= digits.len());
        }
        // SAFETY: the caller selected NEON. Each row was checked above and
        // the output uses full eight-coefficient tiles.
        unsafe {
            add_tiles(
                masses,
                &factors[..group.len()],
                &offsets[..group.len()],
                digits,
            );
        }
    }
}

#[inline(always)]
fn weight<E: Field + 'static>(value: &E) -> Weight {
    *(value as &dyn Any)
        .downcast_ref()
        .expect("the direct mass path checks the field type")
}

fn add_table_rows<E: Field + 'static, const SKIP_ZERO: bool>(
    masses: &mut [Weight],
    half: usize,
    rows: &[(E, u32)],
    digits: &[i8],
) {
    let width = masses.len();
    let mut multiples = [Weight::zero(); 32];
    for (factor, &row) in rows.iter().map(|(value, row)| (weight(value), row)) {
        for offset in 1..=half {
            if half + offset < 2 * half {
                multiples[half + offset] = multiples[half + offset - 1] + factor;
            }
            multiples[half - offset] = multiples[half + 1 - offset] - factor;
        }
        let start = row as usize * width;
        for (mass, &digit) in masses.iter_mut().zip(&digits[start..start + width]) {
            if !SKIP_ZERO || digit != 0 {
                let index = (i16::from(digit) + half as i16) as usize & 31;
                *mass += multiples[index];
            }
        }
    }
}

#[inline(always)]
fn limbs(value: Base) -> [i16; 4] {
    const Q: i128 = (1i128 << 64) - 59;
    const THRESHOLD: i128 = i16::MAX as i128 * (u64::MAX as i128 / u16::MAX as i128);
    let mut integer = value.to_u128_checked().unwrap() as i128;
    if integer > THRESHOLD {
        integer -= Q;
    }
    std::array::from_fn(|_| {
        let limb = integer as i16;
        integer = (integer - i128::from(limb)) >> 16;
        limb
    })
}

#[inline(always)]
fn reduce([a, b, c, d]: [i32; 4]) -> Base {
    Base::from_i128(
        i128::from(a) + (i128::from(b) << 16) + (i128::from(c) << 32) + (i128::from(d) << 48),
    )
}

#[inline]
unsafe fn add_tiles(masses: &mut [Weight], factors: &[[i16; 8]], offsets: &[usize], digits: &[i8]) {
    for (tile, masses) in masses.chunks_exact_mut(TILE).enumerate() {
        let mut sums = [[vdupq_n_s32(0); 2]; TILE];
        for (row, &offset) in factors.iter().zip(offsets) {
            let factor = vld1q_s16(row.as_ptr());
            let low = vget_low_s16(factor);
            let digits = vmovl_s8(vld1_s8(digits.as_ptr().add(offset + tile * TILE)));
            // Each signed limb sum has magnitude at most 256 * 32768 * 16.
            macro_rules! accumulate {
                ($i:literal) => {
                    sums[$i][0] = vmlal_laneq_s16::<$i>(sums[$i][0], low, digits);
                    sums[$i][1] = vmlal_high_laneq_s16::<$i>(sums[$i][1], factor, digits);
                };
            }
            accumulate!(0);
            accumulate!(1);
            accumulate!(2);
            accumulate!(3);
            accumulate!(4);
            accumulate!(5);
            accumulate!(6);
            accumulate!(7);
        }
        for (mass, [left, right]) in masses.iter_mut().zip(sums) {
            let mut a = [0i32; 4];
            let mut b = [0i32; 4];
            vst1q_s32(a.as_mut_ptr(), left);
            vst1q_s32(b.as_mut_ptr(), right);
            *mass += Weight::new(reduce(a), reduce(b));
        }
    }
}

#[cfg(test)]
#[path = "p64_mass/tests/signed_limbs.rs"]
mod tests;
