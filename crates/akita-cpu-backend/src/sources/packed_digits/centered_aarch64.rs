use core::arch::aarch64::*;

use super::SignedDigitBounds;

#[target_feature(enable = "neon")]
pub(super) unsafe fn encode<const D: usize>(
    rows: &[[i32; D]],
    levels: usize,
    width: u8,
    output: &mut [u8],
) -> SignedDigitBounds {
    match width {
        3 => encode_width::<D, 3, 2>(rows, levels, output),
        4 => encode_width::<D, 4, 3>(rows, levels, output),
        _ => unreachable!("checked centered encoding width"),
    }
}

#[target_feature(enable = "neon")]
fn step<const B: i32, const SIGN: i32>(carry: &mut int32x4_t) -> uint32x4_t {
    let raw = vandq_u32(vreinterpretq_u32_s32(*carry), vdupq_n_u32((1 << B) - 1));
    *carry = vaddq_s32(
        vshrq_n_s32::<B>(*carry),
        vreinterpretq_s32_u32(vshrq_n_u32::<SIGN>(raw)),
    );
    raw
}

#[target_feature(enable = "neon")]
fn encode_width<const D: usize, const B: i32, const SIGN: i32>(
    rows: &[[i32; D]],
    levels: usize,
    output: &mut [u8],
) -> SignedDigitBounds {
    let stride = D / 8 * B as usize;
    let mut min = vdupq_n_s8(0);
    let mut max = min;
    let half = vdupq_n_u8(1 << SIGN);
    // SAFETY: the table contains one full 16-byte vector.
    let gather = unsafe {
        vld1q_u8(
            [
                0, 1, 2, 8, 9, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ]
            .as_ptr(),
        )
    };
    for (row, output) in rows.iter().zip(output.chunks_exact_mut(stride * levels)) {
        for (index, values) in row.chunks_exact(16).enumerate() {
            // SAFETY: every load reads four elements within this 16-coefficient group.
            let (mut a, mut b, mut c, mut d) = unsafe {
                let ptr = values.as_ptr();
                (
                    vld1q_s32(ptr),
                    vld1q_s32(ptr.add(4)),
                    vld1q_s32(ptr.add(8)),
                    vld1q_s32(ptr.add(12)),
                )
            };
            let start = index * 2 * B as usize;
            for plane in output.chunks_exact_mut(stride) {
                let a = vmovn_u32(step::<B, SIGN>(&mut a));
                let b = vmovn_u32(step::<B, SIGN>(&mut b));
                let c = vmovn_u32(step::<B, SIGN>(&mut c));
                let d = vmovn_u32(step::<B, SIGN>(&mut d));
                let raw = vcombine_u8(vmovn_u16(vcombine_u16(a, b)), vmovn_u16(vcombine_u16(c, d)));
                let digits = vreinterpretq_s8_u8(vsubq_u8(veorq_u8(raw, half), half));
                min = vminq_s8(min, digits);
                max = vmaxq_s8(max, digits);
                if B == 3 {
                    let pairs = vreinterpretq_u64_u8(raw);
                    let pairs = vandq_u64(
                        vorrq_u64(pairs, vshrq_n_u64::<5>(pairs)),
                        vdupq_n_u64(0x003f003f003f003f),
                    );
                    let quads = vandq_u64(
                        vorrq_u64(pairs, vshrq_n_u64::<10>(pairs)),
                        vdupq_n_u64(0x00000fff00000fff),
                    );
                    let octets = vreinterpretq_u8_u64(vorrq_u64(quads, vshrq_n_u64::<20>(quads)));
                    let packed = vqtbl1q_u8(octets, gather);
                    let word = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(packed));
                    plane[start..start + 6].copy_from_slice(&word.to_le_bytes()[..6]);
                } else {
                    let lo = vget_low_u8(vuzp1q_u8(raw, raw));
                    let hi = vget_low_u8(vuzp2q_u8(raw, raw));
                    let word =
                        vget_lane_u64::<0>(vreinterpret_u64_u8(vorr_u8(lo, vshl_n_u8::<4>(hi))));
                    plane[start..start + 8].copy_from_slice(&word.to_le_bytes());
                }
            }
        }
    }

    SignedDigitBounds {
        negative_abs_max: vminvq_s8(min).unsigned_abs(),
        positive_max: vmaxvq_s8(max) as u8,
    }
}
