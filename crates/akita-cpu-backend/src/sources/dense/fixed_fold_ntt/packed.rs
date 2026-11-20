use super::{PRIME, STRIDE};

pub(super) type PackedSpectrum = [u32; 15];
const _: () = assert!(PRIME.p > 0 && PRIME.p < 1 << 30);

pub(super) fn storage_words(rows: usize, page_words: usize) -> Option<usize> {
    let packed = rows.checked_mul(15)?;
    if page_words != 0 && rows.checked_mul(STRIDE)?.is_multiple_of(page_words) {
        packed.div_ceil(page_words).checked_mul(page_words)
    } else {
        Some(packed)
    }
}

pub(super) fn page_words() -> usize {
    #[cfg(akita_fold_metal)]
    {
        unsafe extern "C" {
            fn getpagesize() -> std::ffi::c_int;
        }
        // SAFETY: Darwin's getpagesize takes no arguments and returns the VM page size in bytes.
        unsafe { getpagesize() as usize / size_of::<u32>() }
    }
    #[cfg(not(akita_fold_metal))]
    {
        0
    }
}

pub(super) fn write(row: &mut PackedSpectrum, lane: usize, centered: i32) {
    let bits = centered as u32 & 0x3fff_ffff;
    let i = lane % 4;
    match lane / 4 {
        0 => row[i] |= bits,
        1 => {
            row[i] |= bits << 30;
            row[4 + i] |= bits >> 2;
        }
        2 => {
            row[4 + i] |= bits << 28;
            row[8 + i] |= bits >> 4;
        }
        3 => {
            row[8 + i] |= bits << 26;
            let tail = bits >> 6;
            let word = 12 + 3 * i / 4;
            let shift = 24 * i % 32;
            row[word] |= tail << shift;
            if shift > 8 {
                row[word + 1] |= tail >> (32 - shift);
            }
        }
        _ => unreachable!(),
    }
}

pub(super) fn unpack(row: &PackedSpectrum) -> [i32; STRIDE] {
    std::array::from_fn(|lane| {
        let i = lane % 4;
        let bits = match lane / 4 {
            0 => row[i],
            1 => row[i] >> 30 | row[4 + i] << 2,
            2 => row[4 + i] >> 28 | row[8 + i] << 4,
            _ => {
                let word = 12 + 3 * i / 4;
                let shift = 24 * i % 32;
                let mut tail = row[word] >> shift;
                if shift > 8 {
                    tail |= row[word + 1] << (32 - shift);
                }
                row[8 + i] >> 26 | tail << 6
            }
        };
        (bits << 2) as i32 >> 2
    })
}
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn unpack_neon(row: &PackedSpectrum) -> [std::arch::aarch64::int32x4_t; 4] {
    use std::arch::aarch64::*;
    // SAFETY: each group contains four words. The final load covers bytes 44 through 59.
    unsafe {
        let a = vld1q_u32(row.as_ptr());
        let b = vld1q_u32(row.as_ptr().add(4));
        let c = vld1q_u32(row.as_ptr().add(8));
        let last = vld1q_u8(row.as_ptr().add(11).cast::<u8>());
        let lanes = [4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255, 13, 14, 15, 255];
        let d = vreinterpretq_u32_u8(vqtbl1q_u8(last, vld1q_u8(lanes.as_ptr())));
        [
            a,
            vorrq_u32(vshrq_n_u32::<30>(a), vshlq_n_u32::<2>(b)),
            vorrq_u32(vshrq_n_u32::<28>(b), vshlq_n_u32::<4>(c)),
            vorrq_u32(vshrq_n_u32::<26>(c), vshlq_n_u32::<6>(d)),
        ]
        .map(|v| vshrq_n_s32::<2>(vreinterpretq_s32_u32(vshlq_n_u32::<2>(v))))
    }
}
