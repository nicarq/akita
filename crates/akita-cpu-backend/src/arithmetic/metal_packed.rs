use crate::sources::packed_digits::PackedSignedDigitView;
use akita_algebra::{CanonicalEncoding, CyclotomicCrtNtt, CyclotomicRing, Field, MontCoeff};
use akita_types::ntt_cache::PreparedNttCache;
use std::ffi::c_char;

#[repr(C)]
struct Shape {
    blocks: u32,
    cols: u32,
    digits: u32,
    prefix_bytes: u32,
    primes: [[i32; 4]; 3],
}

unsafe extern "C" {
    fn akita_metal_packed_commit(
        source: *const c_char,
        shape: *const Shape,
        digits: *const u8,
        matrix: *const i32,
        tables: *const i32,
        out: *mut i32,
    ) -> i32;
}

pub(crate) fn commit<F: Field + CanonicalEncoding, const D: usize>(
    slot: &PreparedNttCache<D>,
    digits: PackedSignedDigitView<'_>,
    cols: usize,
    blocks: usize,
) -> Option<Vec<Vec<CyclotomicRing<F, D>>>> {
    if D != 512 || cols != 2048 || digits.len() < 16 * cols * D || !digits.len().is_multiple_of(D) {
        return None;
    }
    let (stored, bytes) = digits.three_bit_payload()?;
    if stored > blocks.checked_mul(cols)?.checked_mul(D)? {
        return None;
    }
    let view = slot.q64_base()?;
    let params = view.params();
    if !params.crt_capacity().supports::<F, D>(cols, 4) {
        return None;
    }
    let matrix = view.negacyclic()?.get(..2 * cols)?;
    let logical = blocks
        .next_multiple_of(4)
        .checked_mul(cols)?
        .checked_mul(D)?;
    if logical > u32::MAX as usize / 3 {
        return None;
    }
    let shape = Shape {
        blocks: blocks as u32,
        cols: cols as u32,
        digits: stored as u32,
        prefix_bytes: 0,
        primes: params.primes.map(|q| [q.p, q.pinv, q.mont, q.montsq]),
    };
    let mut tables = Vec::with_capacity(3 * 2 * D);
    for (q, tw) in params.primes.iter().zip(&params.twiddles) {
        let [fwd, _, psi, _] = tw.negacyclic_tables();
        tables.extend(fwd.iter().map(|x| x.raw()));
        tables.extend(
            psi.iter()
                .map(|&x| q.mul(x, MontCoeff::from_raw(q.montsq)).raw()),
        );
    }
    const SOURCE: &str = concat!(include_str!("../sources/dense/fixed_fold.metal"), "\0");
    let mut out = vec![CyclotomicCrtNtt::<i32, 3, D>::zero(); blocks * 2];
    // SAFETY: Each CRT ring is transparent over 3×D contiguous i32 words. The blocking
    // command borrows the slices above and writes only the initialized output slice.
    let ok = unsafe {
        akita_metal_packed_commit(
            SOURCE.as_ptr().cast(),
            &shape,
            bytes.as_ptr(),
            matrix.as_ptr().cast(),
            tables.as_ptr(),
            out.as_mut_ptr().cast(),
        )
    };
    if ok == 0 {
        return None;
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        Some(
            out.par_chunks_exact(2)
                .map(|rows| rows.iter().map(|r| r.to_ring::<F>(params)).collect())
                .collect(),
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        Some(
            out.chunks_exact(2)
                .map(|rows| rows.iter().map(|r| r.to_ring::<F>(params)).collect())
                .collect(),
        )
    }
}
