use akita_algebra::{
    cyclic_ntt_with_i16_tail_to_ring, ntt_with_i16_tail_to_ring, CanonicalEncoding,
    CyclotomicCrtNtt, CyclotomicRing, Field, I16TailParams, MontCoeff,
};
use akita_algebra::{
    ntt::tables::{I16_TAIL_PRIME, Q64_MODULUS, Q64_PRIMES},
    CrtNttParamSet,
};
use std::ffi::c_char;
const SOURCE: &str = concat!(include_str!("metal_quotient.metal"), "\0");

#[repr(C)]
struct Shape {
    width: u32,
    chunk: u32,
    primes: [[i32; 4]; 4],
}

#[derive(Clone)]
#[repr(C)]
struct Chunk<const D: usize> {
    base: [CyclotomicCrtNtt<i32, 3, D>; 2],
    tail: [CyclotomicCrtNtt<i16, 1, D>; 2],
}

unsafe extern "C" {
    fn akita_quotient_products(
        source: *const c_char,
        shape: *const Shape,
        matrix: *const u64,
        z: *const i32,
        tables: *const i32,
        out: *mut i32,
        len: usize,
    ) -> i32;
}

pub(super) fn quotient<F: Field + CanonicalEncoding, const D: usize>(
    matrix: &[F],
    rhs: &[[i32; D]],
    bound: u64,
) -> Option<CyclotomicRing<F, D>> {
    if F::CHALLENGE_SET_LIMBS != [Q64_MODULUS, 0, 0, 0]
        || !super::requirements::streamable_quotient_shape(D, 1, rhs.len())
    {
        return None;
    }
    let matrix = F::canonical_u64_slice(matrix.get(..rhs.len() * D)?)?;
    let bp = CrtNttParamSet::<i32, 3, D>::new(Q64_PRIMES);
    let tp = CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]);
    if bound >= bp.primes.iter().map(|q| q.p as u64).min()? {
        return None;
    }
    let chunk = bp
        .crt_capacity()
        .with_prime_modulus(tp.primes[0].p as u128)
        .max_safe_width::<F, D>(bound)?
        .min(rhs.len());
    let chunks = rhs.len().checked_div(chunk)? + usize::from(!rhs.len().is_multiple_of(chunk));
    if chunks > 128 {
        return None;
    }
    let mut shape = Shape {
        width: rhs.len() as u32,
        chunk: chunk as u32,
        primes: [[0; 4]; 4],
    };
    let mut tables = Vec::with_capacity(4 * 2 * D);
    macro_rules! append {
        ($p:expr,$i:expr,$k:expr) => {{
            let q = $p.primes[$i];
            shape.primes[$k] = [q.p as i32, q.pinv as i32, q.mont as i32, q.montsq as i32];
            let [fwd, _, psi, _] = $p.twiddles[$i].negacyclic_tables();
            tables.extend(fwd.iter().map(|x| x.raw() as i32));
            tables.extend(
                psi.iter()
                    .map(|&x| q.mul(x, MontCoeff::from_raw(q.montsq)).raw() as i32),
            );
        }};
    }
    for j in 0..3 {
        append!(bp, j, j);
    }
    append!(tp, 0, 3);
    let mut out = vec![
        Chunk {
            base: std::array::from_fn(|_| CyclotomicCrtNtt::zero()),
            tail: std::array::from_fn(|_| CyclotomicCrtNtt::zero())
        };
        chunks.next_multiple_of(2)
    ];
    // SAFETY: Chunk has C layout over contiguous base and tail spectra. All slices cover
    // the checked dispatch shape. Keys remain registered until the command completes.
    if unsafe {
        akita_quotient_products(
            SOURCE.as_ptr().cast(),
            &shape,
            matrix.as_ptr(),
            rhs.as_ptr().cast(),
            tables.as_ptr(),
            out.as_mut_ptr().cast(),
            size_of_val(out.as_slice()),
        )
    } == 0
    {
        return None;
    }
    let params = I16TailParams::new(bp, tp);
    let reconstruct = |r: &Chunk<D>| {
        let neg = ntt_with_i16_tail_to_ring::<F, 3, D>(&r.base[0], &r.tail[0], &params);
        let mut cyc = cyclic_ntt_with_i16_tail_to_ring::<F, 3, D>(&r.base[1], &r.tail[1], &params);
        for (c, n) in cyc.coefficients_mut().iter_mut().zip(neg.coefficients()) {
            *c = (*c - *n).half();
        }
        cyc
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        Some(
            out[..chunks]
                .par_iter()
                .map(reconstruct)
                .reduce(CyclotomicRing::zero, |mut a, b| {
                    a += b;
                    a
                }),
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        Some(
            out[..chunks]
                .iter()
                .map(reconstruct)
                .fold(CyclotomicRing::zero(), |mut a, b| {
                    a += b;
                    a
                }),
        )
    }
}
