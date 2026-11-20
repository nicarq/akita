use super::CpuPreparedSetup;
use crate::{opaque::CommitInnerPlan, FieldRunPoly};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::{CyclotomicCrtNtt, CyclotomicRing, MontCoeff};
use akita_error::AkitaError;
use akita_types::{ntt_cache::PreparedI32LimbView, NttCacheKey, NttTransformDomain};
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, Field};
use std::{ffi::c_char, ops::Range, sync::Mutex};

#[repr(C)]
struct LookupShape {
    blocks: u32,
    cols: u32,
    chunk: u32,
    table_len: u32,
    column_width: u32,
    code_count: u32,
    live_cells: u32,
    primes: [[i32; 4]; 2],
}

unsafe extern "C" {
    fn akita_metal_lookup_commit(
        source: *const c_char,
        shape: *const LookupShape,
        codes: *const u16,
        lookup: *const i16,
        lookup_bytes: usize,
        spectra: *const i32,
        matrix_bytes: usize,
        twiddles: *const i32,
        ranges: *const u32,
        out: *mut i32,
    ) -> i32;
}

pub(super) fn commit<F: Field + CanonicalEncoding, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    poly: &FieldRunPoly<F>,
    ranges: &[Range<usize>],
    plan: &CommitInnerPlan,
    width: usize,
) -> Result<Option<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError> {
    if D != 512
        || plan.n_a != 3
        || plan.num_digits_inner != 5
        || plan.log_basis_inner != 13
        || width != 5120
        || !(32..=64).contains(&ranges.len())
    {
        return Ok(None);
    }
    let Some(columns) = poly.lookup_columns() else {
        return Ok(None);
    };
    let Some(cells) = columns
        .tables
        .len()
        .checked_mul(columns.tables[0].len())
        .filter(|&n| n <= 1 << 16)
    else {
        return Ok(None);
    };
    if columns.codes.is_empty()
        || columns.end > u32::MAX as usize
        || columns.width > u32::MAX as usize
    {
        return Ok(None);
    }
    #[cfg(feature = "parallel")]
    let workers = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let workers = 1usize;
    let output_bytes = ranges.len() * 9 * 2 * D * 4;
    let scratch = output_bytes * (width.div_ceil(256) + 2) + cells * 5 * 2;
    if scratch > workers.saturating_mul(width * D * 2) {
        return Ok(None);
    }
    static IN_USE: Mutex<()> = Mutex::new(());
    let Ok(_guard) = IN_USE.try_lock() else {
        return Ok(None);
    };
    let key = NttCacheKey::from_matrix_shape(
        D,
        3,
        width,
        NttTransformDomain::ExactNegacyclicI16 {
            width,
            rhs_abs_bound: 4096,
        },
    )?;
    prepared.with_shared_ntt::<D, _>(key, |cache| {
        Ok(cache
            .i32_limb_view()
            .filter(|v| v.limbs == 3 && v.width == width)
            .and_then(|view| {
                let mut input = Vec::with_capacity(ranges.len());
                for r in ranges {
                    if r.len() > width / 5 || r.end > u32::MAX as usize / D {
                        return None;
                    }
                    input.push([r.start as u32, r.len() as u32]);
                }
                let table_len = columns.tables[0].len();
                let params = BalancedDecomposePow2Params::new(5, 13);
                let mut lookup = vec![[0i16; 5]; cells];
                for (col, table) in columns.tables.iter().enumerate() {
                    for &code in columns.used {
                        let i = usize::from(code);
                        let mut planes = [[0i16; 1]; 5];
                        CyclotomicRing::from_coefficients([table[i]])
                            .balanced_decompose_pow2_i16_into(&mut planes, &params);
                        lookup[col * table_len + i] = planes.map(|[d]| d);
                    }
                }
                let shape = LookupShape {
                    blocks: input.len() as u32,
                    cols: width as u32,
                    chunk: 256,
                    table_len: table_len as u32,
                    column_width: columns.width as u32,
                    code_count: columns.codes.len() as u32,
                    live_cells: columns.end as u32,
                    primes: view.params.primes.map(|q| [q.p, q.pinv, q.mont, q.montsq]),
                };
                run(view, &shape, columns.codes, &lookup, &input)
            }))
    })
}

fn run<F: Field + CanonicalEncoding, const D: usize>(
    view: PreparedI32LimbView<'_, D>,
    shape: &LookupShape,
    codes: &[u16],
    lookup: &[[i16; 5]],
    ranges: &[[u32; 2]],
) -> Option<Vec<Vec<CyclotomicRing<F, D>>>> {
    let mut twiddles = Vec::with_capacity(4 * D);
    for (q, tw) in view.params.primes.iter().zip(&view.params.twiddles) {
        let [fwd, _, psi, _] = tw.negacyclic_tables();
        twiddles.extend(fwd.iter().map(|v| v.raw()));
        twiddles.extend(
            psi.iter()
                .map(|&v| q.mul(v, MontCoeff::from_raw(q.montsq)).raw()),
        );
    }
    let matrix = view.spectra.get(..9 * view.width)?;
    let mut output = vec![CyclotomicCrtNtt::<i32, 2, D>::zero(); ranges.len() * 9];
    const SOURCE: &str = concat!(include_str!("../sources/dense/fixed_fold.metal"), "\0");
    // SAFETY: The synchronous command borrows the two-prime matrix and the checked lookup slices until completion.
    let ok = unsafe {
        akita_metal_lookup_commit(
            SOURCE.as_ptr().cast(),
            shape,
            codes.as_ptr(),
            lookup.as_ptr().cast(),
            std::mem::size_of_val(lookup),
            matrix.as_ptr().cast(),
            std::mem::size_of_val(matrix),
            twiddles.as_ptr(),
            ranges.as_ptr().cast(),
            output.as_mut_ptr().cast(),
        )
    };
    if ok == 0 {
        return None;
    }
    let weights = [
        F::pow2(view.limb_bits as usize),
        F::pow2(2 * view.limb_bits as usize),
    ];
    Some(
        cfg_chunks!(output, 9)
            .map(|rows| {
                rows.chunks_exact(3)
                    .map(|limbs| {
                        let mut ring = limbs[0].to_ring::<F>(view.params);
                        for (limb, weight) in limbs[1..].iter().zip(weights) {
                            limb.to_ring::<F>(view.params)
                                .scale_accumulate_into(&mut ring, weight);
                        }
                        ring
                    })
                    .collect()
            })
            .collect(),
    )
}

#[cfg(test)]
#[path = "../../tests/unit/metal_lookup.rs"]
mod tests;
