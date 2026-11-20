use super::super::super::packed_quad_rounds::{inputs, witness_classes, E};
use super::MetalLookup;
use jolt_field::Zero;

unsafe extern "C" {
    fn akita_quad_lookup_buffers_fit(
        limit: usize,
        body: usize,
        tail: usize,
        largest_copy: usize,
    ) -> i32;
}

#[test]
fn oversized_metal_buffers_take_the_cpu_fallback() {
    let page = 16_384;
    let limit = 2 * page;
    let fits = |bytes: usize, copied| {
        // SAFETY: The native size check takes scalar byte counts and accesses no storage.
        unsafe {
            akita_quad_lookup_buffers_fit(limit, bytes / page * page, bytes % page, copied) != 0
        }
    };
    assert!(fits(limit, limit));
    assert!(fits(limit + page - 1, limit));
    assert!(!fits(limit + page, limit));
    assert!(!fits(limit, limit + 1));
    assert!(!fits(usize::MAX, 0));
}

fn check<const G: usize, const SKIP: bool>(handle: &MetalLookup, classes: &[u16]) {
    let pairs = classes.len() / (2 * G);
    let (table, low, high) = inputs::<G>(pairs);
    let mut expected = vec![E::zero(); 2 * pairs];
    let sums = super::super::super::lookup_pairs::<E, G, SKIP>(
        &table,
        4096,
        classes,
        &low,
        &high,
        0..pairs,
        Some(&mut expected),
    );
    let mut allocation = vec![E::zero(); 2 * pairs + 65_536 / size_of::<E>()];
    let begin = allocation.as_ptr().align_offset(65_536);
    let out = &mut allocation[begin..begin + 2 * pairs];
    // SAFETY: The field is the P64 quadratic extension. The packed input has 2*G quads per pair.
    let got = unsafe { handle.evaluate::<E, G, SKIP>(pairs, &table, &low, &high, Some(out)) };
    assert_eq!(got, Some(sums));
    assert_eq!(out, expected);
}

#[test]
#[ignore = "requires a Metal device with 32-wide SIMD groups"]
fn metal_packed_quad_norms_and_materialization_match_the_cpu() {
    let (packed, classes) = witness_classes();
    let bytes = packed.encoded_bytes();
    let mut allocation = vec![0; bytes.len() + 65_536];
    let begin = allocation.as_ptr().align_offset(65_536);
    let raw = &mut allocation[begin..begin + bytes.len()];
    raw.copy_from_slice(bytes);
    // SAFETY: The source allocation outlives the handle and stays immutable after registration.
    let handle = unsafe { MetalLookup::new(raw) }.expect("Metal lookup unavailable");
    check::<2, false>(&handle, &classes);
    check::<2, true>(&handle, &classes);
    check::<4, false>(&handle, &classes);
    check::<4, true>(&handle, &classes);
    check::<8, false>(&handle, &classes);
    check::<8, true>(&handle, &classes);
}
