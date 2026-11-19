use std::sync::Arc;

use jolt_field::{Prime64Offset59, Ring};

use super::{DensePoly, MIN_FLAT_COEFF_LEN};

#[test]
fn shared_key_words_preserve_ring_views_after_owner_drop() {
    type F = Prime64Offset59;
    let mut words = vec![0; MIN_FLAT_COEFF_LEN];
    words[..129].fill(17);
    let words = Arc::new(words);
    let source = DensePoly::<F>::from_shared_canonical_u64_prefix(14, words.clone()).unwrap();
    let clone = source.clone();
    let dense = DensePoly::from_field_prefix(14, vec![F::from_u64(17); 129]).unwrap();
    assert_eq!(source, dense);
    assert_eq!(source.field_coeffs().as_ptr().cast::<u64>(), words.as_ptr());
    assert_eq!(clone.field_coeffs().as_ptr().cast::<u64>(), words.as_ptr());
    assert_eq!(
        source.nonzero_ring_coeffs::<64>().unwrap(),
        dense.nonzero_ring_coeffs::<64>().unwrap()
    );
    assert_eq!(
        source.nonzero_ring_coeffs::<2048>().unwrap(),
        dense.nonzero_ring_coeffs::<2048>().unwrap()
    );
    drop(words);
    drop(source);
    assert_eq!(clone, dense);
}

#[test]
fn shared_key_words_reject_noncanonical_or_unaligned_input() {
    type F = Prime64Offset59;
    let mut words = vec![0; MIN_FLAT_COEFF_LEN];
    words[0] = u64::MAX;
    assert!(DensePoly::<F>::from_shared_canonical_u64_prefix(14, Arc::new(words)).is_err());
    assert!(DensePoly::<F>::from_shared_canonical_u64_prefix(14, Arc::new(vec![0; 3])).is_err());
}
