use super::super::PackedSignedDigits;
use super::PackedQuadPairs;

#[test]
fn signed_classes_match_the_decoded_view_and_reject_partial_groups() {
    let digits: Vec<_> = (-4..4).cycle().take(128).collect();
    let packed = PackedSignedDigits::from_i8_digits_auto(digits);
    for start in 0..16 {
        for len in [8, 16, 24, 32, 56] {
            let view = packed.view().slice(start..start + len).unwrap();
            let Some(quads) = PackedQuadPairs::new(view) else {
                assert_ne!(start % 8, 0);
                continue;
            };
            assert_eq!(start % 8, 0);
            for i in 0..len / 4 {
                let expected = (0..4).fold(0u16, |x, j| {
                    x | (((i16::from(view.get(4 * i + j).unwrap()) + 4) as u16) << (3 * j))
                });
                assert_eq!(quads.at(i), expected);
            }
        }
    }
    assert!(PackedQuadPairs::new(packed.zero_padded(144).unwrap()).is_none());
    assert!(PackedQuadPairs::new(packed.view().slice(0..9).unwrap()).is_none());
    assert!(
        PackedQuadPairs::new(PackedSignedDigits::from_i8_digits_auto(vec![0; 128]).view())
            .is_none()
    );
}
