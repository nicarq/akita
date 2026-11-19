use super::{windowed_mul_acc, I8DigitSpan, WindowedChallenge};
use akita_challenges::SparseChallenge;

fn check<const D: usize>(terms: usize) {
    let challenge = SparseChallenge {
        positions: (0..terms).map(|i| ((D - 1 + i * 7) % D) as u32).collect(),
        coeffs: (0..terms)
            .map(|i| if i % 3 == 0 { 1 } else { -1 })
            .collect(),
    };
    let plan = WindowedChallenge::new::<D>(&challenge, I8DigitSpan::Half, 63).unwrap();
    let plane: [i8; D] = std::array::from_fn(|i| if i % 5 <= 2 { 63 } else { -63 });
    let start: [i16; D] = std::array::from_fn(|i| if i % 2 == 0 { 29_000 } else { -29_000 });
    let mut actual = start;
    let mut expected = start.map(i32::from);
    for (&shift, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        for (index, &value) in plane.iter().enumerate() {
            let destination = index + shift as usize;
            let product = i32::from(coefficient) * i32::from(value);
            if destination < D {
                expected[destination] += product;
            } else {
                expected[destination - D] -= product;
            }
        }
    }
    windowed_mul_acc(&plane, &plan, &mut actual);
    assert_eq!(actual.map(i32::from), expected);
    windowed_mul_acc(&[0; D], &plan, &mut actual);
    assert_eq!(actual.map(i32::from), expected);
}

#[test]
fn shifted_windows_preserve_signed_wrap_and_odd_tails() {
    for terms in [16, 31, 32, 41] {
        check::<128>(terms);
        check::<512>(terms);
    }
}

#[test]
fn shifted_windows_keep_the_generic_path_for_other_inputs() {
    let mut challenge = SparseChallenge {
        positions: (0..31).map(|i| i * 4).collect(),
        coeffs: (0..31).map(|_| -1).collect(),
    };
    assert!(WindowedChallenge::new::<512>(&challenge, I8DigitSpan::Half, 63).is_some());
    assert!(WindowedChallenge::new::<512>(&challenge, I8DigitSpan::Half, 64).is_none());
    assert!(WindowedChallenge::new::<512>(&challenge, I8DigitSpan::Full, 128).is_none());
    challenge.coeffs[0] = 2;
    assert!(WindowedChallenge::new::<512>(&challenge, I8DigitSpan::Half, 63).is_none());
    challenge.coeffs[0] = 1;
    challenge.positions[0] = 512;
    assert!(WindowedChallenge::new::<512>(&challenge, I8DigitSpan::Half, 63).is_none());
}
