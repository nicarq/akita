use super::*;

#[test]
fn signed_limb_boundaries_match_integer_oracle() {
    const Q: u64 = u64::MAX - 58;
    const CENTER: u64 = 0x7fff_7fff_7fff_7fff;
    let boundaries = [
        0,
        1,
        CENTER - 1,
        CENTER,
        CENTER + 1,
        1 << 63,
        Q - 32768,
        Q - 1,
    ];
    for basis in [4, 8, 16, 32] {
        let half = basis as i8 / 2;
        for width in [64, 65, 72] {
            let digits = (0..513 * width)
                .map(|index| [-half, half - 1, 0, -1][(index % width + index / width) % 4])
                .collect::<Vec<_>>();
            for count in [15, 16, 255, 256, 257, 513] {
                let factors: Vec<_> = (0..count)
                    .map(|row| {
                        let coordinates = if row & 1 == 0 {
                            [CENTER + 1, Q - 1]
                        } else {
                            [boundaries[row % 8], boundaries[(row + 3) % 8]]
                        };
                        (coordinates, (512 - row) as u32)
                    })
                    .collect();
                let rows: Vec<_> = factors
                    .iter()
                    .map(|&([a, b], row)| (Weight::new(Base::from_u64(a), Base::from_u64(b)), row))
                    .collect();
                let expected: Vec<_> = (0..width)
                    .map(|column| {
                        let integer: [i128; 2] = std::array::from_fn(|coordinate| {
                            factors.iter().fold(0, |sum, &(values, row)| {
                                let digit = digits[row as usize * width + column];
                                sum + i128::from(values[coordinate]) * i128::from(digit)
                            })
                        });
                        let [a, b] =
                            integer.map(|x| Base::from_u64(x.rem_euclid(i128::from(Q)) as u64));
                        Weight::new(a, b)
                    })
                    .collect();
                for use_neon in [false, std::arch::is_aarch64_feature_detected!("neon")] {
                    let mut actual = vec![Weight::zero(); width];
                    add_rows(&mut actual, half as usize, &rows, &digits, use_neon);
                    assert_eq!(
                        actual, expected,
                        "basis={basis}, width={width}, rows={count}, neon={use_neon}"
                    );
                }
            }
        }
    }
}
