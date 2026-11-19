#[test]
fn p64_direct_mass_matches_canonical_sum_over_repeated_row_batches() {
    use jolt_field::Ring;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    type E = Ext2<Prime64Offset59>;
    let mut rng = StdRng::seed_from_u64(0x6a14_745f_a87a_c521);
    const WIDTH: usize = 64;
    for basis in [8, 16, 32] {
        let half = basis as i8 / 2;
        let values = (0..311 * WIDTH)
            .map(|_| rng.gen_range(-half..half))
            .collect::<Vec<_>>();
        let rows = (0..513)
            .map(|_| (E::random(&mut rng), rng.gen_range(0..311u32)))
            .collect::<Vec<_>>();
        let mut expected = [E::zero(); WIDTH];
        for (factor, row) in &rows {
            let first = *row as usize * WIDTH;
            for (sum, &digit) in expected.iter_mut().zip(&values[first..first + WIDTH]) {
                *sum += *factor * E::from_i64(i64::from(digit));
            }
        }
        for portable in [false, true] {
            let mut mass = WideMass::<E, 32>::new(2, WIDTH, basis);
            mass.force_portable = portable;
            mass.add_rows(1, &rows[..509], &values);
            mass.add_rows(1, &rows[509..], &values);
            let got = mass.finish();
            assert_eq!(&got[..WIDTH], &[E::zero(); WIDTH]);
            assert_eq!(&got[WIDTH..], &expected);
        }
    }
}
