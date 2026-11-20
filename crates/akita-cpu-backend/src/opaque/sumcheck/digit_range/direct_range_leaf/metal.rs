use super::*;
use std::ffi::{c_char, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn akita_quartic_open(
        source: *const c_char,
        digits: *const u8,
        count: u32,
        values: *const c_void,
    ) -> *mut c_void;
    fn akita_quartic_round(
        handle: *mut c_void,
        round: u32,
        low: *const c_void,
        low_len: u32,
        high: *const c_void,
        high_len: u32,
        challenge: *const c_void,
        sums: *mut c_void,
    ) -> i32;
    fn akita_quartic_export(handle: *mut c_void, round: u32, len: usize, rows: *mut c_void);
    fn akita_quartic_release(handle: *mut c_void);
}

struct Handle(NonNull<c_void>);

// SAFETY: Metal owns thread-safe buffers. Rust gives each prover exclusive access to its commands.
unsafe impl Send for Handle {}
// SAFETY: Shared access exposes only the last completed round polynomial.
unsafe impl Sync for Handle {}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: The native open function returned one retained reference.
        unsafe { akita_quartic_release(self.0.as_ptr()) };
    }
}

pub(super) struct MetalQuartic<E: Field> {
    handle: Handle,
    digits: PackedSignedDigits,
    values: Vec<E>,
    challenges: Vec<E>,
    pub(super) polynomial: OmittedConstantPoly<E>,
}

impl<E: Field + Ring + Unreduced> MetalQuartic<E> {
    pub(super) fn start(
        digits: &PackedSignedDigits,
        quad_values: &[E],
        challenge: E,
        eq: &GruenSplitEq<E>,
    ) -> Option<Self> {
        if digits.bit_width() != 3
            || !(1 << 26..=1 << 29).contains(&digits.len())
            || quad_values.len() != 256
        {
            return None;
        }
        let low: Vec<E> = quad_values
            .iter()
            .map(|&v| v * (E::one() - challenge))
            .collect();
        let high: Vec<E> = quad_values.iter().map(|&v| v * challenge).collect();
        let values: Vec<E> = (0..65536).map(|i| low[i & 255] + high[i >> 8]).collect();
        let source = concat!(
            include_str!("../../p64_ext2.metal"),
            include_str!("metal.metal"),
            "\0"
        );
        // SAFETY: DigitRangeSession selects only Ext2<Prime64Offset59>. The handle borrows digits,
        // which stays owned until after the handle drops, and copies the 65,536 field elements.
        let handle = Handle(NonNull::new(unsafe {
            akita_quartic_open(
                source.as_ptr().cast(),
                digits.encoded_bytes().as_ptr(),
                digits.len() as u32,
                values.as_ptr().cast(),
            )
        })?);
        let mut prover = Self {
            handle,
            digits: digits.clone(),
            values,
            challenges: Vec::new(),
            polynomial: OmittedConstantPoly::new(Vec::new()),
        };
        prover.evaluate(3, challenge, eq).then_some(prover)
    }

    fn evaluate(&mut self, round: usize, challenge: E, eq: &GruenSplitEq<E>) -> bool {
        let (low, high) = eq.remaining_eq_tables();
        let mut sums = [E::zero(); 4];
        // SAFETY: Both slices describe the equality table at this round. The native call completes
        // before returning and writes exactly four canonical quadratic-extension elements.
        let ok = unsafe {
            akita_quartic_round(
                self.handle.0.as_ptr(),
                round as u32,
                low.as_ptr().cast(),
                low.len() as u32,
                high.as_ptr().cast(),
                high.len() as u32,
                (&challenge as *const E).cast(),
                sums.as_mut_ptr().cast(),
            )
        };
        if ok == 0 {
            return false;
        }
        for (sum, &scale) in sums.iter_mut().zip(RangePoly::Quartic.taylor_factors()) {
            *sum *= E::from_u64(scale);
        }
        self.polynomial = OmittedConstantPoly::new(sums.to_vec());
        true
    }

    pub(super) fn bind(
        mut self,
        round: usize,
        r: E,
        eq: &GruenSplitEq<E>,
        cached: &mut Option<OmittedConstantPoly<E>>,
    ) -> LowBasisRangeImageStorage<E>
    where
        E: Fold,
    {
        self.challenges.push(r);
        if !self.evaluate(round, r, eq) {
            return LowBasisRangeImageStorage::Materialized(self.recover());
        }
        let count = self.digits.len().div_ceil(1 << round);
        if count > 16384 {
            return LowBasisRangeImageStorage::Metal(self);
        }
        let mut rows = vec![E::zero(); count];
        // SAFETY: The completed round has exactly count live elements, and owns the source buffer.
        unsafe {
            akita_quartic_export(
                self.handle.0.as_ptr(),
                round as u32,
                count,
                rows.as_mut_ptr().cast(),
            )
        };
        *cached = Some(self.polynomial);
        LowBasisRangeImageStorage::Materialized(rows)
    }

    pub(super) fn recover(self) -> Vec<E>
    where
        E: Fold,
    {
        let Self {
            handle,
            digits,
            values,
            challenges,
            ..
        } = self;
        drop(handle);
        let reader = octet_prefix::OctetClassReader::new(&digits, 2, octet_prefix::DigitWidth::W3);
        let pairs = digits.len().div_ceil(16);
        let first = E::precompute(challenges[0]);
        let tile_size = 4096;
        let mut rows = vec![E::zero(); pairs];
        cfg_chunks_mut!(rows, tile_size)
            .enumerate()
            .for_each(|(tile, output)| {
                let begin = tile * tile_size;
                let end = begin + output.len();
                let classes = reader.classes(2 * begin..2 * end);
                for (row, pair) in output.iter_mut().zip(classes.chunks_exact(2)) {
                    *row = E::fold_one(
                        &first,
                        values[usize::from(pair[0])],
                        values[usize::from(pair[1])],
                    );
                }
            });
        for &draw in &challenges[1..] {
            rows = LowBasisRangeCheckProver::fold_live_prefix(&rows, draw);
        }
        rows
    }
}
