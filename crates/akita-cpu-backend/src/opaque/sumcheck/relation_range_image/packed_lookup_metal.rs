use jolt_field::Field;
use std::ffi::{c_char, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn pgl_akita_quad_lookup_open(raw: *const u8, len: usize, shader: *const c_char) -> *mut c_void;
    fn pgl_akita_quad_lookup_close(handle: *mut c_void);
    fn pgl_akita_quad_lookup_round(
        handle: *mut c_void,
        g: u32,
        pairs: u32,
        class_count: u32,
        lut: *const u8,
        low: *const u8,
        low_len: u32,
        high: *const u8,
        high_len: u32,
        skip: i32,
        out: *mut u8,
        totals: *mut u8,
    ) -> i32;
}

pub(in super::super) struct MetalLookup(NonNull<c_void>);
// SAFETY: The Metal buffer holds read-only pages of the witness owned alongside this handle.
unsafe impl Send for MetalLookup {}
// SAFETY: Every command allocates its own output buffers. The Metal command queue is thread safe.
unsafe impl Sync for MetalLookup {}

impl MetalLookup {
    /// # Safety
    /// The storage backing `raw` must remain allocated and immutable until the handle drops.
    pub(super) unsafe fn new(raw: &[u8]) -> Option<Self> {
        let _ = u32::try_from(raw.len()).ok()?;
        let shader = concat!(
            include_str!("../p64_ext2.metal"),
            include_str!("packed_lookup.metal"),
            "\0"
        );
        // SAFETY: The synchronous constructor registers only complete pages of raw. Its owner outlives this handle.
        NonNull::new(unsafe {
            pgl_akita_quad_lookup_open(raw.as_ptr(), raw.len(), shader.as_ptr().cast())
        })
        .map(Self)
    }
    /// # Safety
    /// `E` must be the canonical P64 quadratic extension. The witness must contain
    /// two groups of `G` quads for each pair.
    pub(in super::super) unsafe fn evaluate<E: Field, const G: usize, const SKIP: bool>(
        &self,
        pairs: usize,
        lookups: &[E],
        low: &[E],
        high: &[E],
        out: Option<&mut [E]>,
    ) -> Option<[E; 3]> {
        let class_count = 4096;
        if lookups.len() != G * class_count
            || low.is_empty()
            || high.len() < pairs.div_ceil(low.len())
        {
            return None;
        }
        u32::try_from(pairs.checked_add(low.len() - 1)?).ok()?;
        let pair_count = u32::try_from(pairs).ok()?;
        let low_len = u32::try_from(low.len()).ok()?;
        let high_len = u32::try_from(high.len()).ok()?;
        let dest = match out {
            Some(v) if v.len() == 2 * pairs => v.as_mut_ptr().cast(),
            Some(_) => return None,
            None => std::ptr::null_mut(),
        };
        let mut sums = [E::zero(); 3];
        // SAFETY: Construction selects the canonical P64 quadratic extension. The checked slices remain live and the command completes before return.
        let valid = unsafe {
            pgl_akita_quad_lookup_round(
                self.0.as_ptr(),
                G as u32,
                pair_count,
                class_count as u32,
                lookups.as_ptr().cast(),
                low.as_ptr().cast(),
                low_len,
                high.as_ptr().cast(),
                high_len,
                i32::from(SKIP),
                dest,
                sums.as_mut_ptr().cast(),
            )
        };
        (valid != 0).then_some(sums)
    }
}

impl Drop for MetalLookup {
    fn drop(&mut self) {
        // SAFETY: This handle owns one retained native reference.
        unsafe {
            pgl_akita_quad_lookup_close(self.0.as_ptr());
        }
    }
}

#[cfg(test)]
#[path = "tests/packed_quad_metal.rs"]
mod tests;
