use super::PackedSignedDigitView;
#[derive(Clone, Copy)]
pub(crate) struct PackedQuadPairs<'a>(&'a [[u8; 3]]);

impl<'a> PackedQuadPairs<'a> {
    pub(crate) fn new(view: PackedSignedDigitView<'a>) -> Option<Self> {
        if view.bit_width != 3 {
            return None;
        }
        let (pairs, tail) = view.aligned_bytes(0..view.len)?.as_chunks::<3>();
        tail.is_empty().then_some(Self(pairs))
    }
    pub(crate) fn at(self, i: usize) -> u16 {
        let pair = &self.0[i / 2];
        let offset = i & 1;
        ((u16::from_le_bytes([pair[offset], pair[offset + 1]]) >> (offset * 4)) & 0xfff) ^ 0x924
    }
    pub(crate) fn uniform<const G: usize>(self, start: usize, end: usize) -> bool {
        let tile = &self.0[G * start..G * end];
        let first = &tile[..G];
        first == &tile[tile.len() - G..] && tile.chunks_exact(G).all(|p| p == first)
    }
}

#[cfg(test)]
#[path = "tests/quad_pairs.rs"]
mod tests;
