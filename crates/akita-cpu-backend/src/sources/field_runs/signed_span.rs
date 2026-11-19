use akita_error::AkitaError;
use jolt_field::Field;
use std::ops::Range;

#[derive(Debug)]
pub(super) struct SignedSpan<F> {
    start: usize,
    positive: Vec<u64>,
    negative: Vec<u64>,
    tail: (usize, F),
    live: usize,
    used: [Option<F>; 3],
}

impl<F: Field> SignedSpan<F> {
    pub(super) fn new(
        domain: usize,
        start: usize,
        positive: Vec<u64>,
        negative: Vec<u64>,
        tail: (usize, F),
    ) -> Result<Self, AkitaError> {
        let valid = positive
            .len()
            .checked_mul(64)
            .and_then(|len| start.checked_add(len))
            .is_some_and(|end| end <= tail.0 && tail.0 < domain);
        if !valid
            || positive.len() != negative.len()
            || positive.iter().zip(&negative).any(|(p, n)| p & n != 0)
        {
            return Err(AkitaError::InvalidInput(
                "invalid signed coefficient span".into(),
            ));
        }
        let live = if tail.1.is_zero() {
            positive
                .iter()
                .zip(&negative)
                .enumerate()
                .rev()
                .find_map(|(word, (&p, &n))| {
                    let present = p | n;
                    (present != 0)
                        .then(|| start + 64 * word + 64 - present.leading_zeros() as usize)
                })
                .unwrap_or(0)
        } else {
            tail.0 + 1
        };
        let used = [
            positive.iter().any(|&word| word != 0).then(F::one),
            negative.iter().any(|&word| word != 0).then(|| -F::one()),
            Some(tail.1),
        ];
        Ok(Self {
            start,
            positive,
            negative,
            tail,
            live,
            used,
        })
    }

    pub(super) fn used_values(&self) -> impl Iterator<Item = F> + '_ {
        self.used.iter().copied().flatten()
    }

    pub(super) fn live_len(&self) -> usize {
        self.live
    }

    fn words(&self, range: Range<usize>) -> impl Iterator<Item = (usize, u64, u64)> + '_ {
        let begin = range
            .start
            .saturating_sub(self.start)
            .min(self.positive.len() * 64);
        let end = range
            .end
            .saturating_sub(self.start)
            .min(self.positive.len() * 64);
        (begin / 64..end.div_ceil(64)).map(move |word| {
            let first = begin.saturating_sub(word * 64);
            let last = (end - word * 64).min(64);
            let keep = (u64::MAX << first) & (u64::MAX >> (64 - last));
            (
                self.start + word * 64,
                self.positive[word] & keep,
                self.negative[word] & keep,
            )
        })
    }

    pub(super) fn read(&self, start: usize, out: &mut [F]) {
        out.fill(F::zero());
        if out.is_empty() {
            return;
        }
        let range = start..start.saturating_add(out.len());
        for (word, positive, negative) in self.words(range.clone()) {
            let mut present = positive | negative;
            while present != 0 {
                let bit = present.trailing_zeros() as usize;
                out[(word + bit) - start] = if positive >> bit & 1 == 1 {
                    F::one()
                } else {
                    -F::one()
                };
                present &= present - 1;
            }
        }
        if range.contains(&self.tail.0) {
            out[self.tail.0 - start] = self.tail.1;
        }
    }

    pub(super) fn read_unit_digits(&self, start: usize, out: &mut [i16]) -> bool {
        let range = start..start.saturating_add(out.len());
        if range.contains(&self.tail.0) && !self.tail.1.is_zero() {
            return false;
        }
        out.fill(0);
        if out.is_empty() {
            return true;
        }
        for (word, positive, negative) in self.words(range) {
            let mut present = positive | negative;
            while present != 0 {
                let bit = present.trailing_zeros() as usize;
                out[word + bit - start] = if positive >> bit & 1 == 1 { 1 } else { -1 };
                present &= present - 1;
            }
        }
        true
    }

    pub(super) fn constant(&self, range: Range<usize>) -> Option<F> {
        if range.is_empty() || range.start >= self.live {
            return Some(F::zero());
        }
        if range.contains(&self.tail.0) && !self.tail.1.is_zero() {
            return (range.len() == 1).then_some(self.tail.1);
        }
        self.words(range).all(|(_, p, n)| p | n == 0).then(F::zero)
    }
}
