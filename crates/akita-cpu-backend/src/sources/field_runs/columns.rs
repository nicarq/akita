use akita_error::AkitaError;
use jolt_field::Field;
use std::ops::Range;
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct CodeColumns<F> {
    codes: Arc<Vec<u16>>,
    tables: Vec<Vec<F>>,
    used: Vec<u16>,
    width: usize,
    end: usize,
}

#[cfg(akita_fold_metal)]
pub(crate) struct LookupColumns<'a, F> {
    pub codes: &'a [u16],
    pub used: &'a [u16],
    pub tables: &'a [Vec<F>],
    pub width: usize,
    pub end: usize,
}

impl<F: Field> CodeColumns<F> {
    #[cfg(akita_fold_metal)]
    pub(super) fn view(&self) -> LookupColumns<'_, F> {
        LookupColumns {
            codes: &self.codes,
            used: &self.used,
            tables: &self.tables,
            width: self.width,
            end: self.end,
        }
    }

    pub(super) fn new(
        domain: usize,
        width: usize,
        codes: Arc<Vec<u16>>,
        tables: Vec<Vec<F>>,
    ) -> Result<Self, AkitaError> {
        let Some(table_len) = tables.first().map(Vec::len) else {
            return Err(AkitaError::InvalidInput("empty field lookup".into()));
        };
        if !width.is_power_of_two()
            || width > domain
            || codes.len() > width
            || tables.len() > domain / width
            || table_len > usize::from(u16::MAX)
            || tables.iter().any(|table| table.len() != table_len)
        {
            return Err(AkitaError::InvalidInput(
                "invalid field lookup shape".into(),
            ));
        }
        let mut seen = vec![false; table_len];
        for &code in codes.iter().filter(|&&code| code != u16::MAX) {
            let flag = seen.get_mut(usize::from(code)).ok_or_else(|| {
                AkitaError::InvalidInput("field lookup index out of range".into())
            })?;
            *flag = true;
        }
        let used = seen
            .into_iter()
            .enumerate()
            .filter_map(|(i, set)| set.then_some(i as u16))
            .collect();
        let end = tables
            .iter()
            .enumerate()
            .rev()
            .find_map(|(col, table)| {
                codes
                    .iter()
                    .rposition(|&code| {
                        table
                            .get(usize::from(code))
                            .is_some_and(|value| !value.is_zero())
                    })
                    .map(|last| col * width + last + 1)
            })
            .unwrap_or(0);
        Ok(Self {
            codes,
            tables,
            used,
            width,
            end,
        })
    }

    pub(super) fn live_len(&self) -> usize {
        self.end
    }

    pub(super) fn digit_tables(
        &self,
        decompose: impl Fn(F) -> [i16; 4],
    ) -> Option<Vec<Vec<[i16; 4]>>> {
        let entries = self.tables.iter().map(Vec::len).sum::<usize>();
        if entries > 1 << 16 {
            return None;
        }
        Some(
            self.tables
                .iter()
                .map(|table| table.iter().copied().map(&decompose).collect())
                .collect(),
        )
    }

    pub(super) fn write_digits<const D: usize>(
        &self,
        start: usize,
        tables: &[Vec<[i16; 4]>],
        rows: &mut [[i16; D]],
    ) {
        let [a, b, c, d] = rows else { unreachable!() };
        let mut dest = 0;
        while dest < D {
            let cell = start + dest;
            if cell >= self.end {
                a[dest..].fill(0);
                b[dest..].fill(0);
                c[dest..].fill(0);
                d[dest..].fill(0);
                break;
            }
            let offset = cell % self.width;
            let count = (D - dest).min(self.width - offset);
            let table = &tables[cell / self.width];
            let live = count.min(self.codes.len().saturating_sub(offset));
            for (column, &code) in
                (dest..dest + live).zip(self.codes.get(offset..offset + live).unwrap_or(&[]))
            {
                let digits = table.get(usize::from(code)).copied().unwrap_or([0; 4]);
                a[column] = digits[0];
                b[column] = digits[1];
                c[column] = digits[2];
                d[column] = digits[3];
            }
            let padding = dest + live..dest + count;
            a[padding.clone()].fill(0);
            b[padding.clone()].fill(0);
            c[padding.clone()].fill(0);
            d[padding].fill(0);
            dest += count;
        }
    }

    pub(super) fn used_values(&self) -> impl Iterator<Item = F> + '_ {
        self.tables
            .iter()
            .flat_map(|table| self.used.iter().map(|&i| table[usize::from(i)]))
    }

    pub(super) fn read(&self, mut start: usize, mut out: &mut [F]) {
        while !out.is_empty() {
            if start >= self.end {
                out.fill(F::zero());
                break;
            }
            let col = start / self.width;
            let offset = start % self.width;
            let len = out.len().min(self.width - offset);
            let (part, rest) = out.split_at_mut(len);
            let live = len.min(self.codes.len().saturating_sub(offset));
            if live != 0 {
                let table = &self.tables[col];
                for (cell, &code) in part.iter_mut().zip(&self.codes[offset..][..live]) {
                    *cell = table
                        .get(usize::from(code))
                        .copied()
                        .unwrap_or_else(F::zero);
                }
            }
            part[live..].fill(F::zero());
            start += len;
            out = rest;
        }
    }

    pub(super) fn constant(&self, range: Range<usize>) -> Option<F> {
        if range.is_empty() || range.start >= self.end {
            return Some(F::zero());
        }
        let col = range.start / self.width;
        if col != (range.end - 1) / self.width {
            return None;
        }
        let offset = range.start % self.width;
        if offset >= self.codes.len() {
            return Some(F::zero());
        }
        let indices = &self.codes[offset..(offset + range.len()).min(self.codes.len())];
        let &code = indices.first()?;
        if !indices.iter().all(|&index| index == code) {
            return None;
        }
        let value = self.tables[col]
            .get(usize::from(code))
            .copied()
            .unwrap_or_else(F::zero);
        (indices.len() == range.len() || value.is_zero()).then_some(value)
    }
}
