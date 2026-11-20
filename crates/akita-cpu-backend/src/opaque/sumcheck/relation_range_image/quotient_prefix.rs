//! Compact-witness engine for the leading quotient-factored coefficient rounds.
//!
//! The first `min(coefficient_bits, MAX_PREFIX_ROUNDS)` coefficient rounds run
//! while the witness is still packed signed digits. One scan at construction
//! records everything those rounds need, and the witness is folded to field
//! elements only once, in the last prefix round.
//!
//! - The relation weight `alpha(c) * lane_weight(x)` and every structured
//!   linear weight are linear in the witness, so the scan contracts the lane
//!   coordinate immediately: `alpha_mass[c] = sum_x lane_weight(x) * w(c, x)`,
//!   and each linear source keeps the analogous mass per source lane. A
//!   coefficient round of the relation term is then a product sumcheck between
//!   these short vectors and the folded `alpha` and source tables.
//! - Each witness quad (four consecutive coefficients) is recorded by its digit
//!   class. After the first two challenges a quad's folded value depends only
//!   on its class. Class histograms weighted by the equality factor over the
//!   variables after the quad give the first two range-image messages (through
//!   the `{0, 1, Infinity}^2` grid). For `b <= 8` they also give the third one
//!   (through the even/odd quad split and the histogram of class differences);
//!   basis 16 has too many class differences for a histogram and reads the
//!   third message from the class lookups below.
//! - Each later prefix round reads the folded witness as a sum of class lookups
//!   into challenge-scaled copies of the two-round quad fold. The last prefix
//!   round writes the folded witness for the ordinary suffix path.

use super::prefix_cache::{
    add_basis16_marginals, PrefixBasis, Stage2PrefixCache, BASIS16_MARGINAL_COUNT,
};
use super::wide_mass::WideMass;
use super::*;
use crate::opaque::sumcheck::relation_range_image::evaluation_trace::PreparedLaneWeights;
use crate::opaque::sumcheck::{add_assign_all, parallel_tasks, sum_partials};
use akita_algebra::eq_poly::EqPolynomial;

#[path = "constant_quad_norms.rs"]
mod constant_quad_norms;
use self::constant_quad_norms::ConstantQuadNorms;
#[path = "quad_classes.rs"]
mod quad_classes;
use crate::sources::packed_digits::PackedQuadPairs;
use quad_classes::QuadClasses;

/// Maximum number of coefficient rounds served from the compact witness.
const MAX_PREFIX_ROUNDS: usize = 6;
const _: () = assert!(MAX_PREFIX_ROUNDS >= 2 && MAX_PREFIX_ROUNDS <= 7);

/// Minimum lanes per parallel scan task.
const MIN_SCAN_TASK_LANES: usize = 256;

/// Lanes the scan decodes and adds to the relation masses together.
const SCAN_CHUNK_LANES: usize = 256;
const HISTOGRAM_LOG_TILE: usize = 4;
const HISTOGRAM_TILE: usize = 1 << HISTOGRAM_LOG_TILE;

/// Minimum folded-witness pairs per parallel lookup-round task.
const MIN_LOOKUP_TASK_PAIRS: usize = 1 << 12;

/// Range-image message of one prefix round.
pub(super) enum PrefixNormRound<'a, E: Field> {
    /// Complete message, already scaled by the split-equality scalar.
    Polynomial(UnivariatePoly<E>),
    /// Inner message that the split-equality factor still multiplies.
    Terms(NormRoundTerms<'a, E>),
}

/// Leading-round state built from one scan of the compact witness.
pub(super) struct CompactQuotientPrefix<E: Field> {
    b: usize,
    digit_bits: usize,
    /// Last coefficient round the engine serves; it materializes the witness.
    last_round: usize,
    /// Digit class of every witness quad, in witness order.
    classes: QuadClasses,
    norm_cache: Stage2PrefixCache<E>,
    /// Whether round 2 is read from the histograms below (`b <= 8`).
    histogram_round2: bool,
    /// Round-2 equality weight of each quad class, for even and odd quads.
    even_histogram: Vec<E>,
    odd_histogram: Vec<E>,
    /// Round-2 equality weight of each odd-minus-even quad digit difference.
    delta_histogram: Vec<E>,
    relation_moments: CoefficientRelationMoments<E>,
    /// Folded value of every quad class after the first two challenges.
    quad_fold: Vec<E>,
    challenges: Vec<E>,
}

/// Radix of one digit of a quad difference: differences span `2b - 1` values.
#[inline]
fn delta_radix(b: usize) -> usize {
    2 * b - 1
}

#[inline(always)]
fn quad_class<const DIGIT_BITS: usize>(quad: [i8; 4]) -> u16 {
    // Adding `b / 2` to a digit in `[-b / 2, b / 2)` flips the top bit of its
    // low `DIGIT_BITS` bits; two shift-merge steps then pack the four bytes.
    let bytes = u32::from_le_bytes(quad.map(|digit| digit as u8));
    let biased =
        (bytes & (((1 << DIGIT_BITS) - 1) * 0x0101_0101)) ^ ((1 << (DIGIT_BITS - 1)) * 0x0101_0101);
    let pairs =
        (biased | (biased >> (8 - DIGIT_BITS))) & (((1 << (2 * DIGIT_BITS)) - 1) * 0x0001_0001);
    ((pairs | (pairs >> (16 - 2 * DIGIT_BITS))) & ((1 << (4 * DIGIT_BITS)) - 1)) as u16
}

struct ScanLayout<'a, E: Field> {
    witness: PackedSignedDigitView<'a>,
    packed_lanes: bool,
    lane_weights: &'a [E],
    linear_terms: &'a PreparedProverLinearTerms<E>,
    /// First source-mass block of each linear source.
    source_blocks: &'a [usize],
    source_block_count: usize,
    eq_low: &'a [E],
    eq_high: &'a [E],
    eq_tile: Option<&'a [E]>,
    tau2: E,
    b: usize,
    coeff_count: usize,
    class_count: usize,
    /// Per-class base-`2b - 1` digit code; empty when round 2 is not served.
    delta_code: &'a [i32],
    delta_offset: i32,
    delta_count: usize,
}

struct ScanTotals<E: Field> {
    alpha_mass: Vec<E>,
    source_mass: Vec<E>,
    even_histogram: Vec<E>,
    odd_histogram: Vec<E>,
    delta_histogram: Vec<E>,
}

impl<E: Field> ScanTotals<E> {
    fn merge(mut self, other: Self) -> Self {
        add_assign_all(&mut self.alpha_mass, &other.alpha_mass);
        add_assign_all(&mut self.source_mass, &other.source_mass);
        add_assign_all(&mut self.even_histogram, &other.even_histogram);
        add_assign_all(&mut self.odd_histogram, &other.odd_histogram);
        add_assign_all(&mut self.delta_histogram, &other.delta_histogram);
        self
    }
}

/// Coefficient-direction contractions shared by compact and ordinary factored rounds.
pub(super) struct CoefficientRelationMoments<E: Field> {
    alpha_mass: Vec<E>,
    source_mass: Vec<Vec<E>>,
}

impl<E: Field> CoefficientRelationMoments<E> {
    fn new(alpha_mass: Vec<E>, source_mass: Vec<Vec<E>>) -> Self {
        Self {
            alpha_mass,
            source_mass,
        }
    }

    pub(super) fn relation_message(
        &self,
        alpha: &[E],
        linear_terms: &PreparedProverLinearTerms<E>,
    ) -> RoundMessage<E> {
        let mut rel = RoundMessage::zero();
        let masses = std::iter::once((&self.alpha_mass, alpha)).chain(
            self.source_mass
                .iter()
                .zip(linear_terms.sources.iter().map(|source| &source.values[..])),
        );
        for (mass, weight) in masses {
            debug_assert_eq!(mass.len(), weight.len());
            for (mass, weight) in mass.chunks_exact(2).zip(weight.chunks_exact(2)) {
                rel.at_one += mass[1] * weight[1];
                rel.quadratic += (mass[1] - mass[0]) * (weight[1] - weight[0]);
            }
        }
        rel
    }
}

impl<E: Field + Fold> CoefficientRelationMoments<E> {
    pub(super) fn bind(&mut self, challenge: E) {
        fold_evals_in_place(&mut self.alpha_mass, challenge);
        for mass in &mut self.source_mass {
            if !mass.is_empty() {
                fold_evals_in_place(mass, challenge);
            }
        }
    }
}

struct CoefficientMassChunk<E: Field> {
    alpha_mass: Vec<E>,
    source_mass: Vec<E>,
}

impl<E: Field> CoefficientMassChunk<E> {
    fn merge(mut self, other: Self) -> Self {
        add_assign_all(&mut self.alpha_mass, &other.alpha_mass);
        add_assign_all(&mut self.source_mass, &other.source_mass);
        self
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_basis32_lane_range<E: Field + Unreduced + 'static>(
    witness: PackedSignedDigitView<'_>,
    lane_weights: &[E],
    linear_terms: &PreparedProverLinearTerms<E>,
    source_blocks: &[usize],
    source_block_count: usize,
    coeff_count: usize,
    chunk_lanes: usize,
    digit_capacity: usize,
    first_lane: usize,
    end_lane: usize,
) -> Option<CoefficientMassChunk<E>> {
    let mut alpha_mass = WideMass::<E, 32>::new(1, coeff_count, 32);
    let mut source_mass = WideMass::<E, 32>::new(source_block_count, coeff_count, 32);
    if first_lane == end_lane {
        return Some(CoefficientMassChunk {
            alpha_mass: alpha_mass.finish(),
            source_mass: source_mass.finish(),
        });
    }
    let mut digits = vec![0i8; digit_capacity];
    let mut alpha_rows = Vec::with_capacity(chunk_lanes);
    let mut source_rows = BlockRows::<E, 32>::new(source_block_count);
    let mut first = first_lane;
    while first < end_lane {
        let lanes = (end_lane - first).min(chunk_lanes);
        let digit_len = lanes * coeff_count;
        let chunk = &mut digits[..digit_len];
        witness.decode_range(first * coeff_count, chunk).ok()?;
        alpha_rows.clear();
        alpha_rows.extend(
            lane_weights[first..first + lanes]
                .iter()
                .copied()
                .zip(0u32..),
        );
        alpha_mass.add_rows(0, &alpha_rows, chunk);
        for row in 0..lanes {
            linear_terms.for_each_source_term(first + row, |factor, source_index, source_lane| {
                let block = source_blocks[source_index] + source_lane;
                source_rows.push(block, factor, row as u32);
            });
        }
        source_rows.add_to(&mut source_mass, chunk);
        first += lanes;
    }
    Some(CoefficientMassChunk {
        alpha_mass: alpha_mass.finish(),
        source_mass: source_mass.finish(),
    })
}

impl<E: Field + Unreduced + 'static> CoefficientRelationMoments<E> {
    /// Contract a basis-32 packed witness without building any norm-class table.
    pub(super) fn from_basis32(
        witness: &PackedSignedDigits,
        lane_weights: &[E],
        linear_terms: &PreparedProverLinearTerms<E>,
        live_lane_count: usize,
        coeff_count: usize,
    ) -> Option<Self> {
        if coeff_count == 0
            || !coeff_count.is_power_of_two()
            || linear_terms.coeff_count != coeff_count
            || matches!(&linear_terms.lane_weights, PreparedLaneWeights::Dense(_))
            || live_lane_count.checked_mul(coeff_count)? != witness.len()
            || lane_weights.len() < live_lane_count
            || !witness.bounds().fits_balanced_log_basis(5)
        {
            return None;
        }
        let mut source_blocks = Vec::with_capacity(linear_terms.sources.len());
        let mut source_block_count = 0usize;
        for source in &linear_terms.sources {
            source_blocks.push(source_block_count);
            source_block_count = source_block_count.checked_add(source.lane_count)?;
        }
        source_block_count.checked_mul(coeff_count)?;
        let chunk_lanes = SCAN_CHUNK_LANES;
        let digit_capacity = chunk_lanes.checked_mul(coeff_count)?;

        let task_lanes = live_lane_count
            .div_ceil(parallel_tasks(TASKS_PER_THREAD))
            .max(MIN_SCAN_TASK_LANES);
        let task_count = live_lane_count.div_ceil(task_lanes).max(1);
        let witness = witness.view();
        let scan = |task: usize| {
            let first_lane = task * task_lanes;
            let end_lane = first_lane + (live_lane_count - first_lane).min(task_lanes);
            scan_basis32_lane_range(
                witness,
                lane_weights,
                linear_terms,
                &source_blocks,
                source_block_count,
                coeff_count,
                chunk_lanes,
                digit_capacity,
                first_lane,
                end_lane,
            )
        };
        #[cfg(feature = "parallel")]
        let totals = if task_count == 1 {
            scan(0)?
        } else {
            (0..task_count)
                .into_par_iter()
                .map(scan)
                .reduce_with(|left, right| Some(left?.merge(right?)))??
        };
        #[cfg(not(feature = "parallel"))]
        let totals = (0..task_count)
            .map(scan)
            .reduce(|left, right| Some(left?.merge(right?)))??;
        let source_mass = source_blocks
            .iter()
            .zip(&linear_terms.sources)
            .map(|(&block, source)| {
                let start = block.checked_mul(coeff_count)?;
                let end = block
                    .checked_add(source.lane_count)?
                    .checked_mul(coeff_count)?;
                Some(totals.source_mass.get(start..end)?.to_vec())
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self::new(totals.alpha_mass, source_mass))
    }
}

/// Source-mass rows of one scan chunk, grouped by mass block.
struct BlockRows<E, const MAX_DIGIT_BASIS: usize = 16> {
    /// `(block, factor, row)` in visit order.
    visits: Vec<(usize, E, u32)>,
    /// Per-block row count; zero outside [`add_to`](Self::add_to).
    counts: Vec<usize>,
    /// Blocks with at least one row, in first-visit order.
    touched: Vec<usize>,
    grouped: Vec<(E, u32)>,
}

impl<E: Field + Unreduced + 'static, const MAX_DIGIT_BASIS: usize> BlockRows<E, MAX_DIGIT_BASIS> {
    fn new(block_count: usize) -> Self {
        Self {
            visits: Vec::new(),
            counts: vec![0; block_count],
            touched: Vec::new(),
            grouped: Vec::new(),
        }
    }

    #[inline]
    fn push(&mut self, block: usize, factor: E, row: u32) {
        if self.counts[block] == 0 {
            self.touched.push(block);
        }
        self.counts[block] += 1;
        self.visits.push((block, factor, row));
    }

    /// Add every row to its block of `mass`, one call per block, and clear.
    fn add_to(&mut self, mass: &mut WideMass<E, MAX_DIGIT_BASIS>, digits: &[i8]) {
        // Counting sort: `counts` becomes each block's write cursor.
        let mut start = 0;
        for &block in &self.touched {
            let count = self.counts[block];
            self.counts[block] = start;
            start += count;
        }
        self.grouped.clear();
        self.grouped.resize(self.visits.len(), (E::zero(), 0));
        for &(block, factor, row) in &self.visits {
            self.grouped[self.counts[block]] = (factor, row);
            self.counts[block] += 1;
        }
        let mut start = 0;
        for &block in &self.touched {
            let end = self.counts[block];
            for rows in self.grouped[start..end].chunks(mass.max_rows()) {
                mass.add_rows(block, rows, digits);
            }
            self.counts[block] = 0;
            start = end;
        }
        self.visits.clear();
        self.touched.clear();
    }
}

#[inline(always)]
fn add_paired_class<E: Field>(
    histograms: &mut [&mut [E]; 3],
    delta_code: &[i32],
    delta_offset: i32,
    [even, odd]: [u16; 2],
    weight: E,
) {
    let (even, odd) = (usize::from(even), usize::from(odd));
    histograms[0][even] += weight;
    histograms[1][odd] += weight;
    if !histograms[2].is_empty() {
        let delta = delta_code[odd] - delta_code[even] + delta_offset;
        histograms[2][delta as usize] += weight;
    }
}

fn add_class_tiles<E: Field>(
    classes: &[[u16; 2]],
    first_pair: usize,
    [eq_low, eq_high, eq_tile]: [&[E]; 3],
    delta_code: &[i32],
    delta_offset: i32,
    histograms: &mut [&mut [E]; 3],
) {
    debug_assert!(first_pair.is_multiple_of(HISTOGRAM_TILE));
    debug_assert!(classes.len().is_multiple_of(HISTOGRAM_TILE));
    let low_mask = eq_low.len() - 1;
    let low_bits = eq_low.len().trailing_zeros();
    for (tile_index, tile) in classes.as_chunks::<HISTOGRAM_TILE>().0.iter().enumerate() {
        let pair = first_pair + tile_index * HISTOGRAM_TILE;
        let low_start = pair & low_mask;
        let high = eq_high[pair >> low_bits];
        if tile[1..].iter().all(|&entry| entry == tile[0]) {
            let weight = high * eq_tile[low_start >> HISTOGRAM_LOG_TILE];
            add_paired_class(histograms, delta_code, delta_offset, tile[0], weight);
        } else {
            let low = &eq_low[low_start..low_start + HISTOGRAM_TILE];
            for (&entry, &weight) in tile.iter().zip(low) {
                add_paired_class(histograms, delta_code, delta_offset, entry, high * weight);
            }
        }
    }
}

fn zero_chunk_weight<E: Field>(
    eq_tile: &[E],
    eq_high: &[E],
    first_pair: usize,
    pair_count: usize,
) -> E {
    let width = eq_tile.len();
    let mut tile = first_pair / HISTOGRAM_TILE;
    let end = tile + pair_count / HISTOGRAM_TILE;
    let mut weight = E::zero();
    while tile < end {
        let high = tile / width;
        let low = tile % width;
        let count = (width - low).min(end - tile);
        let low_weight = if count == width {
            E::one()
        } else {
            eq_tile[low..low + count]
                .iter()
                .copied()
                .fold(E::zero(), |a, b| a + b)
        };
        weight += low_weight * eq_high[high];
        tile += count;
    }
    weight
}

fn active_packed_lanes(bytes: &[[u8; 48]]) -> [u64; SCAN_CHUNK_LANES / 64] {
    let mut active = [0; SCAN_CHUNK_LANES / 64];
    for (bits, lanes) in active.iter_mut().zip(bytes.chunks(64)) {
        *bits = lanes.iter().enumerate().fold(0, |mask, (lane, bytes)| {
            mask | (u64::from(*bytes != [0; 48]) << lane)
        });
    }
    active
}

/// Scan the lanes whose quad classes `classes` receives.
///
/// With at least eight coefficients per lane, the round-2 pair of a quad is the
/// adjacent quad in the same lane, so even and odd quads get separate
/// histograms. With four coefficients the pair crosses lanes; only the mixed
/// histogram for the first two rounds exists, and the engine stops at round 1.
fn scan_lanes<E: Field + Unreduced + 'static, const DIGIT_BITS: usize>(
    layout: &ScanLayout<'_, E>,
    first_lane: usize,
    mut classes: Option<&mut [u16]>,
    lane_count: usize,
) -> ScanTotals<E> {
    let coeff_count = layout.coeff_count;
    let quads_per_lane = coeff_count / 4;
    let split_pairs = quads_per_lane >= 2;
    let low_mask = layout.eq_low.len() - 1;
    let low_bits = layout.eq_low.len().trailing_zeros();
    let one_minus_tau2 = E::one() - layout.tau2;
    let zero_class = quad_class::<DIGIT_BITS>([0; 4]);

    let mut alpha_mass = WideMass::<E>::new(1, coeff_count, layout.b);
    let mut source_mass = WideMass::<E>::new(layout.source_block_count, coeff_count, layout.b);
    let chunk_lanes = SCAN_CHUNK_LANES.min(alpha_mass.max_rows());
    let mut digits = vec![0i8; chunk_lanes * coeff_count];
    let mut alpha_rows = Vec::with_capacity(chunk_lanes);
    let mut source_rows = BlockRows::<E>::new(layout.source_block_count);
    let mut even_histogram = vec![
        E::zero();
        if DIGIT_BITS == 4 {
            BASIS16_MARGINAL_COUNT
        } else {
            layout.class_count
        }
    ];
    let mut odd_histogram = if split_pairs && DIGIT_BITS != 4 {
        vec![E::zero(); layout.class_count]
    } else {
        Vec::new()
    };
    let mut delta_histogram = vec![E::zero(); layout.delta_count];

    let mut tile = if classes.is_none() {
        vec![0u16; lane_count.min(chunk_lanes) * quads_per_lane]
    } else {
        Vec::new()
    };
    for start in (0..lane_count).step_by(chunk_lanes) {
        let lanes = (lane_count - start).min(chunk_lanes);
        let chunk_first = first_lane + start;
        let chunk_classes = match classes.as_deref_mut() {
            Some(classes) => &mut classes[start * quads_per_lane..(start + lanes) * quads_per_lane],
            None => &mut tile[..lanes * quads_per_lane],
        };
        let digits = &mut digits[..lanes * coeff_count];
        let range = chunk_first * coeff_count..(chunk_first + lanes) * coeff_count;
        let packed = layout
            .packed_lanes
            .then(|| layout.witness.aligned_bytes(range.clone()))
            .flatten();
        let active = packed.map(|encoded| active_packed_lanes(encoded.as_chunks::<48>().0));
        let is_active =
            |row: usize| active.is_none_or(|bits| bits[row / 64] & (1 << (row % 64)) != 0);
        if active.is_some_and(|bits| bits == [0; SCAN_CHUNK_LANES / 64]) {
            let pairs_per_lane = quads_per_lane / 2;
            let tiles = layout
                .eq_tile
                .expect("packed scan requires aligned equality tiles");
            let weight = zero_chunk_weight(
                tiles,
                layout.eq_high,
                chunk_first * pairs_per_lane,
                lanes * pairs_per_lane,
            );
            chunk_classes.fill(zero_class);
            add_paired_class(
                &mut [
                    even_histogram.as_mut_slice(),
                    odd_histogram.as_mut_slice(),
                    delta_histogram.as_mut_slice(),
                ],
                layout.delta_code,
                layout.delta_offset,
                [zero_class; 2],
                weight,
            );
            continue;
        }
        layout
            .witness
            .decode_range(range.start, digits)
            .expect("compact prefix lanes are in bounds");
        alpha_rows.clear();
        for (row, &weight) in layout.lane_weights[chunk_first..chunk_first + lanes]
            .iter()
            .enumerate()
        {
            if !is_active(row) {
                continue;
            }
            let row_id = row as u32;
            alpha_rows.push((weight, row_id));
            layout.linear_terms.for_each_source_term(
                chunk_first + row,
                |factor, source_index, source_lane| {
                    source_rows.push(
                        layout.source_blocks[source_index] + source_lane,
                        factor,
                        row_id,
                    );
                },
            );
        }
        alpha_mass.add_rows(0, &alpha_rows, digits);
        source_rows.add_to(&mut source_mass, digits);

        for ((local_lane, lane_classes), lane_digits) in chunk_classes
            .chunks_exact_mut(quads_per_lane)
            .enumerate()
            .zip(digits.chunks_exact(coeff_count))
        {
            let lane = chunk_first + local_lane;
            if is_active(local_lane) {
                for (class, &quad) in lane_classes.iter_mut().zip(lane_digits.as_chunks::<4>().0) {
                    *class = quad_class::<DIGIT_BITS>(quad);
                }
            } else {
                lane_classes.fill(zero_class);
            }

            if split_pairs {
                let pairs = lane_classes.as_chunks::<2>().0;
                let first_pair = lane * pairs.len();
                let mut histograms = [
                    even_histogram.as_mut_slice(),
                    odd_histogram.as_mut_slice(),
                    delta_histogram.as_mut_slice(),
                ];
                if let Some(eq_tile) = layout.eq_tile {
                    add_class_tiles(
                        pairs,
                        first_pair,
                        [layout.eq_low, layout.eq_high, eq_tile],
                        layout.delta_code,
                        layout.delta_offset,
                        &mut histograms,
                    );
                    continue;
                }
                for (pair_offset, &pair) in pairs.iter().enumerate() {
                    let pair_index = first_pair + pair_offset;
                    let weight = layout.eq_low[pair_index & low_mask]
                        * layout.eq_high[pair_index >> low_bits];
                    if DIGIT_BITS == 4 {
                        let [even, odd] = pair;
                        if even == odd {
                            add_basis16_marginals(histograms[0], even, weight);
                        } else {
                            let odd_weight = layout.tau2 * weight;
                            add_basis16_marginals(histograms[0], even, weight - odd_weight);
                            add_basis16_marginals(histograms[0], odd, odd_weight);
                        }
                    } else {
                        add_paired_class(
                            &mut histograms,
                            layout.delta_code,
                            layout.delta_offset,
                            pair,
                            weight,
                        );
                    }
                }
            } else {
                let pair_index = lane >> 1;
                let side = if lane & 1 == 0 {
                    one_minus_tau2
                } else {
                    layout.tau2
                };
                let weight = side
                    * layout.eq_low[pair_index & low_mask]
                    * layout.eq_high[pair_index >> low_bits];
                if DIGIT_BITS == 4 {
                    add_basis16_marginals(&mut even_histogram, lane_classes[0], weight);
                } else {
                    even_histogram[usize::from(lane_classes[0])] += weight;
                }
            }
        }
    }

    ScanTotals {
        alpha_mass: alpha_mass.finish(),
        source_mass: source_mass.finish(),
        even_histogram,
        odd_histogram,
        delta_histogram,
    }
}

fn append_basis16_table<E: Field>(table: &mut Vec<E>, weights: [E; 4]) {
    let [d0, d1, d2, d3] = weights
        .map(|weight| std::array::from_fn::<_, 16, _>(|i| weight * E::from_i64(i as i64 - 8)));
    let mut lower = [E::zero(); 256];
    for (&high, row) in d1.iter().zip(lower.chunks_exact_mut(16)) {
        for (out, &low) in row.iter_mut().zip(&d0) {
            *out = high + low;
        }
    }
    for &high in &d3 {
        for &low in &d2 {
            let upper = high + low;
            table.extend(lower.iter().map(|&x| x + upper));
        }
    }
}

/// Accumulate the range-image inner terms of pairs `pairs` of the folded
/// witness at round `2 + log2(G)`, optionally writing the folded pairs.
///
/// Folded entry `m` is `sum_{i < G} lookups[i][classes[m * G + i]]`, where
/// `lookups[i]` is the two-round quad fold scaled by the equality weight of
/// `i` over the challenges after the first two.
#[allow(clippy::too_many_arguments)]
fn lookup_pairs<E: Field + Unreduced, const G: usize, const SKIP_LINEAR: bool>(
    lookups: &[E],
    class_count: usize,
    classes: &[u16],
    eq_low: &[E],
    eq_high: &[E],
    pairs: std::ops::Range<usize>,
    out: Option<&mut [E]>,
) -> [E; 3] {
    let fold = |entry: usize| {
        let entry = &classes[entry * G..(entry + 1) * G];
        let mut value = lookups[usize::from(entry[0])];
        for (j, &c) in entry.iter().enumerate().skip(1) {
            value += lookups[j * class_count + usize::from(c)];
        }
        value
    };
    let uniform = |start: usize, end: usize| {
        let tile = &classes[start * 2 * G..end * 2 * G];
        let first = &tile[..2 * G];
        first == &tile[tile.len() - 2 * G..] && tile.chunks_exact(2 * G).all(|p| p == first)
    };
    lookup_pairs_by::<E, G, SKIP_LINEAR>(class_count, fold, uniform, eq_low, eq_high, pairs, out)
}

fn lookup_pairs_by<E: Field + Unreduced, const G: usize, const SKIP_LINEAR: bool>(
    class_count: usize,
    fold_entry: impl Fn(usize) -> E,
    uniform: impl Fn(usize, usize) -> bool,
    eq_low: &[E],
    eq_high: &[E],
    pairs: std::ops::Range<usize>,
    mut out: Option<&mut [E]>,
) -> [E; 3] {
    let low_mask = eq_low.len() - 1;
    let low_bits = eq_low.len().trailing_zeros();
    const LOOKUP_TILE_PAIRS: usize = 64;
    let mut lookup_tile = [[E::zero(); 2]; LOOKUP_TILE_PAIRS];
    let tile_pairs = if G == 2 && class_count <= 4096 {
        16
    } else {
        LOOKUP_TILE_PAIRS
    };
    let materialize = out.is_some();
    let mut totals = [E::zero(); 3];
    let mut block_start = pairs.start;
    while block_start < pairs.end {
        let high = block_start >> low_bits;
        let block_end = ((high + 1) << low_bits).min(pairs.end);
        let mut inner = ProductNorm::<E, SKIP_LINEAR>::zero();
        if G == 1 {
            for pair in block_start..block_end {
                let w0 = fold_entry(2 * pair);
                let w1 = fold_entry(2 * pair + 1);
                if let Some(out) = out.as_deref_mut() {
                    let local = 2 * (pair - pairs.start);
                    out[local] = w0;
                    out[local + 1] = w1;
                }
                let dw = w1 - w0;
                let e_in = eq_low[pair & low_mask];
                inner.add(w0, dw, e_in);
            }
        } else {
            for start in (block_start..block_end).step_by(tile_pairs) {
                let end = (start + tile_pairs).min(block_end);
                let count = end - start;
                let folded = if let Some(out) = out.as_deref_mut() {
                    let offset = 2 * (start - pairs.start);
                    out[offset..offset + 2 * count].as_chunks_mut::<2>().0
                } else {
                    &mut lookup_tile[..count]
                };
                let weight_start = start & low_mask;
                let weights = &eq_low[weight_start..weight_start + count];
                if count >= 8 && uniform(start, end) {
                    let left = fold_entry(2 * start);
                    let right = fold_entry(2 * start + 1);
                    if materialize {
                        folded.fill([left, right]);
                    }
                    if !left.is_zero() || !right.is_zero() {
                        inner.add(left, right - left, weights.iter().copied().sum());
                    }
                    continue;
                }
                for (pair, [left, right]) in (start..end).zip(folded.iter_mut()) {
                    *left = fold_entry(2 * pair);
                    *right = fold_entry(2 * pair + 1);
                }
                for (&[left, right], &weight) in folded.iter().zip(weights) {
                    inner.add(left, right - left, weight);
                }
            }
        }
        let e_out = eq_high[high];
        for (total, inner) in totals.iter_mut().zip(inner.reduce()) {
            *total += e_out * inner;
        }
        block_start = block_end;
    }
    totals
}

impl<E: Field + Ring + Unreduced> CompactQuotientPrefix<E> {
    /// Scan the compact witness, or return `None` when its geometry or digit
    /// range is outside the engine's domain.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "CompactQuotientPrefix::new")]
    pub(super) fn new(
        witness: &PackedSignedDigits,
        lane_weights: &[E],
        linear_terms: &PreparedProverLinearTerms<E>,
        split_eq: &GruenSplitEq<E>,
        stage1_point: &[E],
        batching_coeff: E,
        b: usize,
        live_lane_count: usize,
        coefficient_bits: usize,
        metal_kernels: bool,
    ) -> Result<Option<Self>, AkitaError>
    where
        E: 'static,
    {
        let (basis, digit_bits) = match b {
            4 => (PrefixBasis::B4, 2),
            8 => (PrefixBasis::B8, 3),
            16 => (PrefixBasis::B16, 4),
            _ => return Ok(None), // Other bases use the ordinary coefficient-round path.
        };
        if coefficient_bits < 2
            || stage1_point.len() < 3
            || !witness.bounds().fits_balanced_log_basis(digit_bits as u32)
            || matches!(linear_terms.lane_weights, PreparedLaneWeights::Dense(_))
            || lane_weights.len() < live_lane_count
            || witness.len() != live_lane_count << coefficient_bits
        {
            return Ok(None); // Unsupported geometry uses the ordinary coefficient-round path.
        }
        // The optimization needs equality tables after two prefix folds.
        let Some((eq_low, eq_high)) = split_eq.remaining_eq_tables_after(2) else {
            return Ok(None);
        };
        let last_round = coefficient_bits.min(MAX_PREFIX_ROUNDS) - 1;
        let coeff_count = 1usize << coefficient_bits;
        let quads_per_lane = coeff_count / 4;
        let class_count = 1usize << (4 * digit_bits);

        let source_blocks = linear_terms
            .sources
            .iter()
            .scan(0usize, |block, source| {
                let start = *block;
                *block += source.lane_count;
                Some(start)
            })
            .collect::<Vec<_>>();
        let source_block_count = linear_terms
            .sources
            .iter()
            .map(|source| source.lane_count)
            .sum();

        let serves_round2 = last_round >= 2 && b <= 8;
        let radix = delta_radix(b);
        let delta_code = if serves_round2 {
            (0..class_count)
                .map(|class| {
                    (0..4).rev().fold(0i32, |code, digit| {
                        code * radix as i32 + ((class >> (digit_bits * digit)) & (b - 1)) as i32
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let delta_offset = ((b - 1) * (1 + radix + radix * radix + radix * radix * radix)) as i32;
        let eq_tile = if b == 8 && quads_per_lane / 2 >= HISTOGRAM_TILE {
            // Summing over the four low variables uses the existing suffix
            // equality table. This only applies when the high table is unchanged.
            split_eq
                .remaining_eq_tables_after(2 + HISTOGRAM_LOG_TILE)
                .filter(|&(low, high)| {
                    low.len() * HISTOGRAM_TILE == eq_low.len() && high.len() == eq_high.len()
                })
                .map(|(low, _)| low)
        } else {
            None
        };
        let layout = ScanLayout {
            witness: witness.view(),
            packed_lanes: witness.bit_width() == 3 && coeff_count == 128 && eq_tile.is_some(),
            lane_weights,
            linear_terms,
            source_blocks: &source_blocks,
            source_block_count,
            eq_low,
            eq_high,
            eq_tile,
            tau2: stage1_point[2],
            b,
            coeff_count,
            class_count,
            delta_code: &delta_code,
            delta_offset,
            delta_count: if serves_round2 { radix.pow(4) } else { 0 },
        };

        let task_lanes = live_lane_count
            .div_ceil(parallel_tasks(TASKS_PER_THREAD))
            .max(MIN_SCAN_TASK_LANES);
        let count = live_lane_count * quads_per_lane;
        let mut classes = QuadClasses::new::<E>(
            witness,
            count,
            metal_kernels && layout.packed_lanes && last_round >= 3,
        );
        let scan = |task: usize, output: Option<&mut [u16]>| {
            let start = task * task_lanes;
            let lanes = (live_lane_count - start).min(task_lanes);
            match digit_bits {
                2 => scan_lanes::<E, 2>(&layout, start, output, lanes),
                3 => scan_lanes::<E, 3>(&layout, start, output, lanes),
                _ => scan_lanes::<E, 4>(&layout, start, output, lanes),
            }
        };
        #[cfg(feature = "parallel")]
        let totals = match &mut classes {
            QuadClasses::Expanded(data) => data
                .par_chunks_mut(task_lanes * quads_per_lane)
                .enumerate()
                .map(|(task, out)| scan(task, Some(out)))
                .reduce_with(ScanTotals::merge),
            QuadClasses::Packed { .. } => (0..live_lane_count.div_ceil(task_lanes))
                .into_par_iter()
                .map(|task| scan(task, None))
                .reduce_with(ScanTotals::merge),
        };
        #[cfg(not(feature = "parallel"))]
        let totals = match &mut classes {
            QuadClasses::Expanded(data) => data
                .chunks_mut(task_lanes * quads_per_lane)
                .enumerate()
                .map(|(task, out)| scan(task, Some(out)))
                .reduce(ScanTotals::merge),
            QuadClasses::Packed { .. } => (0..live_lane_count.div_ceil(task_lanes))
                .map(|task| scan(task, None))
                .reduce(ScanTotals::merge),
        };

        let Some(totals) = totals else {
            return Ok(None);
        };

        let norm_cache = if matches!(basis, PrefixBasis::B16) {
            Stage2PrefixCache::from_basis16_marginals(
                &totals.even_histogram,
                stage1_point[0],
                stage1_point[1],
                batching_coeff,
            )
        } else {
            let tau2 = stage1_point[2];
            let mixed_histogram;
            let histogram = if quads_per_lane >= 2 {
                mixed_histogram = totals
                    .even_histogram
                    .iter()
                    .zip(&totals.odd_histogram)
                    .map(|(&even, &odd)| even + tau2 * (odd - even))
                    .collect::<Vec<_>>();
                &mixed_histogram
            } else {
                &totals.even_histogram
            };
            Stage2PrefixCache::from_norm_histogram(
                histogram,
                basis,
                stage1_point[0],
                stage1_point[1],
                batching_coeff,
            )?
        };
        let (even_histogram, odd_histogram) = if serves_round2 {
            (totals.even_histogram, totals.odd_histogram)
        } else {
            (Vec::new(), Vec::new())
        };
        let source_mass = source_blocks
            .iter()
            .zip(&linear_terms.sources)
            .map(|(&block, source)| {
                totals.source_mass[block * coeff_count..(block + source.lane_count) * coeff_count]
                    .to_vec()
            })
            .collect();

        Ok(Some(Self {
            b,
            digit_bits,
            last_round,
            classes,
            norm_cache,
            histogram_round2: serves_round2,
            even_histogram,
            odd_histogram,
            delta_histogram: totals.delta_histogram,
            relation_moments: CoefficientRelationMoments::new(totals.alpha_mass, source_mass),
            quad_fold: Vec::new(),
            challenges: Vec::new(),
        }))
    }

    #[inline]
    pub(super) fn challenges(&self) -> &[E] {
        &self.challenges
    }

    #[inline]
    pub(super) fn last_round(&self) -> usize {
        self.last_round
    }

    /// Relation and structured-linear message of the current round.
    pub(super) fn relation_message(
        &self,
        alpha: &[E],
        linear_terms: &PreparedProverLinearTerms<E>,
    ) -> RoundMessage<E> {
        self.relation_moments.relation_message(alpha, linear_terms)
    }

    /// Range-image message of a round that keeps the witness compact.
    pub(super) fn norm_round<'a>(
        &self,
        split_eq: &GruenSplitEq<E>,
        recovery: Option<PreparedLinearQRecovery<'a, E>>,
    ) -> PrefixNormRound<'a, E> {
        match self.challenges.len() {
            0 => PrefixNormRound::Polynomial(self.norm_cache.round0_norm_poly()),
            1 => PrefixNormRound::Polynomial(self.norm_cache.round1_norm_poly(self.challenges[0])),
            2 if self.histogram_round2 => PrefixNormRound::Terms(self.round2_norm_terms()),
            round => {
                PrefixNormRound::Terms(self.lookup_round_terms(round, split_eq, recovery, None))
            }
        }
    }

    /// Materialize the witness folded by every challenge so far and return it
    /// with the range-image message of the last prefix round.
    pub(super) fn materialize<'a>(
        &self,
        witness: PackedSignedDigitView<'_>,
        split_eq: &GruenSplitEq<E>,
        recovery: Option<PreparedLinearQRecovery<'a, E>>,
    ) -> (Vec<E>, PrefixNormRound<'a, E>) {
        debug_assert_eq!(self.challenges.len(), self.last_round);
        match self.last_round {
            1 => {
                let r0 = self.challenges[0];
                let half = (self.b / 2) as i16;
                let lut = CompactPairFoldLut::from_contiguous_range(-half, half - 1, r0);
                (
                    RelationRoundState::<E>::materialize_compact_witness(witness, &lut),
                    PrefixNormRound::Polynomial(self.norm_cache.round1_norm_poly(r0)),
                )
            }
            2 if self.histogram_round2 => (
                cfg_into_iter!(0..self.classes.len())
                    .map(|index| self.quad_fold[usize::from(self.classes.at(index))])
                    .collect(),
                PrefixNormRound::Terms(self.round2_norm_terms()),
            ),
            round => {
                let mut folded = vec![E::zero(); self.classes.len() >> (round - 2)];
                let terms = self.lookup_round_terms(round, split_eq, recovery, Some(&mut folded));
                (folded, PrefixNormRound::Terms(terms))
            }
        }
    }

    /// Round-2 range-image terms from the class histograms.
    ///
    /// The folded witness at a quad is linear in its digits, so the round-2
    /// slope of an even/odd quad pair is the quad fold of their digit
    /// difference.
    fn round2_norm_terms<'a>(&self) -> NormRoundTerms<'a, E> {
        let mut at_zero = E::zero();
        let mut at_one = E::zero();
        for ((&fold, &even), &odd) in self
            .quad_fold
            .iter()
            .zip(&self.even_histogram)
            .zip(&self.odd_histogram)
        {
            let norm = fold * (fold + E::one());
            at_zero += even * norm;
            at_one += odd * norm;
        }

        let radix = delta_radix(self.b);
        let quad_weights = self.quad_weights();
        let digit_terms: [Vec<E>; 4] = std::array::from_fn(|digit| {
            (0..radix)
                .map(|value| quad_weights[digit] * E::from_i64(value as i64 - (self.b as i64 - 1)))
                .collect()
        });
        let mut at_infinity = E::zero();
        // The scan allocates radix^4 entries and never resizes this histogram.
        // Nested chunks expose each complete row directly, without a fallible next().
        debug_assert_eq!(self.delta_histogram.len(), radix.pow(4));
        let delta_plane_len = radix * radix;
        let delta_cube_len = delta_plane_len * radix;
        let cubes = self.delta_histogram.chunks_exact(delta_cube_len);
        for (d3, cube) in digit_terms[3].iter().zip(cubes) {
            for (d2, plane) in digit_terms[2]
                .iter()
                .zip(cube.chunks_exact(delta_plane_len))
            {
                let high = *d3 + *d2;
                for (d1, row) in digit_terms[1].iter().zip(plane.chunks_exact(radix)) {
                    let upper = high + *d1;
                    for (&weight, d0) in row.iter().zip(&digit_terms[0]) {
                        if !weight.is_zero() {
                            let slope = upper + *d0;
                            at_infinity += weight * (slope * slope);
                        }
                    }
                }
            }
        }
        NormRoundTerms::Full([at_zero, at_one - at_zero - at_infinity, at_infinity])
    }

    /// Equality weights of the quad corners `[00, 10, 01, 11]` at `(r0, r1)`.
    fn quad_weights(&self) -> [E; 4] {
        let (r0, r1) = (self.challenges[0], self.challenges[1]);
        let (s0, s1) = (E::one() - r0, E::one() - r1);
        [s0 * s1, r0 * s1, s0 * r1, r0 * r1]
    }

    fn lookup_round_terms<'a>(
        &self,
        round: usize,
        split_eq: &GruenSplitEq<E>,
        recovery: Option<PreparedLinearQRecovery<'a, E>>,
        out: Option<&mut [E]>,
    ) -> NormRoundTerms<'a, E> {
        let totals = match (round, recovery.is_some()) {
            (2, false) => self.lookup_round::<1, false>(split_eq, out),
            (2, true) => self.lookup_round::<1, true>(split_eq, out),
            (3, false) => self.lookup_round::<2, false>(split_eq, out),
            (3, true) => self.lookup_round::<2, true>(split_eq, out),
            (4, false) => self.lookup_round::<4, false>(split_eq, out),
            (4, true) => self.lookup_round::<4, true>(split_eq, out),
            (5, false) => self.lookup_round::<8, false>(split_eq, out),
            (5, true) => self.lookup_round::<8, true>(split_eq, out),
            (6, false) => self.lookup_round::<16, false>(split_eq, out),
            (6, true) => self.lookup_round::<16, true>(split_eq, out),
            _ => unreachable!("compact prefix rounds are capped by MAX_PREFIX_ROUNDS"),
        };
        NormRoundTerms::from_totals(totals, recovery)
    }

    fn lookup_round<const G: usize, const SKIP_LINEAR: bool>(
        &self,
        split_eq: &GruenSplitEq<E>,
        out: Option<&mut [E]>,
    ) -> [E; 3] {
        #[cfg(akita_fold_metal)]
        let mut out = out;
        let class_count = 1 << (4 * self.digit_bits);
        let group_weights = EqPolynomial::evals(&self.challenges[2..])
            .expect("compact prefix challenge count is bounded");
        debug_assert_eq!(group_weights.len(), G);
        let mut scaled;
        let lookups = if self.b == 16 {
            scaled = Vec::with_capacity(G * class_count);
            let weights = self.quad_weights();
            for &weight in &group_weights {
                append_basis16_table(&mut scaled, weights.map(|digit| weight * digit));
            }
            &scaled
        } else if G == 1 {
            &self.quad_fold
        } else {
            scaled = group_weights
                .iter()
                .flat_map(|&weight| self.quad_fold.iter().map(move |&fold| weight * fold))
                .collect();
            &scaled
        };
        let (eq_low, eq_high) = split_eq.remaining_eq_tables();
        let pair_count = self.classes.len() / (2 * G);
        #[cfg(akita_fold_metal)]
        if let QuadClasses::Packed {
            metal: Some(metal), ..
        } = &self.classes
        {
            // SAFETY: Packed construction selects Ext2<Prime64Offset59>. Each pair covers 2*G quads.
            if let Some(sums) = unsafe {
                metal.evaluate::<E, G, SKIP_LINEAR>(
                    pair_count,
                    lookups,
                    eq_low,
                    eq_high,
                    out.as_deref_mut(),
                )
            } {
                return sums;
            }
        }
        let task_pairs = pair_count
            .div_ceil(parallel_tasks(TASKS_PER_THREAD))
            .max(MIN_LOOKUP_TASK_PAIRS);
        let constant_quad_norms =
            (G == 2 && SKIP_LINEAR && self.b == 8 && eq_low.len() >= 256 && out.is_none())
                .then(|| ConstantQuadNorms::new([group_weights[0], group_weights[1]]));
        let run = |task: usize, out: Option<&mut [E]>| {
            let start = task * task_pairs;
            let pairs = start..(start + task_pairs).min(pair_count);
            match &self.classes {
                QuadClasses::Expanded(classes) => {
                    if let Some(norms) = &constant_quad_norms {
                        if ConstantQuadNorms::<E>::supports(classes, &pairs) {
                            return norms.sum(lookups, classes, eq_low, eq_high, pairs);
                        }
                    }
                    lookup_pairs::<E, G, SKIP_LINEAR>(
                        lookups,
                        class_count,
                        classes,
                        eq_low,
                        eq_high,
                        pairs,
                        out,
                    )
                }
                QuadClasses::Packed { witness, .. } => {
                    let raw = PackedQuadPairs::new(witness.view())
                        .expect("packed quad witness was validated at construction");
                    let fold = |entry: usize| {
                        (0..G).fold(E::zero(), |sum, j| {
                            sum + lookups[j * class_count + usize::from(raw.at(entry * G + j))]
                        })
                    };
                    lookup_pairs_by::<E, G, SKIP_LINEAR>(
                        class_count,
                        fold,
                        |s, e| raw.uniform::<G>(s, e),
                        eq_low,
                        eq_high,
                        pairs,
                        out,
                    )
                }
            }
        };
        let parts: Vec<[E; 3]> = match out {
            Some(out) => cfg_chunks_mut!(out, 2 * task_pairs)
                .enumerate()
                .map(|(task, chunk)| run(task, Some(chunk)))
                .collect(),
            None => cfg_into_iter!(0..pair_count.div_ceil(task_pairs))
                .map(|task| run(task, None))
                .collect(),
        };
        sum_partials(E::zero(), parts)
    }
}

/// A few tasks per worker for load balance.
const TASKS_PER_THREAD: usize = 4;

impl<E: Field + Ring + Unreduced + Fold> CompactQuotientPrefix<E> {
    /// Bind the relation masses and record `r`.
    pub(super) fn bind(&mut self, r: E) {
        self.relation_moments.bind(r);
        self.challenges.push(r);
        if self.challenges.len() == 3 {
            self.even_histogram = Vec::new();
            self.odd_histogram = Vec::new();
            self.delta_histogram = Vec::new();
        }
        if self.challenges.len() == 2 && self.b != 16 {
            let half = (self.b / 2) as i64;
            let quad_weights = self.quad_weights();
            let digit_terms: [Vec<E>; 4] = std::array::from_fn(|digit| {
                (0..self.b as i64)
                    .map(|value| quad_weights[digit] * E::from_i64(value - half))
                    .collect()
            });
            let mask = self.b - 1;
            self.quad_fold = (0..1usize << (4 * self.digit_bits))
                .map(|class| {
                    (0..4).fold(E::zero(), |sum, digit| {
                        sum + digit_terms[digit][(class >> (self.digit_bits * digit)) & mask]
                    })
                })
                .collect();
        }
    }
}

impl<E: Field + Ring + Unreduced> RelationRangeImageProver<E> {
    #[cfg(test)]
    pub(super) fn compact_quotient_prefix(&self) -> Option<&CompactQuotientPrefix<E>> {
        match &self.phase {
            Phase::CompactPrefix { engine, .. } => Some(engine),
            Phase::Coefficient { .. } | Phase::Lane { .. } => None,
        }
    }
}

impl<E: Field + Ring + Unreduced> RelationRoundState<E> {
    pub(super) fn norm_poly_from_prefix(&self, norm: PrefixNormRound<'_, E>) -> UnivariatePoly<E> {
        match norm {
            PrefixNormRound::Polynomial(poly) => poly,
            PrefixNormRound::Terms(terms) => self.norm_poly_from_terms(terms),
        }
    }

    /// `(combined, range-image)` messages of a compact-prefix round.
    pub(super) fn compact_prefix_round_message(
        &self,
        prefix: &CompactQuotientPrefix<E>,
        weights: &RelationWeightFactorization<E>,
    ) -> (RoundMessage<E>, UnivariatePoly<E>) {
        let norm = prefix.norm_round(&self.split_eq, self.split_eq.prepare_linear_q_recovery());
        let norm_poly = self.norm_poly_from_prefix(norm);
        let mut message =
            prefix.relation_message(weights.common_alpha_factor(), &self.linear_terms);
        message.add_assign(RoundMessage::from_polynomial(&norm_poly));
        (message, norm_poly)
    }
}

impl<E: Field + Ring + Unreduced + Fold> RelationRoundState<E> {
    /// Bind `r` in a compact-prefix round. The round before the last prefix
    /// round also materializes the folded witness and caches the last prefix
    /// round's message.
    pub(super) fn ingest_compact_prefix_challenge(
        &mut self,
        weights: &mut RelationWeightFactorization<E>,
        engine: &mut CompactQuotientPrefix<E>,
        r: E,
    ) {
        fold_evals_in_place(weights.common_alpha_factor_mut(), r);
        self.split_eq.bind(r);
        self.linear_terms.fold_coefficients(r);
        engine.bind(r);
    }
}

#[cfg(test)]
#[path = "basis16_lookup_tests.rs"]
mod basis16_lookup_tests;

#[cfg(test)]
#[path = "class_tile_histogram_tests.rs"]
mod class_tile_histogram_tests;

#[cfg(test)]
#[path = "tests/packed_quad_rounds.rs"]
mod packed_quad_rounds;
