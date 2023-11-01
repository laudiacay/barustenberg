use std::ops::AddAssign;

use ark_ec::scalar_mul::glv::GLVConfig;
use ark_ec::short_weierstrass::{Affine, SWCurveConfig};
use ark_ec::AffineRepr;
use ark_ff::{AdditiveGroup, Field, PrimeField};
use get_msb::Msb;

use crate::{
    common::max_threads::compute_num_threads,
    ecc::scalar_multiplication::{process_buckets::process_buckets, wnaf::fixed_wnaf_with_counts},
    numeric::bitop::get_msb,
};

use super::{
    cube_root_of_unity,
    runtime_states::{
        get_num_rounds, get_optimal_bucket_width, AffinePippengerRuntimeState,
        PippengerRuntimeState,
    },
};

#[inline]
fn count_bits(bucket_counts: &[u32], bit_offsets: &mut [u32], num_buckets: usize, num_bits: usize) {
    for i in 0..num_buckets {
        let count = bucket_counts[i];
        for j in 0..num_bits as usize {
            bit_offsets[j + 1] += count & (1u32 << j)
        }
    }
    bit_offsets[0] = 0;
    for i in 2..(num_bits + 1) {
        bit_offsets[i] += bit_offsets[i - 1];
    }
}

pub(crate) fn generate_pippenger_point_table<C: SWCurveConfig>(
    points: &[Affine<C>],
    table: &mut [Affine<C>],
    num_points: usize,
) {
    // calculate the cube root of unity
    let beta = cube_root_of_unity::<C::BaseField>();

    // iterate backwards, so that `points` and `table` can point to the same memory location
    for i in (0..num_points).rev() {
        table[i * 2] = points[i];
        table[i * 2 + 1].x = beta * points[i].x;
        table[i * 2 + 1].y = -points[i].y;
    }
}

/// Compute the windowed-non-adjacent-form versions of our scalar multipliers.
///
/// We start by splitting our 254 bit scalars into 2 127-bit scalars, using the short weierstrass curve endomorphism
/// (for a point P \in \G === (x, y) \in \Fq, then (\beta x, y) = (\lambda) * P , where \beta = 1^{1/3} mod Fq and
///\lambda = 1^{1/3} mod Fr) (which means we can represent a scalar multiplication (k * P) as (k1 * P + k2 * \lambda *
///P), where k1, k2 have 127 bits) (see field::split_into_endomorphism_scalars for more details)
///
/// Once we have our 127-bit scalar multipliers, we determine the optimal number of pippenger rounds, given the number of
///points we're multiplying. Once we have the number of rounds, `m`, we need to split our scalar into `m` bit-slices.
///Each pippenger round will work on one bit-slice.
///
/// Pippenger's algorithm works by, for each round, iterating over the points we're multplying. For each point, we
///examing the point's scalar multiplier and extract the bit-slice associated with the current pippenger round (we start
///with the most significant slice). We then use the bit-slice to index a 'bucket', which we add the point into. For
///example, if the bit slice is 01101, we add the corresponding point into bucket[13].
///
/// At the end of each pippenger round we concatenate the buckets together. E.g. if we have 8 buckets, we compute:
/// sum = bucket[0] + 2 * bucket[1] + 3 * bucket[2] + 4 * bucket[3] + 5 * bucket[4] + 6 * bucket[5] + 7 * bucket[6] + 8 *
///bucket[7].
///
/// At the end of each pippenger round, the bucket sum will contain the scalar multiplication result for one bit slice.
/// For example, say we have 16 rounds, where each bit slice contains 8 bits (8 * 16 = 128, enough to represent our 127
///bit scalars). At the end of the first round, we will have taken the 8 most significant bits from every scalar
///multiplier. Our bucket sum will be the result of a mini-scalar-multiplication, where we have multiplied every point by
///the 8 most significant bits of each point's scalar multiplier.
///
/// We repeat this process for every pippenger round. In our example, this gives us 16 bucket sums.
/// We need to multiply the most significant bucket sum by 2^{120}, the second most significant bucket sum by 2^{112}
///etc. Once this is done we can add the bucket sums together, to evaluate our scalar multiplication result.
///
/// Pippenger has complexity O(n / logn), because of two factors at play: the number of buckets we need to concatenate
///per round, and the number of points we need to add into buckets per round.
///
/// To minimize the number of point additions per round, we want fewer rounds. But fewer rounds increases the number of
///bucket concatenations. The more points we have, the greater the time saving when reducing the number of rounds, which
///means we can afford to have more buckets per round.
///
/// For a concrete example, with 2^20 points, the sweet spot is 2^15 buckets - with 2^15 buckets we can evaluate our 127
///bit scalar multipliers in 8 rounds (we can represent b-bit windows with 2^{b-1} buckets, more on that below).
///
/// This means that, for each round, we add 2^21 points into buckets (we've split our scalar multpliers into two
///half-width multipliers, so each round has twice the number of points. This is the reason why the endormorphism is
///useful here; without the endomorphism, we would need twice the number of buckets for each round).
///
/// We also concatenate 2^15 buckets for each round. This requires 2^16 point additions.
///
/// Meaning that the total number of point additions is (8 * 2^21) + (8 * 2^16) = 33 * 2^19 ~ 2^24 point additions.
/// If we were to use a simple Montgomery double-and-add ladder to exponentiate each point, we would need 2^27 point
///additions (each scalar multiplier has ~2^7 non-zero bits, and there are 2^20 points).
///
/// This makes pippenger 8 times faster than the naive O(n) equivalent. Given that a circuit with 1 million gates will
///require 9 multiple-scalar-multiplications with 2^20 points, efficiently using Pippenger's algorithm is essential for
///fast provers
///
/// One additional efficiency gain is the use of 2^{b-1} buckets to represent b bits. To do this we represent our
///bit-slices in non-adjacent form. Non-adjacent form represents values using a base, where each 'bit' can take the
///values (-1, 0, 1). This is considerably more efficient than binary form for scalar multiplication, as inverting a
///point can be done by negating the y-coordinate.
///
/// We actually use a slightly different representation than simple non-adjacent form. To represent b bits, a bit slice
///contains values from (-2^{b} - 1, ..., -1, 1, ..., 2^{b} - 1). i.e. we only have odd values. We do this to eliminate
///0-valued windows, as having a conditional branch in our hot loop to check if an entry is 0 is somethin we want to
///avoid.
///
/// The above representation can be used to represent any binary number as long as we add a 'skew' factor. Each scalar
///multiplier's `skew` tracks if the scalar multiplier is even or odd. If it's even, `skew = true`, and we add `1` to our
///multiplier to make it odd.
///
/// We then, at the end of the Pippenger algorithm, subtract a point from the total result, if that point's skew is
///`true`.
///
/// At the end of `compute_wnaf_states`, `state.wnaf_table` will contain our wnaf entries, but unsorted.
///
/// # Type Parameters
/// * `C`: The curve configuration and GLV configuration
///
/// # Arguments
/// * `point_schedule`: Pointer to the output array with all WNAFs
/// * `input_skew_table`: Pointer to the output array with all skews
/// * `round_counts`: The number of points in each round
/// * `scalars`: The pointer to the region with initial scalars that need to be converted into WNAF
/// * `num_initial_points`: The number of points before the endomorphism split
pub(crate) fn compute_wnaf_states<C: SWCurveConfig + GLVConfig>(
    point_schedule: &mut Vec<u64>,
    input_skew_table: &mut Vec<bool>,
    round_counts: &mut Vec<u64>,
    scalars: &mut [C::ScalarField],
    num_initial_points: usize,
) {
    let num_points = num_initial_points * 2;

    let num_rounds = get_num_rounds(num_initial_points * 2);
    let num_threads = compute_num_threads();

    let bits_per_bucket = get_optimal_bucket_width(num_initial_points);

    let wnaf_bits = bits_per_bucket + 1;

    let num_initial_points_per_thread = num_initial_points / num_threads;
    let num_points_per_thread = num_points / num_threads;

    const MAX_NUM_ROUNDS: usize = 256;
    const MAX_NUM_THREADS: usize = 128;
    let mut thread_round_counts = vec![vec![0u64; MAX_NUM_ROUNDS]; MAX_NUM_THREADS];

    for i in 0..num_threads {
        let wnaf_table = &mut point_schedule
            [2 * i * num_initial_points_per_thread..2 * (i + 1) * num_initial_points_per_thread];
        let skew_table = &mut input_skew_table
            [2 * i * num_initial_points_per_thread..2 * (i + 1) * num_initial_points_per_thread];
        let thread_scalars = &mut scalars
            [i * num_initial_points_per_thread..(i + 1) * num_initial_points_per_thread];
        let offset = i * num_points_per_thread;

        for j in 0..num_initial_points_per_thread {
            // TODO: check if have to convert from montgomery form
            // t0 = thread_scalars[j].from_montgomery_form();
            let t0 = thread_scalars[j];

            let ((sgn_t1, mut t1), (sgn_t2, mut t2)) = C::scalar_decomposition(t0);

            if !sgn_t1 {
                t1 = -t1;
            }
            if !sgn_t2 {
                t2 = -t2;
            }

            fixed_wnaf_with_counts(
                t1.into_bigint().into().to_u64_digits(),
                &mut &mut wnaf_table[(j << 1)..],
                &mut skew_table[j << 1],
                &mut thread_round_counts[i][0..],
                (((j as u64) << 1) + offset as u64) << 32,
                num_points,
                wnaf_bits,
            );

            fixed_wnaf_with_counts(
                t2.into_bigint().into().to_u64_digits(),
                &mut wnaf_table[(j << 1) + 1..],
                &mut skew_table[(j << 1) + 1],
                &mut thread_round_counts[i][0..],
                (((j as u64) << 1) + 1 + offset as u64) << 32,
                num_points,
                wnaf_bits,
            );
        }
    }

    for i in 0..num_threads {
        for j in 0..num_rounds {
            round_counts[j] += thread_round_counts[i][j];
        }
    }
}

pub(crate) fn organise_buckets(point_schedule: &mut Vec<u64>, num_points: usize) {
    let num_rounds: usize = get_num_rounds(num_points);

    for i in 0..num_rounds {
        process_buckets(
            point_schedule[i * num_points..i + 1 * num_points].as_mut(),
            num_points,
            (get_optimal_bucket_width(num_points / 2) + 1) as u32,
        )
    }
}

//Placeholder for asm implementation in barretenberg that uses arkworks library to grab the y-coordinate of the affine element, invert it, then copy the element.
//Inline notation is added for thrills but is probs useless w/o the assembly but we gotta keep appearances up #CLOUT_CHASE
//This is based off of : https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/ecc/groups/group_impl_asm.tcc#L55
#[inline]
fn conditionally_negate_affine<C: SWCurveConfig>(
    src: &Affine<C>,
    dest: &mut Affine<C>,
    predicate: u64,
) {
    if predicate != 0 {
        *dest = src.clone();
        let mut y = dest
            .y()
            .expect("Failed to source y coordinate of affine element while conditionally negating");
        //This may not be needed???? adds additionally assignment???
        y = y.inverse().unwrap();
    } else {
        *dest = src.clone();
    }
}

/// This method sorts our points into our required base-2 sequences.
/// `max_bucket_bits` is log2(maximum bucket count).
/// This sets the upper limit on how many iterations we need to perform in `evaluate_addition_chains`.
/// e.g. if `max_bucket_bits == 3`, then we have at least one bucket with >= 8 points in it.
/// which means we need to repeat our pairwise addition algorithm 3 times
/// (e.g. add 4 pairs together to get 2 pairs, add those pairs together to get a single pair, which we add to
/// reduce to our final point)
///
/// # Arguments
/// * `state`: runtime state of the Pippenger algorithm
/// * `empty_bucket_counts`: Whether or not to fill bucket count with number of non-zero windows in first iteration
pub(crate) fn construct_addition_chains<C: SWCurveConfig>(
    state: &mut AffinePippengerRuntimeState<C>,
    empty_bucket_counts: bool,
) -> usize {
    // if this is the first call to `construct_addition_chains`, we need to count up our buckets
    if empty_bucket_counts {
        state.bucket_counts.fill(0);
        //TODO: Note this is const in c++ implementation.
        let first_bucket = (state.point_schedule[0] & 0x7fffffffu64) as usize;
        for i in 0..state.num_points {
            let bucket_index = (state.point_schedule[i] & 0x7fffffffu64) as usize;
            state.bucket_counts[bucket_index - first_bucket] += 1;
        }
        for i in 0..state.num_buckets {
            state.bucket_counts[i] = if state.bucket_counts[i] == 0 {
                1u32
            } else {
                0u32
            };
        }
    }

    let mut max_count = 0u32;
    for i in 0..state.num_buckets {
        max_count = if state.bucket_counts[i] > max_count {
            state.bucket_counts[i]
        } else {
            max_count as u32
        }
    }

    let max_bucket_bits = max_count.get_msb() as usize;

    for i in 0..(max_bucket_bits + 1) {
        state.bit_offsets[i] = 0;
    }

    // theoretically, can be unrolled using templated methods.
    // However, explicitly unrolling the loop by using recursive template calls was slower!
    // Inner loop is currently bounded by a constexpr variable, need to see what the compiler does with that...
    count_bits(
        &state.bucket_counts,
        &mut state.bit_offsets,
        state.num_buckets,
        max_bucket_bits,
    );

    // we need to update `bit_offsets` to compute our point shuffle,
    // but we need the original array later on, so make a copy.
    let bit_offsets_copy = state.bit_offsets.clone();

    // this is where we take each bucket's associated points, and arrange them
    // in a pairwise order, so that we can compute large sequences of additions using the affine trick
    let mut schedule_it = 0;
    let bucket_count_it: &mut Vec<_> = state.bucket_counts.as_mut();

    for _i in 0..state.num_buckets {
        let count = bucket_count_it[0];
        bucket_count_it[0] += 1;
        let num_bits = count.get_msb() + 1;
        for j in 0..num_bits as usize {
            let mut current_offset = bit_offsets_copy[j] as usize;
            let k_end = count & (1u32 << j);
            // This section is a bottleneck - to populate our point array, we need
            // to read from memory locations that are effectively uniformly randomly distributed!
            // (assuming our scalar multipliers are uniformly random...)
            // In the absence of a more elegant solution, we use ugly macro hacks to try and
            // unroll loops, and prefetch memory a few cycles before we need it
            match k_end {
                64 | 32 | 16 => {
                    for k in 0..(k_end >> 4) {
                        //TODO: implement BBERG_SCALAR_MULTIPLICATION_FETCH_BLOCK for now just print msg
                        println!("BBERG_SCALAR_MULTIPLICATION_FETCH_BLOCK");
                    }
                    break;
                }
                8 => {
                    let schedule_a = state.point_schedule[schedule_it];
                    let schedule_b = state.point_schedule[schedule_it + 1];
                    let schedule_c = state.point_schedule[schedule_it + 2];
                    let schedule_d = state.point_schedule[schedule_it + 3];
                    let schedule_e = state.point_schedule[schedule_it + 4];
                    let schedule_f = state.point_schedule[schedule_it + 5];
                    let schedule_g = state.point_schedule[schedule_it + 6];
                    let schedule_h = state.point_schedule[schedule_it + 7];

                    conditionally_negate_affine(
                        &state.points[(schedule_a >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset],
                        (schedule_a >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_b >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 1],
                        (schedule_b >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_c >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 2],
                        (schedule_c >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_d >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 3],
                        (schedule_d >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_e >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 4],
                        (schedule_e >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_f >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 5],
                        (schedule_f >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_g >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 6],
                        (schedule_g >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_h >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 7],
                        (schedule_h >> 31u64) & 1u64,
                    );

                    current_offset += 8;
                    schedule_it += 8;
                    break;
                }
                4 => {
                    let schedule_a = state.point_schedule[schedule_it];
                    let schedule_b = state.point_schedule[schedule_it + 1];
                    let schedule_c = state.point_schedule[schedule_it + 2];
                    let schedule_d = state.point_schedule[schedule_it + 3];

                    conditionally_negate_affine(
                        &state.points[(schedule_a >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset],
                        (schedule_a >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_b >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 1],
                        (schedule_b >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_c >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 2],
                        (schedule_c >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_d >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 3],
                        (schedule_d >> 31u64) & 1u64,
                    );

                    current_offset += 4;
                    schedule_it += 4;
                    break;
                }
                2 => {
                    let schedule_a = state.point_schedule[schedule_it];
                    let schedule_b = state.point_schedule[schedule_it + 1];

                    conditionally_negate_affine(
                        &state.points[(schedule_a >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset],
                        (schedule_a >> 31u64) & 1u64,
                    );
                    conditionally_negate_affine(
                        &state.points[(schedule_b >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset + 1],
                        (schedule_b >> 31u64) & 1u64,
                    );

                    current_offset += 2;
                    schedule_it += 2;
                    break;
                }
                1 => {
                    let schedule_a = state.point_schedule[schedule_it];

                    conditionally_negate_affine(
                        &state.points[(schedule_a >> 32u64) as usize],
                        &mut state.point_pairs_1[current_offset],
                        (schedule_a >> 31u64) & 1u64,
                    );

                    current_offset += 1;
                    schedule_it += 1;
                    break;
                }
                0 => break,
                _ => {
                    for _ in 0..k_end {
                        let schedule = state.point_schedule[schedule_it];
                        let predicate = (schedule >> 31u64) & 1u64;

                        conditionally_negate_affine(
                            &state.points[(schedule >> 32u64) as usize],
                            &mut state.point_pairs_1[current_offset],
                            predicate,
                        );
                        current_offset += 1;
                        schedule_it += 1;
                    }
                }
            }
        }
    }

    return max_bucket_bits;
}

fn add_affine_point_with_edge_cases<F: Field, G: AffineRepr<BaseField = F>>(
    points: &mut [G],
    num_points: usize,
    scratch_space: &mut [F],
) {
    //Fq
    let mut batch_inversion_accumulator = F::one();

    for i in (0..num_points).step_by(2) {
        if points[i].is_zero() || points[i + 1].is_zero() {
            continue;
        }
        let (x1, y1) = points[i]
            .xy()
            .expect("Failed to grab points from x1 in first part of affine addition");
        let (mut x2, mut y2) = points[i + 1]
            .xy()
            .expect("Failed to grab points from x2 in first part of affine addition");
        if x1 == x2 {
            if y1 == y2 {
                scratch_space[i >> 1] = x1.double(); // 2x
                let x_squared = x1.square();
                x2 = y1.double(); // 2y
                y2 = x_squared + x_squared + x_squared; // 3x^2
                y2 *= batch_inversion_accumulator;
                batch_inversion_accumulator *= x2;
                continue;
            }
            //set to infinity
            points[i] = G::zero();
            points[i + 1] = G::zero();
            continue;
        }
        scratch_space[i >> 1] = x1 + x2; // x2 + x1
        x2 -= x1; // x2 - x1
        y2 -= y2; // y2 - y1
        y2 *= batch_inversion_accumulator; // (y2 - y1) * accumulator_old
        batch_inversion_accumulator *= x2;
    }

    if batch_inversion_accumulator.is_zero() {
        batch_inversion_accumulator = batch_inversion_accumulator.inverse().unwrap();
    }

    for i in ((num_points - 2)..0).step_by(2) {
        // TODO: add builtin prefetch

        if points[i].is_zero() {
            points[(i + num_points) >> 1] = points[i + 1];
            continue;
        }

        if points[i + 1].is_zero() {
            points[(i + num_points) >> 1] = points[i];
            continue;
        }

        let (mut x1, y1) = points[i]
            .xy()
            .expect("Failed to grab points from x1 in first part of affine addition");
        let (mut x2, mut y2) = points[i + 1]
            .xy()
            .expect("Failed to grab points from x2 in first part of affine addition");
        let (mut x3, _) = points[(i + num_points) >> 1]
            .xy()
            .expect("Failed to grab points from x2 in first part of affine addition");

        y2 *= batch_inversion_accumulator; //update accumulator
        batch_inversion_accumulator *= x2;
        x2 = y2.square();
        x3 = x2 - scratch_space[i >> 1]; // x3 = lambda_squared - x2 - x1?

        x1 -= x3;
        x1 *= y2;
        x3 = x1 - y1;
    }
}

/// adds a bunch of points together using affine addition formulae.
/// Paradoxically, the affine formula is crazy efficient if you have a lot of independent point additions to perform.
/// Affine formula:
///
/// \lambda = (y_2 - y_1) / (x_2 - x_1)
/// x_3 = \lambda^2 - (x_2 + x_1)
/// y_3 = \lambda*(x_1 - x_3) - y_1
///
/// Traditionally, we avoid affine formulae like the plague, because computing lambda requires a modular inverse,
/// which is outrageously expensive.
///
/// However! We can use Montgomery's batch inversion technique to amortise the cost of the inversion to ~0.
///
/// The way batch inversion works is as follows. Let's say you want to compute \{ 1/x_1, 1/x_2, ..., 1/x_n \}
/// The trick is to compute the product x_1x_2...x_n , whilst storing all of the temporary products.
/// i.e. we have an array A = [x_1, x_1x_2, ..., x_1x_2...x_n]
/// We then compute a single inverse: I = 1 / x_1x_2...x_n
/// Finally, we can use our accumulated products, to quotient out individual inverses.
/// We can get an individual inverse at index i, by computing I.A_{i-1}.(x_nx_n-1...x_i+1)
/// The last product term we can compute on-the-fly, as it grows by one element for each additional inverse that we
/// require.
///
/// TLDR: amortized cost of a modular inverse is 3 field multiplications per inverse.
/// Which means we can compute a point addition with SIX field multiplications in total.
/// The traditional Jacobian-coordinate formula requires 11.
///
/// There is a catch though - we need large sequences of independent point additions!
/// i.e. the output from one point addition in the sequence is NOT an input to any other point addition in the
///sequence.
///
/// We can re-arrange the Pippenger algorithm to get this property, but it's...complicated
///
/// # Type Parameters
/// * `F`: the base field of the curve
/// * `G`: the affine representation of the curve with base field F
///
/// # Parameters
/// * `points`: the points to add together
/// * `num_points`: the number of points
/// * `scratch_space`: scratch space to use for intermediate calculations
pub(crate) fn add_affine_points<F: Field, G: AffineRepr<BaseField = F>>(
    points: &mut [G],
    num_points: usize,
    scratch_space: &mut [F],
) {
    //Fq
    let mut batch_inversion_accumulator = F::one();

    for i in (0..num_points).step_by(2) {
        let (x1, y1) = points[i]
            .xy()
            .expect("Failed to grab points from x1 in first part of affine addition");
        let (mut x2, mut y2) = points[i + 1]
            .xy()
            .expect("Failed to grab points from x2 in first part of affine addition");
        scratch_space[i >> 1] = x1 + x2; // x2 + x1
        x2 -= x1; // x2 - x1
        y2 -= y1; // y2 - y1
        y2 *= batch_inversion_accumulator; // (y2 - y1) * accumulator_old
        batch_inversion_accumulator *= x2;
    }

    if batch_inversion_accumulator.is_zero() {
        panic!("attempted to invert zero in add_affine_points");
    } else {
        batch_inversion_accumulator = batch_inversion_accumulator.inverse().unwrap();
    }

    for i in ((num_points - 2)..0).step_by(2) {
        // TODO: add builtin prefetch
        let (mut x1, y1) = points[i]
            .xy()
            .expect("Failed to grab points from x1 in first part of affine addition");
        let (mut x2, mut y2) = points[i + 1]
            .xy()
            .expect("Failed to grab points from x2 in first part of affine addition");
        let (mut x3, _) = points[(i + num_points) >> 1]
            .xy()
            .expect("Failed to grab points from x3 in first part of affine addition");

        y2 *= batch_inversion_accumulator; //update accumulator
        batch_inversion_accumulator *= x2;
        x2 = y2.square();
        x3 = x2 - scratch_space[i >> 1]; // x3 = lambda_squared - x2 - x1?

        x1 -= x3;
        x1 *= y2;
        x3 = x1 - y1;
    }
}

///
/// evaluate a chain of pairwise additions.
/// The additions are sequenced into base-2 segments
/// i.e. pairs, pairs of pairs, pairs of pairs of pairs etc
/// `max_bucket_bits` indicates the largest set of nested pairs in the array,
/// which defines the iteration depth
///
/// # Type Parameters
/// * `C`: the curve configuration
///
/// # Parameters
/// * `state`: the pippenger runtime state with the points to add
/// * `max_bucket_bits`: the maximum depth of the buckets
/// * `handle_edge_cases`: whether or not to handle edge cases in the addition.
pub(crate) fn evaluate_addition_chains<C: SWCurveConfig>(
    state: &mut AffinePippengerRuntimeState<C>,
    max_bucket_bits: usize,
    handle_edge_cases: bool,
) {
    let end = state.num_points;
    for i in 0..max_bucket_bits {
        let points_in_round = (state.num_points - state.bit_offsets[i + 1] as usize) >> i;
        let start = end - points_in_round;
        let round_points = &mut state.point_pairs_1[start..start + points_in_round];
        if handle_edge_cases {
            add_affine_point_with_edge_cases(
                round_points,
                points_in_round,
                &mut state.scratch_space,
            );
        } else {
            add_affine_points(round_points, points_in_round, &mut state.scratch_space);
        }
    }
}

///
/// This is the entry point for our 'find a way of evaluating a giant multi-product using affine coordinates'
///algorithm By this point, we have already sorted our pippenger buckets. So we have the following situation:
///
/// 1. We have a defined number of buckets points
/// 2. We have a defined number of points, that need to be added into these bucket points
/// 3. number of points >> number of buckets
///
/// The algorithm begins by counting the number of points assigned to each bucket.
/// For each bucket, we then take this count and split it into its base-2 components.
/// e.g. if bucket[3] has 14 points, we split that into a sequence of (8, 4, 2)
/// This base-2 splitting is useful, because we can take the bucket's associated points, and
/// sort them into pairs, quads, octs etc. These mini-addition sequences are independent from one another,
/// which means that we can use the affine trick to evaluate them.
/// Once we're done, we have effectively reduced the number of points in the bucket to a logarithmic factor of the
///input. e.g. in the above example, once we've evaluated our pairwise addition of 8, 4 and 2 elements, we're left
///with 3 points. The next step is to 'play it again Sam', and recurse back into `reduce_buckets`, with our reduced
///number of points. We repeat this process until every bucket only has one point assigned to it.
///
/// # Type Parameters
/// * `C`: the curve configuration
///
/// # Parameters
/// * `state`: the pippenger runtime state with the points to add
/// * `first_round`: whether or not this is the first round of the pippenger algorithm
/// * `handle_edge_cases`: whether or not to handle edge cases in the addition.
pub(crate) fn reduce_buckets<C: SWCurveConfig>(
    state: &mut AffinePippengerRuntimeState<C>,
    first_round: bool,
    handle_edge_cases: bool,
) -> Vec<Affine<C>> {
    let max_bucket_bits = construct_addition_chains(state, first_round);

    // if max_bucket_bits is 0, we're done! we can return
    if max_bucket_bits == 0 {
        return state.point_pairs_1.clone();
    }

    // compute our required additions using the affine trick
    evaluate_addition_chains(state, max_bucket_bits, handle_edge_cases);

    // this next step is a processing step, that computes a new point schedule for our reduced points.
    // In the pippenger algorithm, we use a 64-bit uint to categorize each point.
    // The high 32 bits describes the position of the point in a point array.
    // The low 31 bits describes the bucket index that the point maps to
    // The 32nd bit defines whether the point is actually a negation of our stored point.

    // We want to compute these 'point schedule' uints for our reduced points, so that we can recurse back into
    // `reduce_buckets`
    let end: u32 = state.num_points as u32;

    // The output of `evaluate_addition_chains` has a bit of an odd structure, should probably refactor.
    // Effectively, we used to have one big 1d array, and the act of computing these pair-wise point additions
    // has chopped it up into sequences of smaller 1d arrays, with gaps in between
    for i in 0..max_bucket_bits as usize {
        let points_in_round = (state.num_points as u32 - state.bit_offsets[i + 1]) >> i;
        let points_removed = points_in_round / 2;
        let start = end - points_in_round;
        let modified_start = start + points_removed;
        state.bit_offsets[i + 1] = modified_start;
    }

    // iterate over each bucket. Identify how many remaining points there are, and compute their point scheduels
    let mut new_num_points = 0;
    for i in 0..state.num_buckets {
        let count = state.bucket_counts[i];
        let num_bits = count.get_msb() + 1;
        let mut new_bucket_count = 0;
        for j in 0..num_bits {
            let mut current_offset = state.bit_offsets[i];
            let has_entry = ((count >> j) & 1) == 1;
            if has_entry {
                let schedule: u64 = ((current_offset as u64) << 32) + (i as u64);
                state.point_schedule[new_num_points] = schedule;
                new_num_points += 1;
                new_bucket_count += 1;
                current_offset += 1; // TODO: check if this addition makes sense
            }
        }
        state.bucket_counts[i] = new_bucket_count;
    }

    state.num_points = new_num_points;
    // TODO: check these clones
    let temp: Vec<Affine<C>> = state.point_pairs_1.clone();
    state.points = state.point_pairs_1.clone();
    state.point_pairs_1 = state.point_pairs_2.clone();
    state.point_pairs_2 = temp;

    return reduce_buckets(state, false, handle_edge_cases);
}

fn evaluate_pippenger_rounds<C: SWCurveConfig>(
    state: &PippengerRuntimeState<C>,
    points: Vec<Affine<C>>,
    num_points: usize,
    handle_edge_cases: bool,
) -> Affine<C> {
    let num_threads = compute_num_threads();
    let num_rounds = get_num_rounds(num_points);
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);

    let mut thread_accumulators = vec![Affine::default(); num_threads];

    for j in 0..num_threads {
        for i in 0..num_rounds {
            let num_round_points = state.round_counts[i] as usize;
            let mut accumulator = Affine::default();

            if num_round_points == 0 || (num_round_points < num_threads && j < num_threads - 1) {
                // skip if round points not enough for thread parallelism
            } else {
                let num_round_points_per_thread = num_round_points / num_threads;
                let leftovers = if i == num_rounds - 1 {
                    num_round_points - (num_round_points_per_thread * num_rounds)
                } else {
                    0
                };

                let thread_point_schedule: Vec<u64> = state
                    .point_schedule
                    .clone()
                    .into_iter()
                    .skip(i * (num_round_points_per_thread as usize) + j * num_points)
                    .take(num_round_points_per_thread as usize)
                    .collect::<Vec<u64>>();
                let first_bucket = thread_point_schedule[0] & 0x7FFFFFFFu64;
                let last_bucket = thread_point_schedule
                    [(num_round_points_per_thread as usize) - 1 + (leftovers as usize)]
                    & 0x7FFFFFFFu64;
                let num_thread_buckets = (last_bucket - first_bucket) as usize + 1;

                let mut affine_product_state =
                    state.get_affine_pippenger_runtime_state(num_threads, j);
                affine_product_state.point_schedule = thread_point_schedule;
                affine_product_state.points = points.clone();
                affine_product_state.num_points = num_round_points_per_thread + leftovers;
                affine_product_state.num_buckets = num_thread_buckets;

                // reduce the wnaf entries into added buckets
                let output_buckets =
                    reduce_buckets(&mut affine_product_state, true, handle_edge_cases);

                let mut running_sum = Affine::default();

                // add the buckets together
                // one nice side-effect of the affine trick, is that half of the bucket concatenation
                // algorithm can use mixed addition formulae, instead of full addition formulae
                let mut output_it = affine_product_state.num_points - 1;
                for k in (0..num_thread_buckets as usize).rev() {
                    if !affine_product_state.bucket_empty_status[k] {
                        running_sum = (running_sum + output_buckets[output_it]).into();
                        output_it = output_it - 1;
                    }
                    accumulator = (accumulator + running_sum).into();
                }

                // we now need to scale up 'running sum' up to the value of the first bucket.
                // e.g. if first bucket is 0, no scaling
                // if first bucket is 1, we need to add (2 * running_sum)
                if first_bucket > 0 {
                    let multiplier = first_bucket << 1;
                    let mut shift = multiplier.get_msb() as i64;
                    let mut init = false;
                    let mut rolling_accumulator = Affine::default();
                    while shift >= 0 {
                        if init {
                            rolling_accumulator =
                                (rolling_accumulator + rolling_accumulator).into();
                            if (multiplier >> shift) & 1 == 1 {
                                rolling_accumulator = (rolling_accumulator + running_sum).into();
                            }
                        } else {
                            rolling_accumulator = (rolling_accumulator + running_sum).into();
                        }
                        init = true;
                        shift = shift - 1;
                    }
                }
            }

            // Divide the points and subtract skewed points once
            if i == num_rounds - 1 {
                let num_points_per_thread = (state.num_points as usize) / num_threads;
                for k in 0..=num_points_per_thread {
                    if state.skew_table[j * num_points_per_thread + k] {
                        accumulator =
                            (accumulator + (-points[j * num_points_per_thread + k])).into();
                    }
                }
            }

            // scale thread accumulator after each round
            if i > 0 {
                for _ in 0..=bits_per_bucket {
                    thread_accumulators[j] =
                        (thread_accumulators[j] + thread_accumulators[j]).into();
                }
            }

            thread_accumulators[j] = (thread_accumulators[j] + accumulator).into();
        }
    }

    let mut result = Affine::default();
    for (_, accumulator) in thread_accumulators.iter().enumerate() {
        result = (result + accumulator).into();
    }

    result
}

fn pippenger_internal<C: SWCurveConfig + GLVConfig>(
    scalars: &mut [C::ScalarField],
    points: &mut [Affine<C>],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<C>,
    handle_edge_cases: bool,
) -> Affine<C> {
    compute_wnaf_states::<C>(
        state.point_schedule.as_mut(),
        state.skew_table.as_mut(),
        state.round_counts.as_mut(),
        scalars,
        num_initial_points,
    );
    organise_buckets(state.point_schedule.as_mut(), num_initial_points * 2);
    evaluate_pippenger_rounds::<C>(
        state,
        points.into(),
        num_initial_points * 2,
        handle_edge_cases,
    )
}

///  It's pippenger! But this one has go-faster stripes and a prediliction for questionable life choices.
///  We use affine-addition formula in this method, which paradoxically is ~45% faster than the mixed addition
/// formulae. See `scalar_multiplication.cpp` for a more detailed description.
///
///  It's...unsafe, because we assume that the incomplete addition formula exceptions are not triggered i.e. that all the
///  points provided as arguments to the msm are distinct.
///  We don't bother to check for this to avoid conditional branches in a critical section of our code.
///  This is fine for situations where your bases are linearly independent (i.e. KZG10 polynomial commitments where
///  there should be no equal points in the SRS), because triggering the incomplete addition exceptions is about as hard
/// as solving the disrete log problem. This is ok for the prover, but GIANT RED CLAXON WARNINGS FOR THE VERIFIER Don't
/// use this in a verification algorithm! That would be a really bad idea. Unless you're a malicious adversary, then it
/// would be a great idea!
pub(crate) fn pippenger_unsafe<C: SWCurveConfig + GLVConfig>(
    scalars: &mut [C::ScalarField],
    points: &mut [Affine<C>],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<C>,
) -> Affine<C> {
    pippenger::<C>(scalars, points, num_initial_points, state, false)
}

pub(crate) fn pippenger<C: SWCurveConfig + GLVConfig>(
    scalars: &mut [C::ScalarField],
    points: &mut [Affine<C>],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<C>,
    handle_edge_cases: bool,
) -> Affine<C> {
    // our windowed non-adjacent form algorthm requires that each thread can work on at least 8 points.
    // If we fall below this theshold, fall back to the traditional scalar multiplication algorithm.
    // For 8 threads, this neatly coincides with the threshold where Strauss scalar multiplication outperforms
    // Pippenger
    let threshold = compute_num_threads() * 8;

    if num_initial_points == 0 {
        return Affine::identity();
    }

    if num_initial_points <= threshold {
        let mut exponentiation_results = vec![Affine::zero(); num_initial_points];

        for i in 0..num_initial_points {
            exponentiation_results[i] = (points[i * 2] * scalars[i]).into();
        }

        for i in (1..num_initial_points).rev() {
            exponentiation_results[i - 1] =
                (exponentiation_results[i - 1] + exponentiation_results[i]).into();
        }
        return exponentiation_results[0];
    }

    let slice_bits = num_initial_points.get_msb();
    let num_slice_points = 1 << slice_bits;

    let result = pippenger_internal(scalars, points, num_slice_points, state, handle_edge_cases);

    if num_slice_points != num_initial_points {
        let leftovers_points = num_initial_points - num_slice_points;
        return (result
            + pippenger(
                scalars[num_slice_points..].as_mut(),
                points[num_slice_points * 2..].as_mut(),
                leftovers_points,
                state,
                handle_edge_cases,
            ))
        .into();
    } else {
        return result;
    }
}

pub(crate) fn pippenger_without_endomorphism_basis_points<C: SWCurveConfig + GLVConfig>(
    scalars: &mut [C::ScalarField],
    points: &mut [Affine<C>],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<C>,
) -> Affine<C> {
    let mut g_mod: Vec<Affine<C>> = vec![Affine::zero(); num_initial_points * 2];
    generate_pippenger_point_table::<C>(points, g_mod.as_mut_slice(), num_initial_points);
    pippenger::<C>(
        scalars,
        g_mod.as_mut_slice(),
        num_initial_points,
        state,
        false,
    )
}
