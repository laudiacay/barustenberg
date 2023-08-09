use std::num;

use ark_ec::{
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr,
};
use ark_ff::Field;
use get_msb::Msb;
use ark_std::{ops::{Add, Sub, SubAssign, Mul, MulAssign}};

use crate::{
    common::max_threads::compute_num_threads,
    ecc::scalar_multiplication::process_buckets::process_buckets, numeric::bitop::get_msb,
};

use super::{
    cube_root_of_unity,
    runtime_states::{
        get_num_rounds, get_optimal_bucket_width, AffinePippengerRuntimeState,
        PippengerRuntimeState,
    },
};

pub(crate) fn generate_pippenger_point_table<F: Field, G: AffineRepr<BaseField = F>>(
    points: &mut [G],
    table: &mut [G],
    num_points: usize,
) {
    // calculate the cube root of unity
    let beta = cube_root_of_unity::<F>();

    // iterate backwards, so that `points` and `table` can point to the same memory location
    for i in (0..num_points).rev() {
        table[i * 2] = points[i];
        let (table_x, table_y) = table[i * 2 + 1].xy().unwrap();
        let (x, y) = points[i].xy().unwrap();
        *table_x = beta * x;
        *table_y = y.neg();
    }
}

fn compute_wnaf_states<F: Field>(
    point_schedule: &mut Vec<u64>,
    skew_table: &mut Vec<bool>,
    round_counts: &mut Vec<u64>,
    scalars: &[F],
    num_points: usize,
) {
    todo!("implement");
}

fn organise_buckets(point_schedule: &mut Vec<u64>, num_points: usize) {
    let num_rounds: usize = get_num_rounds(num_points);

    for i in 0..num_rounds {
        // TODO: might be weird shit here for passing slice
        process_buckets(
            point_schedule[i * num_points..i + 1 * num_points].as_mut(),
            num_points,
            (get_optimal_bucket_width(num_points / 2) + 1) as u32,
        )
    }
}

// We implement our own endomorphism split in lue of using arkworks for two reasons
// 1.) the output of arkworks implementation returns a (sign, field) instead of just the field element
// 2.) glv decomposition exists as a trait we would need to implement for each curve and requires the same amount of thought and code as just doing this
// In the future i do think these implementations should live as a trait. But not a trait defined by arkworks #fuck_arkworks
fn split_into_endomorphism_scalars<F: Field>(scalar: F) -> Vec<F> {
    todo!();
}

fn split_into_endomorphism_scalars_384<F: Field>(scalar: F) -> Vec<F> {
    todo!();
}

fn construct_addition_chains<F: Field, G: AffineRepr<BaseField = F>>(
    state: &mut AffinePippengerRuntimeState<F, G>,
    first_round: bool,
) -> usize {
    todo!()
}

fn add_affine_point_with_edge_cases<F: Field, G: AffineRepr<BaseField = F>>(
    points: &mut [G],
    num_points: usize,
    scratch_space: &mut [F],
) {
    //Fq
    let batch_inversion_accumulator = F::one();

    for i in (0..num_points).step_by(2) {
        if points[i].is_zero() || points[i + 1].is_zero() {
            continue;
        }
        let (x1, y1) = points[i].xy().expect("Failed to grab points from x1 in first part of affine addition");
        let (x2, y2) = points[i + 1].xy().expect("Failed to grab points from x2 in first part of affine addition");
        if x1 == x2 {
            if y1 == y2 {
                scratch_space[i >> 1] = x1.double(); // 2x
                let x_squared = x1.square();
                *x2 = y1.double();  // 2y
                //TODO: could be better way to do this in arkworks
                *y2 = x_squared + x_squared + x_squared; // 3x^2
                *y2 *= batch_inversion_accumulator;
                batch_inversion_accumulator *= x2;
            }
            //set to infinity
            points[i] = G::zero();
            points[i+1] = G::zero();
        }
        scratch_space[i >> 1] = *x1 + x2;   // x2 + x1
        *x2 -= x1;                          // x2 - x1
        *y2 -= y2;                          // y2 - y1
        *y2 *= batch_inversion_accumulator;  // (y2 - y1) * accumulator_old
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

        let (x1, y1) = points[i].xy().expect("Failed to grab points from x1 in first part of affine addition");
        let (x2, y2) = points[i + 1].xy().expect("Failed to grab points from x2 in first part of affine addition");
        let (x3, y3) = points[(i + num_points) >> 1].xy().expect("Failed to grab points from x2 in first part of affine addition");

        *y2 *= batch_inversion_accumulator; //update accumulator
        batch_inversion_accumulator * x2;
        *x2 = y2.square();
        *x3 = *x2 - scratch_space[i >> 1]; // x3 = lambda_squared - x2 - x1?
        
        *x1 -= x3;
        *x1 *= y2;
        *x3 = *x1 - y1;
    }
}

/**
 * adds a bunch of points together using affine addition formulae.
 * Paradoxically, the affine formula is crazy efficient if you have a lot of independent point additions to perform.
 * Affine formula:
 *
 * \lambda = (y_2 - y_1) / (x_2 - x_1)
 * x_3 = \lambda^2 - (x_2 + x_1)
 * y_3 = \lambda*(x_1 - x_3) - y_1
 *
 * Traditionally, we avoid affine formulae like the plague, because computing lambda requires a modular inverse,
 * which is outrageously expensive.
 *
 * However! We can use Montgomery's batch inversion technique to amortise the cost of the inversion to ~0.
 *
 * The way batch inversion works is as follows. Let's say you want to compute \{ 1/x_1, 1/x_2, ..., 1/x_n \}
 * The trick is to compute the product x_1x_2...x_n , whilst storing all of the temporary products.
 * i.e. we have an array A = [x_1, x_1x_2, ..., x_1x_2...x_n]
 * We then compute a single inverse: I = 1 / x_1x_2...x_n
 * Finally, we can use our accumulated products, to quotient out individual inverses.
 * We can get an individual inverse at index i, by computing I.A_{i-1}.(x_nx_n-1...x_i+1)
 * The last product term we can compute on-the-fly, as it grows by one element for each additional inverse that we
 * require.
 *
 * TLDR: amortized cost of a modular inverse is 3 field multiplications per inverse.
 * Which means we can compute a point addition with SIX field multiplications in total.
 * The traditional Jacobian-coordinate formula requires 11.
 *
 * There is a catch though - we need large sequences of independent point additions!
 * i.e. the output from one point addition in the sequence is NOT an input to any other point addition in the
 *sequence.
 *
 * We can re-arrange the Pippenger algorithm to get this property, but it's...complicated
 **/
fn add_affine_points<F: Field, G: AffineRepr<BaseField = F>>(
    points: &mut [G],
    num_points: usize,
    scratch_space: &mut [F],
) {
    //Fq
    let batch_inversion_accumulator = F::one();

    for i in (0..num_points).step_by(2) {
        let (x1, y1) = points[i].xy().expect("Failed to grab points from x1 in first part of affine addition");
        let (x2, y2) = points[i + 1].xy().expect("Failed to grab points from x2 in first part of affine addition");
        scratch_space[i >> 1] = *x1 + x2;   // x2 + x1
        *x2 -= x1;                          // x2 - x1
        *y2 -= y2;                          // y2 - y1
        *y2 *= batch_inversion_accumulator;  // (y2 - y1) * accumulator_old
        batch_inversion_accumulator *= x2;
    }

    if batch_inversion_accumulator.is_zero() {
        panic!("attempted to invert zero in add_affine_points");
    } else {
        batch_inversion_accumulator = batch_inversion_accumulator.inverse().unwrap();
    }

    for i in ((num_points - 2)..0).step_by(2) {
        // TODO: add builtin prefetch
        let (x1, y1) = points[i].xy().expect("Failed to grab points from x1 in first part of affine addition");
        let (x2, y2) = points[i + 1].xy().expect("Failed to grab points from x2 in first part of affine addition");
        let (x3, y3) = points[(i + num_points) >> 1].xy().expect("Failed to grab points from x2 in first part of affine addition");

        *y2 *= batch_inversion_accumulator; //update accumulator
        batch_inversion_accumulator * x2;
        *x2 = y2.square();
        *x3 = *x2 - scratch_space[i >> 1]; // x3 = lambda_squared - x2 - x1?
        
        *x1 -= x3;
        *x1 *= y2;
        *x3 = *x1 - y1;
    }
}

fn evaluate_addition_chains<F: Field, G: AffineRepr<BaseField = F>>(
    state: &mut AffinePippengerRuntimeState<F, G>,
    max_bucket_bits: usize,
    handle_edge_cases: bool,
) {
    let end = state.num_points;
    let start = 0;
    for i in 0..max_bucket_bits {
        let points_in_round = (state.num_points - state.bit_offsets[i + 1] as usize) >> i;
        let start = end - points_in_round;
        let round_points = &mut state.point_pairs_1[start..start + points_in_round];
        if handle_edge_cases {
            add_affine_point_with_edge_cases(round_points, points_in_round, &mut state.scratch_space);
        } else {
            add_affine_points(round_points, points_in_round, &mut state.scratch_space);
        }
    }
}

fn reduce_buckets<F: Field, G: AffineRepr<BaseField = F>>(
    state: &mut AffinePippengerRuntimeState<F, G>,
    first_round: bool,
    handle_edge_cases: bool,
) -> Vec<G> {
    let max_bucket_bits = construct_addition_chains(state, first_round);

    if max_bucket_bits == 0 {
        return state.point_pairs_1;
    }

    evaluate_addition_chains(state, max_bucket_bits, handle_edge_cases);

    let end: u32 = state.num_points as u32;
    let start: u32 = 0;

    for i in 0..max_bucket_bits {
        let points_in_round = (state.num_points as u32 - state.bit_offsets[i + 1]) >> i;
        let points_removed = points_in_round / 2;
        let start = end - points_in_round;
        let modified_start = start + points_removed;
        state.bit_offsets[i + 1] = modified_start;
    }

    let new_num_points = 0;
    for i in 0..state.num_buckets {
        let count = state.bucket_counts[i];
        let num_bits = count.get_msb();
        let new_bucket_count = 0;
        for j in 0..num_bits {
            let current_offset = state.bit_offsets[i];
            let has_entry = ((count >> j) & 1) == 1;
            if has_entry {
                let schedule: u64 = ((current_offset as u64) << 32) + (i as u64);
                state.point_schedule[new_num_points] = schedule;
                new_num_points += 1;
                new_bucket_count += 1;
                current_offset += 1;
            }
        }
        state.bucket_counts[i] = new_bucket_count;
    }

    state.num_points = new_num_points;
    let temp: Vec<G> = state.point_pairs_1;
    state.points = state.point_pairs_1;
    state.point_pairs_1 = state.point_pairs_2;
    state.point_pairs_2 = temp;

    return reduce_buckets(state, false, handle_edge_cases);
}

fn evaluate_pippenger_rounds<F: Field, G: AffineRepr<BaseField = F>>(
    state: &PippengerRuntimeState<F, G>,
    points: Vec<G>,
    num_points: usize,
    handle_edge_cases: bool,
) -> G {
    let num_threads = compute_num_threads();
    let num_rounds = get_num_rounds(num_points);
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);

    let mut thread_accumulators = vec![G::default(); num_threads];

    for j in 0..num_threads {
        for i in 0..num_rounds {
            let num_round_points = state.round_counts[i] as usize;
            let accumulator = G::default();

            if num_round_points == 0 || (num_round_points < num_threads && j < num_threads - 1) {
                // skip if round points not enough for thread parallelism
            } else {
                let num_round_points_per_thread = num_round_points / num_threads;
                let leftovers = if i == num_rounds - 1 {
                    num_round_points - (num_round_points_per_thread * num_rounds)
                } else {
                    0
                };

                let mut thread_point_schedule = state
                    .point_schedule
                    .into_iter()
                    .skip(i * (num_round_points_per_thread as usize) + j * num_points)
                    .take(num_round_points_per_thread as usize)
                    .collect::<Vec<u64>>();
                let first_bucket = thread_point_schedule[0] & 0x7FFFFFFFu64;
                let last_bucket = thread_point_schedule
                    [(num_round_points_per_thread as usize) - 1 + (leftovers as usize)]
                    & 0x7FFFFFFFu64;
                let num_thread_buckets = (last_bucket - first_bucket) as usize + 1;

                let mut affine_product_state: AffinePippengerRuntimeState<F, G> =
                    state.get_affine_pippenger_runtime_state(num_threads, j);
                affine_product_state.point_schedule = thread_point_schedule;
                affine_product_state.points = points.clone();
                affine_product_state.num_points = num_round_points_per_thread + leftovers;
                affine_product_state.num_buckets = num_thread_buckets;

                // reduce the wnaf entries into added buckets
                let output_buckets =
                    reduce_buckets(&mut affine_product_state, true, handle_edge_cases);

                let running_sum = G::default();

                // add the buckets together
                let output_it = affine_product_state.num_points as usize - 1;
                for k in (0..num_thread_buckets as usize).rev() {
                    if !affine_product_state.bucket_empty_status[k] {
                        running_sum = (running_sum + output_buckets[output_it]).into();
                        output_it = output_it - 1;
                    }
                    // TODO: fix
                    accumulator = (accumulator + running_sum).into();
                }

                // we now need to scale up 'running sum' up to the value of the first bucket.
                // e.g. if first bucket is 0, no scaling
                // if first bucket is 1, we need to add (2 * running_sum)
                if first_bucket > 0 {
                    let multiplier = first_bucket << 1;
                    let mut shift = multiplier.get_msb();
                    let init = false;
                    let mut rolling_accumulator = G::default();
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
                let addition_temporary = G::zero();
                let num_points_per_thread = (state.num_points as usize) / num_threads;
                for k in 0..=bits_per_bucket {
                    if state.skew_table[j * num_points_per_thread + k] {
                        //accumulator.mul_assign(points[j * num_points_per_thread + k]);
                    }
                }
            }

            // scale thread accumulator after each round
            if i > 0 {
                for k in 0..=bits_per_bucket {
                    thread_accumulators[j] =
                        (thread_accumulators[j] + thread_accumulators[j]).into()
                }
            }

            thread_accumulators[j] = (thread_accumulators[j] + accumulator).into();
        }
    }

    //TODO for now to compile made this change
    let mut result = G::zero();
    for (_, accumulator) in thread_accumulators.iter().enumerate() {
        // TODO: this is 100% wrong
        result = (result + *accumulator).into();
    }

    result
}

fn pippenger_internal<F: Field, G: AffineRepr<BaseField = F>>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<F, G>,
    handle_edge_cases: bool,
) -> G {
    compute_wnaf_states(
        state.point_schedule.as_mut(),
        state.skew_table.as_mut(),
        state.round_counts.as_mut(),
        scalars,
        num_initial_points,
    );
    organise_buckets(state.point_schedule.as_mut(), num_initial_points * 2);
    evaluate_pippenger_rounds(
        state,
        points.into(),
        num_initial_points * 2,
        handle_edge_cases,
    )
}

pub(crate) fn pippenger_unsafe<F: Field, G: AffineRepr<BaseField = F>>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<F, G>,
) -> G::Group {
    pippenger::<F, G>(scalars, points, num_initial_points, state, false)
}

pub(crate) fn pippenger<F: Field, G: AffineRepr<BaseField = F>>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<F, G>,
    handle_edge_cases: bool,
) -> G::Group {
    let threshold = compute_num_threads();
    if num_initial_points == 0 {
        //Don't think this is supposed to be the identity but yolo until other functions are working
        let out = G::zero();
        // TODO: find what is equivalent to this in arkworks
        // out.self_set_infinity()
        return out.into();
    }

    if num_initial_points <= threshold {
        let mut exponentiation_results = vec![G::zero(); num_initial_points];
        /*
        for i in 0..num_initial_points {
            exponentiation_results[i] = points[i * 2] * scalars[i];
        }
        */
        for i in (1..num_initial_points).rev() {
            exponentiation_results[i - 1] + exponentiation_results[i];
        }
        return exponentiation_results[0].into();
    }

    let slice_bits = num_initial_points.get_msb();
    let num_slice_points = 1 << slice_bits;

    let result = pippenger_internal(scalars, points, num_slice_points, state, handle_edge_cases);

    if num_slice_points != num_initial_points {
        let leftovers = num_initial_points - num_slice_points;
        // TODO: correct this
        return result
            + pippenger(
                scalars,
                points,
                num_initial_points,
                state,
                handle_edge_cases,
            );
    } else {
        return result.into();
    }
}

pub(crate) fn pippenger_without_endomorphism_basis_points<F: Field, G: AffineRepr<BaseField = F>>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &mut PippengerRuntimeState<F, G>,
) -> G::Group {
    let mut g_mod: Vec<G> = vec![G::zero(); num_initial_points * 2];
    generate_pippenger_point_table::<F, G>(points.into(), g_mod.as_mut_slice(), num_initial_points);
    pippenger::<F, G>(
        scalars,
        g_mod.as_mut_slice(),
        num_initial_points,
        state,
        false,
    )
}
