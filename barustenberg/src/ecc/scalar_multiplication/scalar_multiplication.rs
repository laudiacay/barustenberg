use core::num;
use std::{f32::consts::PI, ops::AddAssign};

use ark_ec::{
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr,
};
use ark_ff::{FftField, Field};
use get_msb::Msb;

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

pub(crate) fn generate_pippenger_point_table<C: SWCurveConfig, G: AffineRepr>(
    points: &mut [G],
    table: &mut [G],
    num_points: usize,
) {
    // calculate the cube root of unity
    todo!("implement")
    // let beta = cube_root_of_unity::<C::BaseField>();

    // // iterate backwards, so that `points` and `table` can point to the same memory location
    // for i in (0..num_points).rev() {
    //     table[i * 2] = points[i];
    //     table[i * 2 + 1].x = beta * points[i].x;
    //     table[i * 2 + 1].y = -points[i].y;
    // }
}

pub(crate) fn pippenger<F: Field + FftField, G: AffineRepr, C: SWCurveConfig>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &PippengerRuntimeState<F, G>,
    handle_edge_cases: bool,
) -> G {
    todo!("implement");
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

fn reduce_buckets<F: Field, G: AffineRepr>(
    mut state: AffinePippengerRuntimeState<F, G>,
    first_round: bool,
    handle_edge_cases: bool,
) -> Vec<G> {
    todo!("implement");
}

fn evaluate_pippenger_rounds<F: Field, G: AffineRepr>(
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
                let output_buckets = reduce_buckets(affine_product_state, true, handle_edge_cases);

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
                let addition_temporary = G::default();
                let num_points_per_thread = (state.num_points as usize) / num_threads;
                for k in 0..=bits_per_bucket {
                    if state.skew_table[j * num_points_per_thread + k] {
                        accumulator = (accumulator - points[j * num_points_per_thread + k]).into();
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

    let mut result = G::default();
    for (_, accumulator) in thread_accumulators.iter().enumerate() {
        // TODO: this is 100% wrong
        result = (result + *accumulator).into();
    }

    result
}

fn pippenger_internal<F: Field, G: AffineRepr, C: SWCurveConfig>(
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
    evaluate_pippenger_rounds(state, points, num_initial_points * 2, handle_edge_cases)
}

pub(crate) fn pippenger_unsafe<F: Field + FftField, G: AffineRepr, C: SWCurveConfig>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &PippengerRuntimeState<F, G>,
) -> G {
    pippenger::<F, G, C>(scalars, points, num_initial_points, state, false)
}

pub(crate) fn pippenger_without_endomorphism_basis_points<
    F: Field + FftField,
    G: AffineRepr,
    C: SWCurveConfig,
>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &PippengerRuntimeState<F, G>,
) -> G {
    let mut g_mod: Vec<G> = vec![G::default(); num_initial_points * 2];
    generate_pippenger_point_table::<C, G>(points, g_mod.as_mut_slice(), num_initial_points);
    pippenger::<F, G, C>(
        scalars,
        g_mod.as_mut_slice(),
        num_initial_points,
        state,
        false,
    )
}
