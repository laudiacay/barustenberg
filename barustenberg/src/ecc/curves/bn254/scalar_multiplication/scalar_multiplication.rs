use crate::ecc::curves::bn254::{fq::Fq, fr::Fr, g1::G1};

use super::runtime_states::{
    get_optimal_bucket_width, AffineProductRuntimeState, PippengerRuntimeState,
};

pub fn get_num_buckets(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    return 1 << bits_per_bucket;
}

/**
 * pointers that describe how to add points into buckets, for the pippenger algorithm.
 * `wnaf_table` is an unrolled two-dimensional array, with each inner array being of size `n`,
 * where `n` is the number of points being multiplied. The second dimension size is defined by
 * the number of pippenger rounds (fixed for a given `n`, see `get_num_rounds`)
 *
 * An entry of `wnaf_table` contains the following three pieces of information:
 * 1: the point index that we're working on. This is stored in the high 32 bits
 * 2: the bucket index that we're adding the point into. This is stored in the low 31 bits
 * 3: the sign of the point we're adding (i.e. do we actually need to subtract). This is stored in the 32nd bit.
 *
 * We pack this information into a 64 bit unsigned integer, so that we can more efficiently sort our wnaf entries.
 * For a given round, we want to sort our wnaf entries in increasing bucket index order.
 *
 * This is so that we can efficiently use multiple threads to execute the pippenger algorithm.
 * For a given round, a given point's bucket index will be uniformly randomly distributed,
 * assuming the inputs are from a zero-knowledge proof. This is because the scalar multiplier will be uniformly randomly
 *distributed, and the bucket indices are derived from the scalar multiplier.
 *
 * This means that, if we were to iterate over all of our points in order, and add each point into its associated
 *bucket, we would be accessing all of our buckets in a completely random pattern.
 *
 * Aside from memory latency problems this incurs, this makes the naive algorithm unsuitable for multithreading - we
 *cannot assign a thread a tranche of points, because each thread will be adding points into the same set of buckets,
 *triggering race conditions. We do not want to manage the overhead of thread locks for each bucket; the process of
 *adding a point into a bucket takes, on average, only 400 CPU cycles, so the slowdown of managing mutex locks would add
 *considerable overhead.
 *
 * The solution is to sort the buckets. If the buckets are sorted, we can assign a tranche of buckets to individual
 *threads, safe in the knowledge that there will be no race conditions, with one condition. A thread's starting bucket
 *may be equal to the previous thread's end bucket, so we need to ensure that each thread works on a local array of
 *buckets. This adds little overhead (for 2^20 points, we have 32,768 buckets. With 8 threads, the amount of bucket
 *overlap is ~16 buckets, so we could incur 16 extra 'additions' in pippenger's bucket concatenation phase, but this is
 *an insignificant contribution).
 *
 * The alternative approach (the one we used to use) is to slice up all of the points being multiplied amongst all
 *available threads, and run the complete pippenger algorithm for each thread. This is suboptimal, because the
 *complexity of pippenger is O(n / logn) point additions, and a sequence of smaller pippenger calls will have a smaller
 *`n`.
 *
 * This is the motivation for multi-threading the actual Pippenger algorithm. In addition, the above approach performs
 *extremely poorly for GPUs, where the number of threads can be as high as 2^10 (for a multi-scalar-multiplication of
 *2^20 points, this doubles the number of pippenger rounds per thread)
 *
 * To give concrete numbers, the difference between calling pippenger on 2^20 points, and calling pippenger 8 times on
 *2^17 points, is 5-10%. Which means that, for 8 threads, we need to ensure that our sorting algorithm adds less than 5%
 *to the total runtime of pippenger. Given a single cache miss per point would increase the run-time by 25%, this is not
 *much room to work with!
 *
 * However, a radix sort, combined with the fact that the total number of buckets is quite small (2^16 at most), seems
 *to be fast enough. Benchmarks indicate (i7-8650U, 8 threads) that, for 2^20 points, the total runtime is <1200ms and
 *of that, the radix sort consumes 58ms (4.8%)
 *
 * One advantage of sorting by bucket order vs point order, is that a 'bucket' is 96 bytes large (sizeof(g1::element),
 *buckets have z-coordinates). Points, on the other hand, are 64 bytes large (affine points, no z-coordinate). This
 *makes fetching random point locations in memory more efficient than fetching random bucket locations, as each point
 *occupies a single cache line. Using __builtin_prefetch to recover the point just before it's needed, seems to improve
 *the runtime of pippenger by 10-20%.
 *
 * Finally, `skew_table` tracks whether a scalar multplier is even or odd
 * (if it's even, we need to subtract the point from the total result,
 * because our windowed non-adjacent form values can only be odd)
 *
 **/

struct MultiplicationThreadState {
    buckets: Vec<G1>,
    point_schedule: Vec<u64>,
}

fn compute_wnaf_states(
    point_schedule: &mut [u64],
    input_skew_table: &[bool],
    round_counts: &mut [u64],
    scalars: &[Fr],
    num_initial_points: usize,
) {
    todo!("compute_wnaf_states")
}

pub fn generate_pippenger_point_table(
    points: &[G1::Affine],
    table: &mut [G1::Affine],
    num_points: usize,
) {
    todo!("generate_pippenger_point_table")
}

fn organize_buckets(point_schedule: &mut [u64], round_counts: &[u64], num_points: usize) {
    todo!("organize_buckets")
}

fn count_bits(bucket_counts: &[u32], bit_offsets: &mut [u32], num_buckets: usize, num_bits: usize) {
    todo!("count_bits")
    /*
        for (size_t i = 0; i < num_buckets; ++i) {
        const uint32_t count = bucket_counts[i];
        for (uint32_t j = 0; j < num_bits; ++j) {
            bit_offsets[j + 1] += (count & (1U << j));
        }
    }
    bit_offsets[0] = 0;
    for (size_t i = 2; i < num_bits + 1; ++i) {
        bit_offsets[i] += bit_offsets[i - 1];
    }
     */
}

fn add_affine_points(points: &mut [G1::Affine], num_points: usize, scratch_space: &mut [Fq]) {
    todo!("add_affine_points")
}

fn add_affine_points_with_edge_cases(
    points: &mut [G1],
    num_points: usize,
    scratch_space: &mut [Fq],
) {
    todo!("add_affine_points_with_edge_cases")
}

impl AffineProductRuntimeState {
    fn reduce_buckets(
        &mut self,
        first_round: Option<bool>,
        handle_edge_cases: Option<bool>,
    ) -> Vec<G1::Affine> {
        let first_round = first_round.unwrap_or(true);
        let handle_edge_cases = handle_edge_cases.unwrap_or(false);
        todo!("reduce_buckets")
    }

    fn evaluate_addition_chains(&mut self, max_bucket_bits: usize, handle_edge_cases: bool) {
        todo!("evaluate_addition_chains")
    }

    fn construct_addition_chains(&mut self, empty_bucket_counts: Option<bool>) -> u32 {
        let empty_bucket_counts = empty_bucket_counts.unwrap_or(true);
        todo!("construct_addition_chains")
    }
}

impl PippengerRuntimeState {
    fn pippenger_internal(
        &mut self,
        points: &mut [G1::Affine],
        scalars: &[Fr],
        num_initial_points: usize,
        handle_edge_cases: bool,
    ) -> G1 {
        todo!("pippenger_internal")
    }

    fn evaluate_pippenger_rounds(
        &mut self,
        points: &mut [G1::Affine],
        num_points: usize,
        handle_edge_cases: Option<bool>,
    ) -> G1 {
        let handle_edge_cases = handle_edge_cases.unwrap_or(false);
        todo!("evaluate_pippenger_rounds")
    }
    pub fn pippenger(
        &mut self,
        scalars: &[Fr],
        points: &mut [G1::Affine],
        num_points: usize,
        handle_edge_cases: Option<bool>,
    ) -> G1 {
        let handle_edge_cases = handle_edge_cases.unwrap_or(true);
        todo!("pippenger")
    }

    fn pippenger_unsafe(
        &mut self,
        scalars: &[Fr],
        points: &mut [G1::Affine],
        num_initial_points: usize,
    ) -> G1 {
        todo!("pippenger_unsafe")
    }

    fn pippenger_without_endomorphism_basis_points(
        &mut self,
        scalars: &[Fr],
        points: &mut [G1::Affine],
        num_initial_points: usize,
    ) -> G1 {
        todo!("pippenger_without_endomorphism_basis_points")
    }
}
