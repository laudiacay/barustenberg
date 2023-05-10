use crate::ecc::{
    curves::bn254::{fq::Fq, g1::G1},
    groups::wnaf::WNAF_SIZE,
};

/// simple helper functions to retrieve pointers to pre-allocated memory for the scalar multiplication algorithm.
/// This is to eliminate page faults when allocating (and writing) to large tranches of memory.
pub fn get_optimal_bucket_width(num_points: usize) -> usize {
    if num_points >= 14_617_149 {
        return 21;
    }
    if num_points >= 1_139_094 {
        return 18;
    }
    if num_points >= 155_975 {
        return 15;
    }
    if num_points >= 144_834 {
        return 14;
    }
    if num_points >= 25_067 {
        return 12;
    }
    if num_points >= 13_926 {
        return 11;
    }
    if num_points >= 7_659 {
        return 10;
    }
    if num_points >= 2_436 {
        return 9;
    }
    if num_points >= 376 {
        return 7;
    }
    if num_points >= 231 {
        return 6;
    }
    if num_points >= 97 {
        return 5;
    }
    if num_points >= 35 {
        return 4;
    }
    if num_points >= 10 {
        return 3;
    }
    if num_points >= 2 {
        return 2;
    }
    1
}

fn get_num_rounds(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    WNAF_SIZE(bits_per_bucket + 1)
}

#[derive(Copy, Eq)]
pub struct AffineProductRuntimeState {
    points: Vec<G1::Affine>,
    point_pairs_1: Vec<G1::Affine>,
    point_pairs_2: Vec<G1::Affine>,
    scratch_space: Vec<Fq>,
    bucket_counts: Vec<u32>,
    bit_offsets: Vec<u32>,
    point_schedule: Vec<u64>,
    num_points: u32,
    num_buckets: u32,
    bucket_empty_status: Vec<bool>,
}

#[derive(Copy, Eq)]
pub struct PippengerRuntimeState {
    point_schedule: Vec<u64>,
    skew_table: Vec<bool>,
    point_pairs_1: Vec<G1::Affine>,
    point_pairs_2: Vec<G1::Affine>,
    scratch_space: Vec<Fq>,
    bucket_counts: Vec<u32>,
    bit_counts: Vec<u32>,
    bucket_empty_status: Vec<bool>,
    round_counts: Vec<u64>,
    num_points: u64,
}
// TODO what is aligned_free! oh jeez. you need to think on that one.
impl PippengerRuntimeState {
    fn get_affine_product_runtime_state(
        &self,
        num_threads: usize,
        thread_index: usize,
    ) -> AffineProductRuntimeState {
        todo!("get_affine_product_runtime_state")
    }

    pub fn new(num_initial_points: u64) -> Self {
        /*
                           constexpr size_t MAX_NUM_ROUNDS = 256;
            num_points = num_initial_points * 2;
            const size_t num_points_floor = static_cast<size_t>(1ULL << (numeric::get_msb(num_points)));
            const size_t num_buckets = static_cast<size_t>(
                1U << barretenberg::scalar_multiplication::get_optimal_bucket_width(static_cast<size_t>(num_initial_points)));
        #ifndef NO_MULTITHREADING
            const size_t num_threads = max_threads::compute_num_threads();
        #else
            const size_t num_threads = 1;
        #endif
            const size_t prefetch_overflow = 16 * num_threads;
            const size_t num_rounds =
                static_cast<size_t>(barretenberg::scalar_multiplication::get_num_rounds(static_cast<size_t>(num_points_floor)));
            point_schedule = (uint64_t*)(aligned_alloc(
                64, (static_cast<size_t>(num_points) * num_rounds + prefetch_overflow) * sizeof(uint64_t)));
            skew_table = (bool*)(aligned_alloc(64, pad(static_cast<size_t>(num_points) * sizeof(bool), 64)));
            point_pairs_1 = (g1::affine_element*)(aligned_alloc(
                64, (static_cast<size_t>(num_points) * 2 + (num_threads * 16)) * sizeof(g1::affine_element)));
            point_pairs_2 = (g1::affine_element*)(aligned_alloc(
                64, (static_cast<size_t>(num_points) * 2 + (num_threads * 16)) * sizeof(g1::affine_element)));
            scratch_space = (fq*)(aligned_alloc(64, static_cast<size_t>(num_points) * sizeof(g1::affine_element)));
            bucket_counts = (uint32_t*)(aligned_alloc(64, num_threads * num_buckets * sizeof(uint32_t)));
            bit_counts = (uint32_t*)(aligned_alloc(64, num_threads * num_buckets * sizeof(uint32_t)));
            bucket_empty_status = (bool*)(aligned_alloc(64, num_threads * num_buckets * sizeof(bool)));
            round_counts = (uint64_t*)(aligned_alloc(32, MAX_NUM_ROUNDS * sizeof(uint64_t)));

            const size_t points_per_thread = static_cast<size_t>(num_points) / num_threads;
        #ifndef NO_MULTITHREADING
        #pragma omp parallel for
        #endif
            for (size_t i = 0; i < num_threads; ++i) {
                const size_t thread_offset = i * points_per_thread;
                memset((void*)(point_pairs_1 + thread_offset + (i * 16)),
                       0,
                       (points_per_thread + 16) * sizeof(g1::affine_element));
                memset((void*)(point_pairs_2 + thread_offset + (i * 16)),
                       0,
                       (points_per_thread + 16) * sizeof(g1::affine_element));
                memset((void*)(scratch_space + thread_offset), 0, (points_per_thread) * sizeof(fq));
                for (size_t j = 0; j < num_rounds; ++j) {
                    const size_t round_offset = (j * static_cast<size_t>(num_points));
                    memset((void*)(point_schedule + round_offset + thread_offset), 0, points_per_thread * sizeof(uint64_t));
                }
                memset((void*)(skew_table + thread_offset), 0, points_per_thread * sizeof(bool));
            }

            memset((void*)bucket_counts, 0, num_threads * num_buckets * sizeof(uint32_t));
            memset((void*)bit_counts, 0, num_threads * num_buckets * sizeof(uint32_t));
            memset((void*)bucket_empty_status, 0, num_threads * num_buckets * sizeof(bool));
            memset((void*)round_counts, 0, MAX_NUM_ROUNDS * sizeof(uint64_t));
        }
                         */
        todo!("see comment")
    }
}
