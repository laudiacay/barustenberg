// TODO todo - stubs to get the compiler to cooperate.

pub trait FieldElement {
    type SizeInBytes: typenum::Unsigned; // do a typenum here
}

pub trait Field {
    type Element: FieldElement;
}

trait GroupElement {
    type SizeInBytes: typenum::Unsigned; // do a typenum here
}

pub trait Group {
    type Element: GroupElement;
}

pub trait Pairing<G1: Group, G2: Group> {
    type Output: Group;
}

pub struct Pippenger {}

pub mod curves {
    pub mod bn254 {
        use ark_bn254::{Fq, G1Affine};
        #[derive(Copy, Eq)]
        pub struct PippengerRuntimeState {
            point_schedule: Vec<u64>,
            skew_table: Vec<bool>,
            point_pairs_1: Vec<G1Affine>,
            point_pairs_2: Vec<G1Affine>,
            scratch_space: Vec<Fq>,
            bucket_counts: Vec<u32>,
            bit_counts: Vec<u32>,
            bucket_empty_status: Vec<bool>,
            round_counts: Vec<u64>,
            num_points: u64,
        }
        // TODO what is aligned_free! oh jeez. you need to think on that one.
        impl PippengerRuntimeState {
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
    }

    pub mod grumpkin {
        pub struct Fr;
        impl super::super::FieldElement for Fr {
            // TODO compilation placeholder come back here bb
            type SizeInBytes = typenum::U0;
        }
        impl super::super::Field for Fr {
            // TODO compilation placeholder come back here bb
            type Element = Fr;
        }
    }
}
