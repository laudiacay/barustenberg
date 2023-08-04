use crate::{
    common::max_threads::compute_num_threads,
    ecc::scalar_multiplication::process_buckets::{get_num_rounds, get_optimal_bucket_width},
    numeric::bitop::Msb,
};
// use ark_bn254::{G1Affine, G1Projective};
use crate::ecc::groups::wnaf;
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

const NUM_BITS: usize = 8;
// From what I've seen UL == u64
const NUM_BUCKETS: usize = (1u64 << NUM_BITS) as usize;
const MASK: u64 = NUM_BUCKETS as u64 - 1u64;

pub(crate) const fn get_optimal_bucket_width(num_points: usize) -> usize {
    if num_points >= 14617149 {
        return 21;
    }
    if num_points >= 1139094 {
        return 18;
    }
    // if (num_points >= 100000)
    if num_points >= 155975 {
        return 15;
    }
    if num_points >= 144834
    // if (num_points >= 100000)
    {
        return 14;
    }
    if num_points >= 25067 {
        return 12;
    }
    if num_points >= 13926 {
        return 11;
    }
    if num_points >= 7659 {
        return 10;
    }
    if num_points >= 2436 {
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
    return 1;
}

pub(crate) const fn get_num_buckets(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    return (1u64 << bits_per_bucket) as usize;
}

pub(crate) const fn get_num_rounds(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    wnaf::wnaf_size(bits_per_bucket + 1)
}

#[derive(Clone, Default, Debug)]
pub(crate) struct PippengerRuntimeState<Fr: Field, G: AffineRepr> {
    // TODO: maybe arc should be used here for threads. think later.
    // TODO: check why are they used, for now commenting them
    pub point_schedule: Vec<u64>,
    pub point_pairs_1: Vec<G>,
    pub point_pairs_2: Vec<G>,
    pub scratch_space: Vec<Fr>,
    pub skew_table: Vec<bool>,
    pub bucket_counts: Vec<u32>,
    pub bit_counts: Vec<u32>,
    pub bucket_empty_status: Vec<bool>,
    pub round_counts: Vec<u64>,
    pub num_points: u32,
}

#[derive(Default, Clone, Debug)]
pub(crate) struct AffinePippengerRuntimeState<Fr: Field, G: AffineRepr> {
    pub points: Vec<G>,
    pub point_pairs_1: Vec<G>,
    pub point_pairs_2: Vec<G>,
    pub scratch_space: Vec<Fr>,
    pub point_schedule: Vec<u64>,
    pub bucket_counts: Vec<u32>,
    pub bit_offsets: Vec<u32>,
    pub bucket_empty_status: Vec<bool>,
    pub num_points: usize,
    pub num_buckets: usize,
}

impl<Fr: Field, G: AffineRepr> PippengerRuntimeState<Fr, G> {
    pub(crate) fn new(num_initial_points: usize) -> Self {
        const MAX_NUM_ROUNDS: u32 = 256;
        let num_points = num_initial_points * 2;
        let num_points_floor: usize = 1 << num_points.get_msb();
        let num_buckets = 1 << get_optimal_bucket_width(num_initial_points);
        let num_threads = compute_num_threads();
        let prefetch_overflow = 16 * num_threads;
        let num_rounds = get_num_rounds(num_points_floor);

        let point_schedule = vec![0u64; num_points * num_rounds + prefetch_overflow];
        let point_pairs_1 = vec![G::default(); num_points * 2 + (num_threads * 16)];
        let point_pairs_2 = vec![G::default(); (num_points * 2) + (num_threads * 16)];
        let scratch_space = vec![Fr::default(); num_points];

        let skew_table = vec![false; num_points];
        let bucket_counts = vec![0u32; num_threads * num_buckets];
        let bit_counts = vec![0u32; num_threads * num_buckets];
        let bucket_empty_status = vec![false; num_threads * num_buckets];
        let round_counts = vec![0u64; MAX_NUM_ROUNDS as usize];

        PippengerRuntimeState {
            point_schedule,
            point_pairs_1,
            point_pairs_2,
            scratch_space,
            skew_table,
            bucket_counts,
            bit_counts,
            bucket_empty_status,
            round_counts,
            num_points: num_points as u32,
        }
    }

    pub(crate) fn get_affine_pippenger_runtime_state(
        &self,
        num_threads: usize,
        thread_index: usize,
    ) -> AffinePippengerRuntimeState<Fr, G> {
        let points_per_thread: usize = self.num_points as usize / (num_threads);
        let num_buckets = 1 << get_optimal_bucket_width(self.num_points as usize);
        // TODO: check if we can just send that particular thread's vars
        let point_pairs_1: Vec<G> = self
            .clone()
            .point_pairs_1
            .into_iter()
            .skip((points_per_thread + 16) * thread_index)
            .collect();
        let point_pairs_2 = self
            .clone()
            .point_pairs_2
            .into_iter()
            .skip((points_per_thread + 16) * thread_index)
            .collect();
        let scratch_space = self
            .clone()
            .scratch_space
            .into_iter()
            .skip(thread_index * (points_per_thread as usize / 2))
            .collect();
        let bucket_counts = self
            .clone()
            .bucket_counts
            .into_iter()
            .skip(thread_index * num_buckets)
            .collect();
        let bit_offsets = self
            .clone()
            .bit_counts
            .into_iter()
            .skip(thread_index * num_buckets)
            .collect();
        let bucket_empty_status = self
            .clone()
            .bucket_empty_status
            .into_iter()
            .skip(thread_index * num_buckets)
            .collect();

        // TODO: see how to initialise the vars not passed here
        AffinePippengerRuntimeState {
            points: Vec::default(),
            point_schedule: vec![],
            point_pairs_1,
            point_pairs_2,
            scratch_space,
            bucket_counts,
            bit_offsets,
            bucket_empty_status,
            num_points: 0,
            num_buckets: 0,
        }
    }

    pub(crate) fn pippenger_unsafe(
        &mut self,
        _mul_scalars: &mut [Fr],
        _srs_points: &[G],
        _msm_size: usize,
    ) -> G {
        todo!()
    }

    pub(crate) fn pippenger(
        &mut self,
        _scalars: &mut [Fr],
        _points: &[G],
        _num_initial_points: usize,
        _handle_edge_cases: bool,
    ) -> G {
        todo!()
    }
}

pub(crate) type GrumpkinRuntimeState = PippengerRuntimeState<grumpkin::Fq, grumpkin::SWAffine>;
pub(crate) type Bn254RuntimeState = PippengerRuntimeState<ark_bn254::Fq, ark_bn254::G1Affine>;
