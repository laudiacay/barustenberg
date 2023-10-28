use ark_ff::AdditiveGroup;
use ark_ff::One;
use std::time::SystemTime;

use crate::{
    common::max_threads::compute_num_threads,
    ecc::scalar_multiplication::{
        runtime_states::{get_num_buckets, AffinePippengerRuntimeState, PippengerRuntimeState},
        scalar_multiplication::{
            compute_wnaf_states, construct_addition_chains, organise_buckets, pippenger_unsafe,
            reduce_buckets,
        },
    },
    srs::{
        io::read_transcript,
        reference_string::{
            file_reference_string::FileReferenceStringFactory, ProverReferenceString,
            ReferenceStringFactory,
        },
    },
};
use ark_bn254::{Fq, Fr, G1Affine, G1Projective, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::{Field, UniformRand};
use ark_std::Zero;

use super::{
    runtime_states::get_num_rounds,
    scalar_multiplication::{self, generate_pippenger_point_table, pippenger},
};

type Bn254Config = ark_bn254::g1::Config;

fn get_points_table_size(num_points: usize) -> usize {
    let num_threads = compute_num_threads();
    let prefetch_overflow = 16 * num_threads;
    2 * num_points + prefetch_overflow
}

#[test]
fn test_reduce_buckets_simple() {
    const NUM_POINTS: usize = 128;
    let crs = FileReferenceStringFactory::new("../srs_db/ignition".to_string());
    let prs = crs.get_prover_crs(NUM_POINTS / 2).unwrap().unwrap();
    let monomials = prs.read().unwrap().get_monomial_points();

    let mut points_schedule = vec![0u64; get_points_table_size(NUM_POINTS / 2)];
    let mut transcript = vec![0u64; NUM_POINTS];
    let mut transcript_points = vec![0u64; NUM_POINTS];
    transcript_points[0] = 0x0;
    transcript_points[1] = 0x2;
    transcript_points[2] = 0x4;
    transcript_points[3] = 0x6;
    transcript_points[4] = 0xb;
    transcript_points[5] = 0xc;
    transcript_points[6] = 0xe;
    transcript_points[7] = 0x11;
    transcript_points[8] = 0x13;
    transcript_points[9] = 0x14;
    transcript_points[10] = 0x15;
    transcript_points[11] = 0x16;
    transcript_points[12] = 0x17;
    transcript_points[13] = 0x18;
    transcript_points[14] = 0x20;
    transcript_points[15] = 0x21;
    transcript_points[16] = 0x22;
    transcript_points[17] = 0x27;
    transcript_points[18] = 0x29;
    transcript_points[19] = 0x2b;
    transcript_points[20] = 0x2c;
    transcript_points[21] = 0x2d;
    transcript_points[22] = 0x2e;
    transcript_points[23] = 0x36;
    transcript_points[24] = 0x37;
    transcript_points[25] = 0x38;
    transcript_points[26] = 0x3e;
    transcript_points[27] = 0x3f;
    transcript_points[28] = 0x4e;
    transcript_points[29] = 0x4f;
    transcript_points[30] = 0x50;
    transcript_points[31] = 0x51;
    transcript_points[32] = 0x41;
    transcript_points[33] = 0x52;
    transcript_points[34] = 0x53;
    transcript_points[35] = 0x54;
    transcript_points[36] = 0x43;
    transcript_points[37] = 0x57;
    transcript_points[38] = 0x46;
    transcript_points[39] = 0x58;
    transcript_points[40] = 0x5b;
    transcript_points[41] = 0x5e;
    transcript_points[42] = 0x42;
    transcript_points[43] = 0x47;
    transcript_points[44] = 0x4b;
    transcript_points[45] = 0x4d;
    transcript_points[46] = 0x6b;
    transcript_points[47] = 0x65;
    transcript_points[48] = 0x6d;
    transcript_points[49] = 0x67;
    transcript_points[50] = 0x6f;
    transcript_points[51] = 0x68;
    transcript_points[52] = 0x69;
    transcript_points[53] = 0x6a;
    transcript_points[54] = 0x71;
    transcript_points[55] = 0x72;
    transcript_points[56] = 0x73;
    transcript_points[57] = 0x74;
    transcript_points[58] = 0x75;
    transcript_points[59] = 0x66;
    transcript_points[60] = 0x79;
    transcript_points[62] = 0x7c;
    transcript_points[61] = 0x7e;
    transcript_points[63] = 0x7f;
    transcript_points[64] = 0x1;
    transcript_points[65] = 0x3;
    transcript_points[66] = 0x5;
    transcript_points[67] = 0x7;
    transcript_points[68] = 0x8;
    transcript_points[69] = 0x9;
    transcript_points[70] = 0xa;
    transcript_points[71] = 0xd;
    transcript_points[72] = 0xf;
    transcript_points[73] = 0x10;
    transcript_points[74] = 0x12;
    transcript_points[75] = 0x19;
    transcript_points[76] = 0x1a;
    transcript_points[77] = 0x1b;
    transcript_points[78] = 0x1c;
    transcript_points[79] = 0x1d;
    transcript_points[80] = 0x1e;
    transcript_points[81] = 0x1f;
    transcript_points[82] = 0x23;
    transcript_points[83] = 0x24;
    transcript_points[84] = 0x25;
    transcript_points[85] = 0x26;
    transcript_points[86] = 0x28;
    transcript_points[87] = 0x2a;
    transcript_points[88] = 0x2f;
    transcript_points[89] = 0x30;
    transcript_points[90] = 0x31;
    transcript_points[91] = 0x32;
    transcript_points[92] = 0x33;
    transcript_points[93] = 0x34;
    transcript_points[94] = 0x35;
    transcript_points[95] = 0x39;
    transcript_points[96] = 0x3a;
    transcript_points[97] = 0x3b;
    transcript_points[98] = 0x3c;
    transcript_points[99] = 0x3d;
    transcript_points[100] = 0x48;
    transcript_points[101] = 0x49;
    transcript_points[102] = 0x55;
    transcript_points[103] = 0x56;
    transcript_points[104] = 0x4a;
    transcript_points[105] = 0x44;
    transcript_points[106] = 0x45;
    transcript_points[107] = 0x40;
    transcript_points[108] = 0x59;
    transcript_points[109] = 0x5a;
    transcript_points[110] = 0x5c;
    transcript_points[111] = 0x5d;
    transcript_points[112] = 0x5f;
    transcript_points[113] = 0x60;
    transcript_points[114] = 0x61;
    transcript_points[115] = 0x62;
    transcript_points[116] = 0x63;
    transcript_points[117] = 0x4c;
    transcript_points[118] = 0x6c;
    transcript_points[119] = 0x6e;
    transcript_points[120] = 0x64;
    transcript_points[121] = 0x70;
    transcript_points[122] = 0x77;
    transcript_points[123] = 0x78;
    transcript_points[124] = 0x76;
    transcript_points[125] = 0x7a;
    transcript_points[126] = 0x7b;
    transcript_points[127] = 0x7d;

    for i in 0..64 {
        transcript[i] = 0;
        transcript[i + 64] = 1;
    }

    for i in 0..NUM_POINTS {
        points_schedule[i] = transcript_points[i] << 32 + transcript[i];
    }

    let mut expected = vec![G1Projective::default(); NUM_POINTS];
    for i in 0..NUM_POINTS {
        let schedule = transcript[i] & 0x7FFFFFFFu64;
        expected[schedule as usize] += monomials[transcript_points[i] as usize];
    }

    let points_pairs = vec![G1Affine::default(); NUM_POINTS];
    let output_buckets = vec![G1Affine::default(); NUM_POINTS];
    let scratch_space = vec![Fq::default(); NUM_POINTS];
    let bucket_counts = vec![0u32; NUM_POINTS];
    let bit_offsets = vec![0u32; NUM_POINTS];
    let bucket_empty_status = vec![true; NUM_POINTS];
    let mut product_state = AffinePippengerRuntimeState::<Bn254Config> {
        points: monomials.to_vec(),
        point_pairs_1: points_pairs,
        point_pairs_2: output_buckets,
        scratch_space,
        bucket_counts,
        bit_offsets,
        point_schedule: points_schedule,
        num_points: NUM_POINTS,
        num_buckets: 2,
        bucket_empty_status,
    };

    let output = reduce_buckets::<Bn254Config>(&mut product_state, true, false);

    for i in 0..product_state.num_buckets {
        let affine_expected_point: G1Affine = expected[i].into();
        assert_eq!(output[i], affine_expected_point);
    }
}

#[test]
fn test_reduce_buckets() {
    let num_initial_points = 1 << 12;
    let num_points = num_initial_points * 2;

    let mut monomials = vec![G1Affine::default(); num_points * 2];
    let mut g2_x = G2Affine::default();
    read_transcript(
        &mut monomials,
        &mut g2_x,
        num_initial_points,
        "../srs_db/ignition",
    );

    let monomials_clone = monomials.clone();
    generate_pippenger_point_table(&monomials_clone, &mut monomials, num_initial_points);

    let mut rng = ark_std::test_rng();
    let mut scalars = vec![Fr::default(); num_initial_points];
    for i in 0..num_initial_points {
        scalars[i] = Fr::rand(&mut rng);
    }

    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_initial_points);

    let mut now = SystemTime::now();
    compute_wnaf_states::<Bn254Config>(
        &mut state.point_schedule,
        &mut state.skew_table,
        &mut state.round_counts,
        &mut scalars,
        num_initial_points,
    );
    println!("wnaf time: {} ms", now.elapsed().unwrap().as_millis());

    now = SystemTime::now();
    organise_buckets(&mut state.point_schedule, num_points);
    println!(
        "organize bucket time: {} ms",
        now.elapsed().unwrap().as_millis()
    );

    now = SystemTime::now();

    let max_num_buckets = get_num_buckets(num_points * 2);

    let mut point_schedule_copy = vec![0u64; num_points * 2];
    for i in 0..num_points {
        state.point_schedule[i + num_points] =
            state.point_schedule[i + num_points] & 0xFFFFFFFF7FFFFFFFu64;
        point_schedule_copy[i] = state.point_schedule[i + num_points];
    }

    let first_bucket = point_schedule_copy[0] & 0x7FFFFFFFu64;
    let last_bucket = point_schedule_copy[num_points - 1] & 0x7FFFFFFFu64;
    let num_buckets = last_bucket - first_bucket + 1;

    let points_pairs = vec![G1Affine::default(); num_points * 2];
    let scratch_points = vec![G1Affine::default(); num_points * 2];
    let scratch_space = vec![Fq::default(); num_points * 2];
    let bucket_counts = vec![0u32; num_buckets as usize * 100];
    let bit_offsets = vec![0u32; 22];
    let bucket_empty_status = vec![true; num_points * 2];
    let mut affine_product_state = AffinePippengerRuntimeState::<Bn254Config> {
        points: monomials.to_vec(),
        point_pairs_1: points_pairs,
        point_pairs_2: scratch_points,
        scratch_space,
        bucket_counts,
        bit_offsets,
        point_schedule: state
            .point_schedule
            .into_iter()
            .skip(num_points)
            .take(num_points)
            .collect::<Vec<_>>(),
        num_points,
        num_buckets: num_buckets as usize,
        bucket_empty_status,
    };

    let mut expected_buckets = vec![G1Projective::default(); num_points * 2 as usize];
    for i in 0..num_points {
        let schedule = point_schedule_copy[i];
        let bucket_index = schedule as usize & 0x7FFFFFFFu64 as usize;
        let points_index = schedule >> 32;
        let predicate = (schedule >> 31) & 1;

        if predicate == 1 {
            expected_buckets[bucket_index - first_bucket as usize] -=
                monomials[points_index as usize];
        } else {
            expected_buckets[bucket_index - first_bucket as usize] +=
                monomials[points_index as usize];
        }
    }

    let mut it = 0;
    let output_buckets = reduce_buckets(&mut affine_product_state, true, false);

    println!("num_buckets: {}", num_buckets);
    for i in 0..num_buckets {
        if !affine_product_state.bucket_empty_status[i as usize] {
            let expected = expected_buckets[i as usize];
            assert_eq!(output_buckets[it], expected);
            it += 1;
        } else {
            println!("recorded empty bucket at index {}", i);
        }
    }
}

fn test_add_affine_points<F: Field, G: AffineRepr<BaseField = F>>() {
    let num_points = 20;
    let mut rng = ark_std::test_rng();

    let mut points = (0..num_points)
        .map(|_| G::rand(&mut rng))
        .collect::<Vec<G>>();
    let mut points_copy = points.clone();
    let mut scratch_space = vec![F::default(); num_points * 2];
    let mut count = num_points - 1;
    for i in (0..num_points - 2).rev().step_by(2) {
        points_copy[count] = (points_copy[i] + points_copy[i + 1]).into();
        count -= 1;
        // TODO: what's this?
        // points_copy[count + 1] = points_copy[count + 1].normalize();
    }

    scalar_multiplication::add_affine_points(
        points.as_mut_slice(),
        num_points,
        scratch_space.as_mut_slice(),
    );
    for i in (num_points - 1 - (num_points / 2)..num_points).rev() {
        assert_eq!(points[i], points_copy[i]);
    }
}

#[test]
fn test_add_affine_points_generic() {
    test_add_affine_points::<ark_bn254::Fq, ark_bn254::G1Affine>();
    test_add_affine_points::<grumpkin::Fq, grumpkin::Affine>();
}

#[test]
fn test_construct_addition_chains() {
    let num_initial_points = 1 << 12;
    let num_points = num_initial_points * 2;

    let mut monomials = vec![G1Affine::default(); num_points * 2];
    let mut g2_x = G2Affine::default();
    read_transcript(
        &mut monomials,
        &mut g2_x,
        num_initial_points,
        "../srs_db/ignition",
    );

    let mut rng = ark_std::test_rng();
    let mut source_scalar = Fr::rand(&mut rng);
    let mut scalars = vec![Fr::default(); num_initial_points];
    for i in 0..num_initial_points {
        source_scalar.square_in_place();
        scalars[i] = source_scalar;
    }

    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_initial_points);
    let monomials_clone = monomials.clone();
    generate_pippenger_point_table(&monomials_clone, &mut monomials, num_initial_points);

    let mut now = SystemTime::now();
    compute_wnaf_states::<Bn254Config>(
        &mut state.point_schedule,
        &mut state.skew_table,
        &mut state.round_counts,
        &mut scalars,
        num_initial_points,
    );
    println!("wnaf time: {} ms", now.elapsed().unwrap().as_millis());

    now = SystemTime::now();
    organise_buckets(&mut state.point_schedule, num_points);
    println!(
        "organize bucket time: {} ms",
        now.elapsed().unwrap().as_millis()
    );

    let max_num_buckets = get_num_buckets(num_points * 2);

    let bit_offsets = vec![0u32; 22];
    let bucket_empty_status = vec![true; num_points * 2];

    let first_bucket = state.point_schedule[0] & 0x7FFFFFFFu64;
    let last_bucket = state.point_schedule[num_points - 1] & 0x7FFFFFFFu64;
    let num_buckets = last_bucket - first_bucket + 1;

    let points_pairs = vec![G1Affine::default(); num_points * 2];
    let scratch_points = vec![G1Affine::default(); num_points * 2];
    let bucket_counts = vec![0u32; max_num_buckets];

    // TODO: might have to use Rc for points here
    let mut affine_product_state = AffinePippengerRuntimeState::<Bn254Config> {
        points: monomials.to_vec(),
        point_pairs_1: points_pairs,
        point_pairs_2: scratch_points,
        scratch_space: vec![],
        bucket_counts,
        bit_offsets,
        point_schedule: state.point_schedule,
        num_points,
        num_buckets: num_buckets as usize,
        bucket_empty_status,
    };

    now = SystemTime::now();
    construct_addition_chains(&mut affine_product_state, true);
    println!(
        "construct addition chains: {} ms",
        now.elapsed().unwrap().as_millis()
    );
}

// #[test]
// fn test_endomorphism_split() {
//     let mut rng = ark_std::test_rng();
//     let scalar = Fr::rand(&mut rng);

//     let expected = G1Affine::default() * scalar;
//     let split_scalars: Vec<Fr> = split_into_endomorphism_scalars(scalar);

//     let k1 = split_scalars[0];
//     let k2 = split_scalars[1];

//     let t1 = G1Affine::default() * k1;
//     let beta = cube_root_of_unity::<Fq>();
//     let mut generator = G1Affine::default();

//     generator.x = generator.x * beta;
//     generator.y = -generator.y;
//     let t2 = generator * k2;
//     let result = t1 + t2;

//     assert_eq!(expected, result);
// }
#[test]
fn test_organise_buckets() {
    let target_degree = 1 << 8;
    let num_rounds = get_num_rounds(target_degree * 2);
    let num_points = target_degree * 2;

    let mut monomials = vec![G1Affine::default(); num_points * 2];
    let mut g2_x = G2Affine::default();
    read_transcript(
        &mut monomials,
        &mut g2_x,
        target_degree,
        "../srs_db/ignition",
    );

    let mut rng = ark_std::test_rng();
    let mut source_scalar = Fr::rand(&mut rng);
    let mut scalars = vec![Fr::default(); target_degree];
    for i in 0..target_degree {
        source_scalar.square_in_place();
        scalars[i] = source_scalar;
    }

    let mut state = PippengerRuntimeState::<Bn254Config>::new(target_degree);
    compute_wnaf_states::<Bn254Config>(
        &mut state.point_schedule,
        &mut state.skew_table,
        &mut state.round_counts,
        &mut scalars,
        target_degree,
    );
    let wnaf_copy = state.point_schedule.clone();

    organise_buckets(&mut state.point_schedule, target_degree * 2);

    for i in 0..num_rounds {
        let unsorted_wnaf = wnaf_copy
            .clone()
            .into_iter()
            .skip(i * num_rounds)
            .take(target_degree * 2)
            .collect::<Vec<u64>>();
        // TODO: this clone isn't correct, check at other places too
        let sorted_wnaf = state
            .point_schedule
            .clone()
            .into_iter()
            .skip(i * num_rounds)
            .take(target_degree * 2)
            .collect::<Vec<u64>>();

        // match elements of sorted_wnaf in unsorted_wnaf idiomatically
        for j in 0..target_degree * 2 {
            let index = unsorted_wnaf
                .iter()
                .position(|&x| x == sorted_wnaf[j])
                .unwrap();
            assert!(index < target_degree * 2);
            assert!((sorted_wnaf[j] & 0x7FFFFFFFu64) >= (sorted_wnaf[j - 1] & 0x7FFFFFFFu64));
        }
    }
}

#[test]
fn test_pippenger_oversized_inputs() {
    let transcript_degree = 1 << 20;
    let target_degree = 1200000;

    let mut monomials = vec![G1Affine::default(); target_degree * 2];
    let mut g2_x = G2Affine::default();
    read_transcript(
        &mut monomials,
        &mut g2_x,
        transcript_degree,
        "../srs_db/ignition",
    );

    monomials.copy_within(
        0..(2 * target_degree - 2 * transcript_degree),
        2 * transcript_degree,
    );
    let monomials_clone = monomials.clone();
    generate_pippenger_point_table(&monomials_clone, monomials.as_mut_slice(), target_degree);

    let mut rng = ark_std::test_rng();
    let source_scalar = Fr::rand(&mut rng);
    let mut accumulator = source_scalar;
    let mut scalars = vec![Fr::default(); target_degree];
    for i in 0..target_degree {
        accumulator *= source_scalar;
        scalars[i] = accumulator;
    }

    let mut state = PippengerRuntimeState::<Bn254Config>::new(target_degree);
    // TODO: add both base field and scalar field generic type
    let first = pippenger::<Bn254Config>(
        scalars.as_mut_slice(),
        monomials.as_mut_slice(),
        target_degree,
        &mut state,
        true,
    );
    // TODO: check what is normalize in arkworks
    // first = first.normalize();

    scalars.iter_mut().for_each(|x| *x = *x.neg_in_place());

    let mut state_2 = PippengerRuntimeState::<Bn254Config>::new(target_degree);
    // TODO: add both base field and scalar field generic type
    let second = pippenger::<Bn254Config>(
        scalars.as_mut_slice(),
        monomials.as_mut_slice(),
        target_degree,
        &mut state_2,
        true,
    );
    // TODO: check what is normalize in arkworks
    // second = second.normalize();

    assert_eq!(first.x, second.x);
    // assert_eq!(first.z, second.z);
    // assert_eq!(first.z, Fq::one());
    assert_eq!(first.y, -second.y);
}

#[test]
fn test_pippenger_undersized_inputs() {
    let num_points = 17;

    let mut rng = ark_std::test_rng();
    let mut scalars = vec![Fr::default(); num_points];
    let mut points = vec![G1Affine::default(); num_points * 2 + 1];
    for i in 0..num_points {
        scalars[i] = Fr::rand(&mut rng);
        points[i] = G1Affine::rand(&mut rng);
    }

    let mut expected = G1Projective::default();
    for (_, (point, scalar)) in points.iter().zip(&scalars).enumerate() {
        expected += *point * scalar;
    }
    // TODO: check normalize
    // expected = expected.normalize();

    generate_pippenger_point_table(&points.clone(), &mut points, num_points);
    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_points);

    let result = pippenger::<Bn254Config>(
        &mut scalars,
        points.as_mut_slice(),
        num_points,
        &mut state,
        true,
    );

    assert_eq!(result, expected);
}

#[test]
fn test_pippenger_small() {
    let num_points = 8192;

    let mut rng = ark_std::test_rng();
    let mut scalars = vec![Fr::default(); num_points];
    let mut points = vec![G1Affine::default(); num_points * 2 + 1];
    for i in 0..num_points {
        scalars[i] = Fr::rand(&mut rng);
        points[i] = G1Affine::rand(&mut rng);
    }

    let mut expected = G1Projective::default();
    for (_, (point, scalar)) in points.iter().zip(&scalars).enumerate() {
        expected += *point * scalar;
    }
    // TODO: check normalize
    // expected = expected.normalize();

    generate_pippenger_point_table(&points.clone(), &mut points, num_points);
    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_points);

    let result = pippenger::<Bn254Config>(
        &mut scalars,
        points.as_mut_slice(),
        num_points,
        &mut state,
        true,
    );

    assert_eq!(result, expected);
}

#[test]
fn test_pippenger_edge_case_dbl() {
    let num_points = 128;

    let mut rng = ark_std::test_rng();
    let mut scalars = vec![Fr::default(); num_points];
    let mut points = vec![G1Affine::default(); num_points * 2 + 1];
    let point = G1Affine::rand(&mut rng);
    for i in 0..num_points {
        scalars[i] = Fr::rand(&mut rng);
        points[i] = point.clone();
    }

    let mut expected = G1Projective::default();
    for (_, (point, scalar)) in points.iter().zip(&scalars).enumerate() {
        expected += *point * scalar;
    }
    // TODO: check normalize
    // expected = expected.normalize();

    generate_pippenger_point_table(&points.clone(), &mut points, num_points);
    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_points);

    let result = pippenger::<Bn254Config>(
        &mut scalars,
        points.as_mut_slice(),
        num_points,
        &mut state,
        true,
    );

    assert_eq!(result, expected);
}

#[test]
fn test_pippenger_short_inputs() {
    // TODO: implement this
}

#[test]
fn test_pippenger_unsafe() {
    let num_points = 8192;

    let mut rng = ark_std::test_rng();
    let mut scalars = vec![Fr::default(); num_points];
    let mut points = vec![G1Affine::default(); num_points * 2 + 1];
    let point = G1Affine::rand(&mut rng);
    for i in 0..num_points {
        scalars[i] = Fr::rand(&mut rng);
        points[i] = point.clone();
    }

    let mut expected = G1Projective::default();
    for (_, (point, scalar)) in points.iter().zip(&scalars).enumerate() {
        expected += *point * scalar;
    }
    // TODO: check normalize
    // expected = expected.normalize();

    generate_pippenger_point_table(&points.clone(), &mut points, num_points);
    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_points);

    let result = pippenger_unsafe::<Bn254Config>(
        &mut scalars,
        points.as_mut_slice(),
        num_points,
        &mut state,
    );

    assert_eq!(result, expected);
}

#[test]
fn test_pippenger_one() {
    let num_points = 1;

    let mut rng = ark_std::test_rng();
    let mut scalars = vec![Fr::default(); num_points];
    let mut points = vec![G1Affine::default(); num_points * 2 + 1];
    for i in 0..num_points {
        scalars[i] = Fr::one();
        points[i] = G1Affine::rand(&mut rng);
    }

    let mut expected = G1Projective::default();
    for (_, (point, scalar)) in points.iter().zip(&scalars).enumerate() {
        expected += *point * scalar;
    }
    // TODO: check normalize
    // expected = expected.normalize();

    generate_pippenger_point_table(&points.clone(), &mut points, num_points);
    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_points);

    let result = pippenger::<Bn254Config>(
        &mut scalars,
        points.as_mut_slice(),
        num_points,
        &mut state,
        true,
    );

    assert_eq!(result, expected);
}

#[test]
fn test_pippenger_zero_points() {
    let num_points = 0;
    let mut scalars = vec![Fr::default(); 1];
    let mut points = vec![G1Affine::default(); 2 + 1];

    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_points);

    let result = pippenger::<Bn254Config>(
        &mut scalars,
        points.as_mut_slice(),
        num_points,
        &mut state,
        true,
    );

    assert!(result.is_zero());
}

#[test]
fn test_pippenger_mul_by_zero() {
    let num_points = 1;

    let mut rng = ark_std::test_rng();
    let mut scalars = vec![Fr::default(); num_points];
    let mut points = vec![G1Affine::default(); num_points * 2 + 1];
    scalars[0] = Fr::zero();
    // TODO: check if default is correct
    points[0] = G1Affine::default();

    generate_pippenger_point_table(&points.clone(), &mut points, num_points);
    let mut state = PippengerRuntimeState::<Bn254Config>::new(num_points);

    let result = pippenger::<Bn254Config>(
        &mut scalars,
        points.as_mut_slice(),
        num_points,
        &mut state,
        true,
    );

    assert!(result.is_zero());
}
