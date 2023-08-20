use ark_bn254::Fr;

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::{Field, Zero};

use crate::srs::io::read_transcript_g1;

use anyhow::Result;

#[inline]
fn cube_root_of_unity<F: ark_ff::Field>() -> F {
    // // endomorphism i.e. lambda * [P] = (beta * x, y)
    // if constexpr (Params::cube_root_0 != 0) {
    //     constexpr field result{
    //         Params::cube_root_0, Params::cube_root_1, Params::cube_root_2, Params::cube_root_3
    //     };
    //     return result;
    // } else {
    let two_inv = F::from(2_u32).inverse().unwrap();
    let numerator = (-F::from(3_u32)).sqrt().unwrap() - F::from(1_u32);
    two_inv * numerator
    // constexpr field two_inv = field(2).invert();
    // constexpr field numerator = (-field(3)).sqrt() - field(1);
    // constexpr field result = two_inv * numerator;
    // return result;
    //}
}
pub(crate) fn is_point_at_infinity(point: &G1Projective) -> bool {
    !(point.x.is_zero() && point.y.is_zero()) && point.z.is_zero()
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Pippenger {
    monomials: Vec<G1Affine>,
    num_points: usize,
}

impl Pippenger {
    pub(crate) fn get_num_points(&self) -> usize {
        todo!()
    }

    pub(crate) fn from_points(_points: &[G1Affine], _num_points: usize) -> Self {
        todo!()
    }

    pub(crate) fn from_path(path: &str, num_points: usize) -> Result<Self> {
        let mut monomials = vec![G1Affine::default(); num_points];
        read_transcript_g1(&mut monomials, num_points, path)?;
        let point_table = monomials.clone();
        generate_pippenger_point_table(&point_table, &mut monomials, num_points);
        Ok(Self {
            monomials,
            num_points,
        })
    }
}

#[derive(Clone, Default, Debug)]
pub(crate) struct PippengerRuntimeState<Fq: Field, G1Affine: AffineRepr> {
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

impl<Fq: Field, G1: AffineRepr> PippengerRuntimeState<Fq, G1> {
    pub(crate) fn new(_size: usize) -> Self {
        todo!()
    }
    pub(crate) fn pippenger_unsafe(
        &mut self,
        _mul_scalars: &mut [Fr],
        _srs_points: &[G1],
        _msm_size: usize,
    ) -> G1 {
        todo!()
    }

    pub(crate) fn pippenger(
        &mut self,
        _scalars: &mut [Fr],
        _points: &[G1],
        _num_initial_points: usize,
        _handle_edge_cases: bool,
    ) -> G1 {
        todo!()
    }
}
pub(crate) fn generate_pippenger_point_table(
    points: &[G1Affine],
    table: &mut [G1Affine],
    num_points: usize,
) {
    // calculate the cube root of unity
    let beta = cube_root_of_unity::<ark_bn254::Fq>();

    // iterate backwards, so that `points` and `table` can point to the same memory location
    for i in (0..num_points).rev() {
        table[i * 2] = points[i];
        table[i * 2 + 1].x = beta * points[i].x;
        table[i * 2 + 1].y = -points[i].y;
    }
}
