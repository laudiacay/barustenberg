use std::marker::PhantomData;

use ark_bn254::{G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, Zero};

pub(crate) type G1AffineGroup = <ark_ec::short_weierstrass::Affine<
    <ark_bn254::Config as ark_ec::bn::BnConfig>::G1Config,
> as ark_ec::AffineRepr>::Group;

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

#[derive(Clone, Default, Debug)]
pub(crate) struct PippengerRuntimeState<Fr: Field + FftField, G: AffineRepr> {
    phantom: PhantomData<(Fr, G)>,
}

impl<Fr: Field + FftField, G: AffineRepr> PippengerRuntimeState<Fr, G> {
    pub(crate) fn new(_size: usize) -> Self {
        todo!()
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
pub(crate) fn generate_pippenger_point_table(
    points: &mut [G1Affine],
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
