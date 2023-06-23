use std::marker::PhantomData;

use crate::ecc::fieldext::FieldExt;
use ark_bn254::G1Affine;
use ark_ec::Group;

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

pub(crate) fn is_on_curve<G: Group>(point: &G) -> bool {
    todo!("is_on_curve")
}

pub(crate) fn is_point_at_infinity<G: Group>(point: &G) -> bool {
    todo!("is_point_at_infinity")
}

#[derive(Clone, Default)]
pub(crate) struct PippengerRuntimeState<Fr: ark_ff::FftField + ark_ff::Field + FieldExt, G: Group> {
    phantom: PhantomData<(Fr, G)>,
}

impl<Fr: ark_ff::FftField + ark_ff::Field + FieldExt, G: Group> PippengerRuntimeState<Fr, G> {
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
}

pub fn generate_pippenger_point_table(
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
