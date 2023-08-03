//pub mod runtime_states;
pub(crate) mod process_buckets;
pub(crate) mod runtime_states;
pub(crate) mod scalar_multiplication;
// pub(crate) mod wnaf;

// #[cfg(test)]
// pub mod tests;

use ark_bn254::{Fq, G1Affine};
use ark_ff::Field;

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

pub(crate) fn generate_pippenger_point_table<F: Field>(
    points: &mut [G1Affine],
    table: &mut [G1Affine],
    num_points: usize,
) {
    // calculate the cube root of unity
    let beta = cube_root_of_unity::<Fq>();

    // iterate backwards, so that `points` and `table` can point to the same memory location
    for i in (0..num_points).rev() {
        table[i * 2] = points[i];
        table[i * 2 + 1].x = beta * points[i].x;
        table[i * 2 + 1].y = -points[i].y;
    }
}
