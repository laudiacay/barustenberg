use ark_ec::{
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr,
};
use ark_ff::{FftField, Field};

use super::{cube_root_of_unity, runtime_states::PippengerRuntimeState};

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

fn pippenger_internal<F: Field + FftField, G: AffineRepr, C: SWCurveConfig>(
    scalars: &mut [F],
    points: &mut [G],
    num_initial_points: usize,
    state: &PippengerRuntimeState<F, G>,
    handle_edge_cases: bool,
) -> G {
    todo!("implement");
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
