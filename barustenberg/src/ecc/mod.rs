use ark_bn254::{Fq12, G1Affine};

// TODO todo - stubs to get the compiler to cooperate.
pub(crate) mod curves;

struct EllCoeffs<QuadFP: ark_ff::Field> {
    o: QuadFP,
    vw: QuadFP,
    vv: QuadFP,
}

const PRECOMPUTED_COEFFICIENTS_LENGTH: usize = 87;

pub(crate) struct MillerLines {
    lines: [EllCoeffs<Fq12>; PRECOMPUTED_COEFFICIENTS_LENGTH],
}

pub(crate) fn reduced_ate_pairing_batch_precomputed(
    _p_affines: &[G1Affine],
    _miller_lines: &MillerLines,
    _num_points: usize,
) -> Fq12 {
    // TODO compilation placeholder come back here bb
    todo!("see comment")
}

#[derive(Clone, Default)]
pub(crate) struct PippengerRuntimeState {}
