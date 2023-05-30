use ark_bn254::{Fq12, G1Affine};

// TODO todo - stubs to get the compiler to cooperate.
pub(crate) mod curves;

struct EllCoeffs<QuadFP: ark_ff::Field> {
    o: QuadFP,
    vw: QuadFP,
    vv: QuadFP,
}

const PRECOMPUTED_COEFFICIENTS_LENGTH: usize = 87;

struct MillerLines {
    lines: [EllCoeffs<Fq12>; PRECOMPUTED_COEFFICIENTS_LENGTH],
}

pub fn reduced_ate_pairing_batch_precomputed(
    p_affines: &[G1Affine],
    miller_lines: &MillerLines,
    num_points: usize,
) -> Fq12 {
    // TODO compilation placeholder come back here bb
    todo!("see comment")
}

#[derive(Clone, Default)]
pub struct PippengerRuntimeState {}
