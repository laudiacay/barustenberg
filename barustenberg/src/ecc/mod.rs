use std::marker::PhantomData;

use ark_bn254::{Fq12, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::Field;
use num_bigint::BigUint;

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
pub(crate) struct PippengerRuntimeState<Fr: Field, G1Affine: AffineRepr> {
    phantom: PhantomData<(Fr, G1Affine)>,
}

impl<Fr: Field, G1Affine: AffineRepr> PippengerRuntimeState<Fr, G1Affine> {
    pub(crate) fn new(_size: usize) -> Self {
        todo!()
    }
    pub(crate) fn pippenger_unsafe(
        &mut self,
        _mul_scalars: &mut [Fr],
        _srs_points: &[G1Affine],
        _msm_size: usize,
    ) -> G1Affine {
        todo!()
    }
}

#[inline]
pub(crate) fn conditionally_subtract_from_double_modulus<Fr: Field>(
    this: &Fr,
    predicate: u64,
) -> Fr {
    todo!("see comment")
    // yikes man
}

#[inline]
pub(crate) fn tag_coset_generator<Fr: Field>() -> Fr {
    todo!("see comment")
    // yikes man
}
#[inline]
pub(crate) fn coset_generator<Fr: Field>(_n: u8) -> Fr {
    todo!("see comment")
    // yikes man
}
