use std::marker::PhantomData;

use ark_bn254::Fq12;
use ark_ec::Group;

use self::fieldext::FieldExt;

pub(crate) mod fieldext;
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

pub(crate) fn reduced_ate_pairing_batch_precomputed<G: Group>(
    _p_affines: &[G],
    _miller_lines: &Vec<MillerLines>,
    _num_points: usize,
) -> Fq12 {
    // TODO compilation placeholder come back here bb
    todo!("see comment")
}

#[derive(Clone, Default)]
pub(crate) struct PippengerRuntimeState<Fr: FieldExt, G: Group> {
    phantom: PhantomData<(Fr, G)>,
}

impl<Fr: FieldExt, G: Group> PippengerRuntimeState<Fr, G> {
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

#[inline]
pub(crate) fn conditionally_subtract_from_double_modulus<Fr: FieldExt>(
    this: &Fr,
    predicate: u64,
) -> Fr {
    todo!("see comment")
    // yikes man
}

#[inline]
pub(crate) fn tag_coset_generator<Fr: FieldExt>() -> Fr {
    todo!("see comment")
    // yikes man
}
#[inline]
pub(crate) fn coset_generator<Fr: FieldExt>(_n: u8) -> Fr {
    todo!("see comment")
    // yikes man
}
