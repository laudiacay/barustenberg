use ark_ec::AffineRepr;

use crate::{
    ecc::fieldext::FieldExt,
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub(crate) trait ProverRandomWidget<
    'a,
    H: BarretenHasher,
    Fr: ark_ff::Field + ark_ff::FftField + FieldExt,
    G: AffineRepr,
>
{
    fn compute_round_commitments(
        &self,
        _transcript: &mut Transcript<H>,
        _size: usize,
        _work_queue: &mut WorkQueue<'a, H, Fr, G>,
    );

    fn compute_quotient_contribution(&self, _alpha_base: Fr, _transcript: &Transcript<H>) -> Fr;
}
