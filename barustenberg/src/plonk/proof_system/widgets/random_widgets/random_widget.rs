use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::{
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub(crate) trait ProverRandomWidget<
    'a,
    H: BarretenHasher,
    Fr: Field + FftField,
    G1Affine: AffineRepr,
>
{
    fn compute_round_commitments(
        &self,
        _transcript: &mut Transcript<H, Fr, G1Affine>,
        _size: usize,
        _work_queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
    );

    fn compute_quotient_contribution(
        &self,
        _alpha_base: Fr,
        _transcript: &Transcript<H, Fr, G1Affine>,
    ) -> Fr;
}
