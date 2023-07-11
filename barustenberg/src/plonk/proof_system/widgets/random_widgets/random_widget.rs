use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::{
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub(crate) trait ProverRandomWidget<'a>: std::fmt::Debug + 'a {
    type Hasher: BarretenHasher + 'a;
    type Fr: Field + FftField + 'a;
    type G1: AffineRepr + 'a;

    fn compute_round_commitments(
        &self,
        _transcript: &mut Transcript<Self::Hasher>,
        _size: usize,
        _work_queue: &mut WorkQueue<'a, Self::Hasher, Self::Fr, Self::G1>,
    );

    fn compute_quotient_contribution(
        &self,
        _alpha_base: Self::Fr,
        _transcript: &Transcript<Self::Hasher>,
    ) -> Self::Fr;
}
