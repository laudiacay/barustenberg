use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::{
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub(crate) trait ProverRandomWidget: std::fmt::Debug {
    type Hasher: BarretenHasher;
    type Fr: Field + FftField;
    type G1: AffineRepr;

    fn compute_round_commitments(
        &self,
        _transcript: &mut Transcript<Self::Hasher>,
        _size: usize,
        _work_queue: &mut WorkQueue<Self::Hasher>,
    );

    fn compute_quotient_contribution(
        &self,
        _alpha_base: Self::Fr,
        _transcript: &Transcript<Self::Hasher>,
    ) -> Self::Fr;
}
