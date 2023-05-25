use std::marker::PhantomData;

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::{
    plonk::proof_system::proving_key::ProvingKey,
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub struct ProverRandomWidget<H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr> {
    pub key: ProvingKey<Fr, G1Affine>,
    phantom: PhantomData<H>,
}

impl<H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr>
    ProverRandomWidget<H, Fr, G1Affine>
{
    pub fn new(input_key: &ProvingKey<Fr, G1Affine>) -> Self {
        todo!("ProverRandomWidget::new")
    }

    fn compute_round_commitments(
        &self,
        transcript: &mut Transcript<H>,
        size: usize,
        work_queue: &mut WorkQueue<H, Fr, G1Affine>,
    ) {
        todo!("ProverRandomWidget::compute_round_commitments")
    }

    fn compute_quotient_contribution(&self, alpha_base: Fr, transcript: &Transcript<H>) -> Fr {
        todo!("ProverRandomWidget::compute_quotient_contribution")
    }
}

impl<H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr> Clone
    for Box<ProverRandomWidget<H, Fr, G1Affine>>
{
    fn clone(&self) -> Self {
        Box::new(*self.as_ref().clone())
    }
}

impl<H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr> PartialEq
    for ProverRandomWidget<H, Fr, G1Affine>
{
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

// // TODO is this right
// impl<H: BarretenHasher, Fr: FieldParams> PartialEq for ProverRandomWidget<H, Fr> {
//     fn eq(&self, other: &Self) -> bool {
//         self.key() == other.key()
//     }
// }

impl<H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr> Eq
    for ProverRandomWidget<H, Fr, G1Affine>
{
}
