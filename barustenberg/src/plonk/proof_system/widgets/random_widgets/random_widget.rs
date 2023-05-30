use std::marker::PhantomData;

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::{
    plonk::proof_system::proving_key::ProvingKey,
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub(crate) struct ProverRandomWidget<
    'a,
    H: BarretenHasher,
    Fr: Field + FftField,
    G1Affine: AffineRepr,
> {
    pub(crate) key: ProvingKey<'a, Fr, G1Affine>,
    phantom: PhantomData<H>,
}

impl<'a, H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr>
    ProverRandomWidget<'a, H, Fr, G1Affine>
{
    pub(crate) fn new(_input_key: &ProvingKey<'a, Fr, G1Affine>) -> Self {
        todo!("ProverRandomWidget::new")
    }

    fn compute_round_commitments(
        &self,
        _transcript: &mut Transcript<H, Fr, G1Affine>,
        _size: usize,
        _work_queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
    ) {
        todo!("ProverRandomWidget::compute_round_commitments")
    }

    fn compute_quotient_contribution(
        &self,
        _alpha_base: Fr,
        _transcript: &Transcript<H, Fr, G1Affine>,
    ) -> Fr {
        todo!("ProverRandomWidget::compute_quotient_contribution")
    }
}

impl<'a, H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr> PartialEq
    for ProverRandomWidget<'_, H, Fr, G1Affine>
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
    for ProverRandomWidget<'_, H, Fr, G1Affine>
{
}
