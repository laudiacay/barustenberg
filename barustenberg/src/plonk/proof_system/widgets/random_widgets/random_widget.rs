use std::marker::PhantomData;

use crate::{
    ecc::fields::field::FieldParams,
    plonk::proof_system::proving_key::ProvingKey,
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub struct ProverRandomWidget<H: BarretenHasher, Fr: FieldParams> {
    phantom: PhantomData<(H, Fr)>,
}

impl<H: BarretenHasher, Fr: FieldParams> ProverRandomWidget<H, Fr> {
    pub fn new(input_key: &ProvingKey<Fr>) -> Self {
        todo!("ProverRandomWidget::new")
    }

    fn compute_round_commitments(
        &self,
        transcript: &mut Transcript<H>,
        size: usize,
        work_queue: &mut WorkQueue<H, Fr>,
    ) {
        todo!("ProverRandomWidget::compute_round_commitments")
    }

    fn compute_quotient_contribution(&self, alpha_base: Fr, transcript: &Transcript<H>) -> Fr {
        todo!("ProverRandomWidget::compute_quotient_contribution")
    }
}

impl<H: BarretenHasher, Fr: FieldParams> Clone for Box<ProverRandomWidget<H, Fr>> {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

pub trait BoxedCloneProverRandomWidget {
    fn boxed_clone(&self) -> Self;
}

impl<H: BarretenHasher, Fr: FieldParams> BoxedCloneProverRandomWidget for ProverRandomWidget<H, Fr> {
    fn boxed_clone(&self) -> Box<ProverRandomWidget<H, Fr>> {
        Box::new(self.clone())
    }
}

impl<H: BarretenHasher, Fr: FieldParams> PartialEq for ProverRandomWidget<H, Fr> {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

// // TODO is this right
// impl<H: BarretenHasher, Fr: FieldParams> PartialEq for ProverRandomWidget<H, Fr> {
//     fn eq(&self, other: &Self) -> bool {
//         self.key() == other.key()
//     }
// }

impl<H: BarretenHasher, Fr: FieldParams> Eq for ProverRandomWidget<H, Fr> {}
