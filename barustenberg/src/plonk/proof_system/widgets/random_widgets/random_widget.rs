use crate::{
    ecc::fields::field::FieldParams,
    plonk::proof_system::proving_key::ProvingKey,
    proof_system::work_queue::WorkQueue,
    transcript::{BarretenHasher, Transcript},
};

pub trait ProverRandomWidget<H: BarretenHasher, Fr: FieldParams> {
    fn new(input_key: &ProvingKey<Fr>) -> Self;

    fn compute_round_commitments(
        &self,
        transcript: &mut Transcript<H>,
        size: usize,
        work_queue: &mut WorkQueue<H, Fr>,
    );

    fn compute_quotient_contribution(&self, alpha_base: Fr, transcript: &Transcript<H>) -> Fr;
}

impl<H: BarretenHasher, Fr: FieldParams> Clone for Box<dyn ProverRandomWidget<H, Fr>> {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

pub trait BoxedCloneProverRandomWidget {
    fn boxed_clone(&self) -> Self;
}

impl<H, Fr> BoxedCloneProverRandomWidget for dyn ProverRandomWidget<H, Fr> {
    fn boxed_clone(&self) -> Box<dyn ProverRandomWidget<H, Fr>> {
        Box::new(self.clone())
    }
}

impl<H: BarretenHasher, Fr: FieldParams> PartialEq for dyn ProverRandomWidget<H, Fr> {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl<H: BarretenHasher, Fr: FieldParams, T: ProverRandomWidget<H, Fr>> PartialEq<T>
    for dyn ProverRandomWidget<H, Fr>
{
    fn eq(&self, other: &T) -> bool {
        self.key() == other.key()
    }
}

impl<H: BarretenHasher, Fr: FieldParams> Eq for dyn ProverRandomWidget<H, Fr> {}
