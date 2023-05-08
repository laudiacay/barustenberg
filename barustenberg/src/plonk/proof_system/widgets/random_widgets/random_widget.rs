use crate::{
    plonk::proof_system::proving_key::ProvingKey,
    proof_system::work_queue::WorkQueue,
    transcript::{HasherType, Transcript},
};
use ark_ff::Field;

pub trait ProverRandomWidget<H: HasherType, Fr: Field> {
    fn new(input_key: &ProvingKey<Fr>) -> Self;

    fn compute_round_commitments(
        &self,
        transcript: &mut Transcript<H>,
        size: usize,
        work_queue: &mut WorkQueue<Fr>,
    );

    fn compute_quotient_contribution(&self, alpha_base: Fr, transcript: &Transcript<H>) -> Fr;
}

impl<H: HasherType, Fr: Field> Clone for Box<dyn ProverRandomWidget<H, Fr>> {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

pub trait BoxedCloneProverRandomWidget {
    fn boxed_clone(&self) -> Self;
}

impl<H: HasherType, Fr: Field, T> BoxedCloneProverRandomWidget for T
where
    T: 'static + ProverRandomWidget<H, Fr> + Clone,
{
    fn boxed_clone(&self) -> Box<dyn ProverRandomWidget<H, Fr>> {
        Box::new(self.clone())
    }
}

impl<H: HasherType, Fr: Field> PartialEq for dyn ProverRandomWidget<H, Fr> {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl<H: HasherType, Fr: Field, T: ProverRandomWidget<H, Fr>> PartialEq<T>
    for dyn ProverRandomWidget<H, Fr>
{
    fn eq(&self, other: &T) -> bool {
        self.key() == other.key()
    }
}

impl<H: HasherType, Fr: Field> Eq for dyn ProverRandomWidget<H, Fr> {}
