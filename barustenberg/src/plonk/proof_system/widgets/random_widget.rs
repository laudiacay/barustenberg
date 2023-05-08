use crate::ecc::curves::bn254::fr::Fr;
use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::proof_system::work_queue::WorkQueue;
use crate::transcript::Transcript;

pub trait ProverRandomWidgetTrait<T> {
    fn compute_round_commitments(
        &mut self,
        transcript: &mut Transcript<T>,
        size: usize,
        work_queue: &mut WorkQueue<Fr>,
    );

    fn compute_quotient_contribution(&mut self, alpha_base: Fr, transcript: &Transcript<T>) -> Fr;
}

pub struct ProverRandomWidget<'a, T: ProverRandomWidgetTrait<T>> {
    key: &'a mut ProvingKey<Fr>,
    widget: T,
}

impl<'a, T: ProverRandomWidgetTrait<T>> ProverRandomWidget<'a, T> {
    pub fn new(key: &'a mut ProvingKey<Fr>, widget: T) -> Self {
        ProverRandomWidget { key, widget }
    }

    pub fn from_other(other: &'a mut Self) -> Self {
        ProverRandomWidget {
            key: other.key,
            widget: other.widget.clone(),
        }
    }
}
