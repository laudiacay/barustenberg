use ark_ff::Field;

use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::proof_system::work_queue::WorkQueue;
use crate::plonk::proof_system::widgets::random_widgets::random_widget::ProverRandomWidget;
use crate::transcript::{HasherType, Transcript};
use std::marker::PhantomData;
use std::sync::Arc;
use crate::plonk::proof_system::types::hasher::Hasher;
pub struct VerifierPermutationWidget<
    Field,
    Group,
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
> where
    Field: ark_ff::Field,
{
    transcript: Transcript<Hash>,
    phantom: PhantomData<(Field, Group)>,
}

impl<Field, Group, const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize>
    VerifierPermutationWidget<Field, Group, NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL>
where
    Field: Field,
{
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }

    pub fn compute_quotient_evaluation_contribution(
        key: &Arc<TS::Key>,
        alpha_base: Field,
        transcript: &TS,
        quotient_numerator_eval: &mut Field,
        idpolys: bool,
    ) -> Field {
        // ...
    }

    pub fn append_scalar_multiplication_inputs(
        key: &Arc<TS::Key>,
        alpha_base: Field,
        transcript: &TS,
    ) -> Field {
        // ...
    }
}

pub struct ProverPermutationWidget<
    Fr: Field,
    Hash: HasherType,
    const PROGRAM_WIDTH: usize,
    const IDPOLYS: bool,
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
> {
    prover_random_widget: ProverRandomWidget,
}

impl<
        Fr: Field,
        Hash: HasherType,
        const PROGRAM_WIDTH: usize,
        const IDPOLYS: bool,
        const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
    >
    ProverPermutationWidget<
        Fr: Field,
        Hash: HasherType,
        PROGRAM_WIDTH,
        IDPOLYS,
        NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL,
    >
{
    pub fn new(proving_key: Arc<ProvingKey<Fr>>) -> Self {
        Self {
            prover_random_widget: ProverRandomWidget::new(&proving_key),
        }
    }

    pub fn compute_round_commitments(
        &mut self,
        transcript: &mut Transcript<Hash>,
        round_number: usize,
        queue: &mut WorkQueue<Fr>,
    ) {
        // ...
    }

    pub fn compute_quotient_contribution(
        &self,
        alpha_base: Fr,
        transcript: &Transcript<Hash>,
    ) -> Fr {
        // ...
    }
}
