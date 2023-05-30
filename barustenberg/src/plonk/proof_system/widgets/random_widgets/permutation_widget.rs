use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::plonk::proof_system::widgets::random_widgets::random_widget::ProverRandomWidget;
use crate::proof_system::work_queue::WorkQueue;
use crate::transcript::{BarretenHasher, Transcript, TranscriptKey};
use std::marker::PhantomData;
use std::sync::Arc;

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

pub(crate) struct VerifierPermutationWidget<
    H: BarretenHasher,
    F: Field,
    Group,
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
> {
    transcript: Transcript<H>,
    phantom: PhantomData<(F, Group)>,
}

impl<H, F, Group, const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize>
    VerifierPermutationWidget<H, F, Group, NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL>
where
    H: BarretenHasher,
    F: Field,
{
    pub(crate) fn new() -> Self {
        Self {
            transcript: Transcript::<H>::default(),
            phantom: PhantomData,
        }
    }

    pub(crate) fn compute_quotient_evaluation_contribution<'a>(
        _key: &Arc<TranscriptKey<'a>>,
        _alpha_base: F,
        _transcript: &Transcript<H>,
        _quotient_numerator_eval: &mut F,
        _idpolys: bool,
    ) -> F {
        todo!("VerifierPermutationWidget::compute_quotient_evaluation_contribution")
        // ...
    }

    pub(crate) fn append_scalar_multiplication_inputs<'a>(
        _key: &Arc<TranscriptKey<'a>>,
        _alpha_base: F,
        _transcript: &Transcript<H>,
    ) -> F {
        // ...
        todo!("VerifierPermutationWidget::append_scalar_multiplication_inputs")
    }
}

pub(crate) struct ProverPermutationWidget<
    'a,
    Fr: Field + FftField,
    Hash: BarretenHasher,
    G1Affine: AffineRepr,
    const PROGRAM_WIDTH: usize,
    const IDPOLYS: bool,
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
> {
    prover_random_widget: ProverRandomWidget<'a, Hash, Fr, G1Affine>,
}

impl<
        'a,
        Fr: Field + FftField,
        G1Affine: AffineRepr,
        Hash: BarretenHasher,
        const PROGRAM_WIDTH: usize,
        const IDPOLYS: bool,
        const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
    >
    ProverPermutationWidget<
        'a,
        Fr,
        Hash,
        G1Affine,
        PROGRAM_WIDTH,
        IDPOLYS,
        NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL,
    >
{
    pub(crate) fn new(proving_key: Arc<ProvingKey<'a, Fr, G1Affine>>) -> Self {
        Self {
            prover_random_widget: ProverRandomWidget::new(&proving_key),
        }
    }

    pub(crate) fn compute_round_commitments(
        &mut self,
        _transcript: &mut Transcript<Hash>,
        _round_number: usize,
        _queue: &mut WorkQueue<'a, Hash, Fr, G1Affine>,
    ) {
        // ...
    }

    pub(crate) fn compute_quotient_contribution(
        &self,
        _alpha_base: Fr,
        _transcript: &Transcript<Hash>,
    ) -> Fr {
        // ...
        todo!("ProverPermutationWidget::compute_quotient_contribution")
    }
}
