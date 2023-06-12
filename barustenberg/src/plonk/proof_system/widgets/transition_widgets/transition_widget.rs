use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use ark_bn254::G1Affine;
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::{
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::{polynomial_manifest::PolynomialIndex, prover_settings::Settings},
    },
    transcript::{BarretenHasher, Transcript, TranscriptKey},
};

use super::{
    containers::{ChallengeArray, CoefficientArray, PolyArray, PolyContainer, PolyPtrMap},
    getters::{BaseGetter, EvaluationGetter, EvaluationGetterImpl, FFTGetter, FFTGetterImpl},
};

pub(crate) trait KernelBase<
    H: BarretenHasher,
    S: Settings<H>,
    F: Field,
    NumIndependentRelations: generic_array::ArrayLength<F>,
>
{
    fn get_required_polynomial_ids() -> HashSet<PolynomialIndex>;
    fn quotient_required_challenges() -> u8;
    fn update_required_challenges() -> u8;
    fn compute_linear_terms<
        PC: PolyContainer<F>,
        G: BaseGetter<H, F, S, PC, NumIndependentRelations>,
    >(
        polynomials: &PC,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        linear_terms: &mut CoefficientArray<F>,
        index: Option<usize>,
    );
    fn sum_linear_terms<PC: PolyContainer<F>, G: BaseGetter<H, F, S, PC, NumIndependentRelations>>(
        polynomials: &PC,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        linear_terms: &CoefficientArray<F>,
        index: usize,
    ) -> F;
    fn compute_non_linear_terms<
        PC: PolyContainer<F>,
        G: BaseGetter<H, F, S, PC, NumIndependentRelations>,
    >(
        polynomials: &PC,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        quotient_term: &mut F,
        index: usize,
    );
    fn update_kate_opening_scalars(
        linear_terms: &CoefficientArray<F>,
        scalars: &mut HashMap<String, F>,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
    );
}

pub(crate) trait TransitionWidget<
    'a,
    H: BarretenHasher,
    F: Field + FftField,
    G1Affine: AffineRepr,
    S: Settings<H>,
    NIndependentRelations,
    KB,
> where
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase<H, S, F, NIndependentRelations>,
{
    /// make sure to inline me...
    fn get_key(&self) -> Arc<ProvingKey<'a, F, G1Affine>>;

    // other methods and trait implementations
    fn compute_quotient_contribution(
        &self,
        alpha_base: F,
        transcript: &Transcript<H, F, G1Affine>,
        rng: Arc<Mutex<dyn rand::RngCore + Send + Sync>>,
    ) -> F {
        let key = self.get_key();

        let required_polynomial_ids = KB::get_required_polynomial_ids();
        let polynomials =
            FFTGetterImpl::<H, F, G1Affine, S, NIndependentRelations>::get_polynomials(
                &key,
                &required_polynomial_ids,
            );

        let challenges =
            FFTGetterImpl::<H, F, G1Affine, S, NIndependentRelations>::get_challenges::<G1Affine>(
                transcript,
                alpha_base,
                KB::quotient_required_challenges(),
                rng,
            );

        let mut quotient_term;

        // TODO: hidden missing multithreading here
        for i in 0..key.large_domain.size {
            let mut linear_terms = CoefficientArray::default();
            KB::compute_linear_terms::<
                PolyPtrMap<'a, F>,
                FFTGetterImpl<H, F, G1Affine, S, NIndependentRelations>,
            >(&polynomials, &challenges, &mut linear_terms, Some(i));
            let sum_of_linear_terms = KB::sum_linear_terms::<
                PolyPtrMap<'a, F>,
                FFTGetterImpl<H, F, G1Affine, S, NIndependentRelations>,
            >(&polynomials, &challenges, &linear_terms, i);

            quotient_term = key.quotient_polynomial_parts[i >> key.small_domain.log2_size]
                [i & (key.circuit_size - 1)];
            quotient_term += sum_of_linear_terms;
            KB::compute_non_linear_terms::<
                PolyPtrMap<'a, F>,
                FFTGetterImpl<H, F, G1Affine, S, NIndependentRelations>,
            >(&polynomials, &challenges, &mut quotient_term, i);
        }

        FFTGetterImpl::<H, F, G1Affine, S, NIndependentRelations>::update_alpha(&challenges)
    }
}

pub(crate) trait GenericVerifierWidget<
    'a,
    F: Field + FftField,
    H: BarretenHasher,
    G: EvaluationGetter<H, F, S, NIndependentRelations>,
    NIndependentRelations,
    S: Settings<H>,
    KB,
> where
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase<H, S, F, NIndependentRelations>,
{
    fn compute_quotient_evaluation_contribution<G1Affine: AffineRepr>(
        key: &Arc<TranscriptKey<'a, F>>,
        alpha_base: F,
        transcript: &Transcript<H, F, G1Affine>,
        quotient_numerator_eval: &mut F,
        rng: Arc<Mutex<dyn rand::RngCore + Send + Sync>>,
    ) -> F {
        let polynomial_evaluations = G::get_polynomial_evaluations::<G1Affine>(
            &key.as_ref().polynomial_manifest,
            transcript,
        );
        let challenges = G::get_challenges::<G1Affine>(
            transcript,
            alpha_base,
            KB::quotient_required_challenges(),
            rng,
        );

        let mut linear_terms = CoefficientArray::default();
        KB::compute_linear_terms::<
            PolyArray<F>,
            EvaluationGetterImpl<H, F, S, NIndependentRelations>,
        >(
            &polynomial_evaluations,
            &challenges,
            &mut linear_terms,
            Some(0),
        );
        *quotient_numerator_eval +=
            KB::sum_linear_terms::<
                PolyArray<F>,
                EvaluationGetterImpl<H, F, S, NIndependentRelations>,
            >(&polynomial_evaluations, &challenges, &linear_terms, 0);
        KB::compute_non_linear_terms::<
            PolyArray<F>,
            EvaluationGetterImpl<H, F, S, NIndependentRelations>,
        >(
            &polynomial_evaluations,
            &challenges,
            quotient_numerator_eval,
            0,
        );

        G::update_alpha(&challenges)
    }

    fn append_scalar_multiplication_inputs(
        _key: &Arc<TranscriptKey<'_, F>>,
        alpha_base: F,
        transcript: &Transcript<H, F, G1Affine>,
        _scalar_mult_inputs: &mut HashMap<String, F>,
        rng: Arc<Mutex<dyn rand::RngCore + Send + Sync>>,
    ) -> F {
        let challenges = G::get_challenges::<G1Affine>(
            transcript,
            alpha_base,
            KB::quotient_required_challenges() | KB::update_required_challenges(),
            rng,
        );

        G::update_alpha(&challenges)
    }
}
