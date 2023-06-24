use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use ark_ec::AffineRepr;

use crate::{
    ecc::fieldext::FieldExt,
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::{polynomial_manifest::PolynomialIndex, prover_settings::Settings},
        verification_key::VerificationKey,
    },
    transcript::{BarretenHasher, Transcript},
};

use super::{
    containers::{ChallengeArray, CoefficientArray, PolyArray, PolyContainer, PolyPtrMap},
    getters::{BaseGetter, EvaluationGetter, FFTGetter, FFTGetterImpl},
};

pub(crate) trait KernelBase<
    H: BarretenHasher,
    S: Settings<H, F, G>,
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    G: AffineRepr,
    NumIndependentRelations: generic_array::ArrayLength<F>,
>
{
    fn get_required_polynomial_ids() -> HashSet<PolynomialIndex>;
    fn quotient_required_challenges() -> u8;
    fn update_required_challenges() -> u8;
    fn compute_linear_terms<
        PC: PolyContainer<F>,
        Get: BaseGetter<H, F, G, S, PC, NumIndependentRelations>,
    >(
        polynomials: &PC,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        linear_terms: &mut CoefficientArray<F>,
        index: Option<usize>,
    );
    fn sum_linear_terms<
        PC: PolyContainer<F>,
        Get: BaseGetter<H, F, G, S, PC, NumIndependentRelations>,
    >(
        polynomials: &PC,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        linear_terms: &CoefficientArray<F>,
        index: usize,
    ) -> F;
    fn compute_non_linear_terms<
        PC: PolyContainer<F>,
        Get: BaseGetter<H, F, G, S, PC, NumIndependentRelations>,
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

pub(crate) trait TransitionWidgetBase<
    'a,
    H: BarretenHasher,
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    G: AffineRepr,
>
{
    fn compute_quotient_contribution(
        &self,
        alpha_base: F,
        transcript: &Transcript<H>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> F;
}

pub(crate) struct TransitionWidget<
    'a,
    H: BarretenHasher,
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    G: AffineRepr,
    S: Settings<H, F, G>,
    NIndependentRelations,
    KB,
> where
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase<H, S, F, G, NIndependentRelations>,
{
    key: Rc<ProvingKey<'a, F, G>>,
    phantom: PhantomData<(H, F, S, NIndependentRelations, KB)>,
}
impl<
        'a,
        H: BarretenHasher,
        F: ark_ff::Field + ark_ff::FftField + FieldExt,
        G: AffineRepr,
        S: Settings<H, F, G>,
        NIndependentRelations: generic_array::ArrayLength<F>,
        KB,
    > TransitionWidgetBase<'a, H, F, G>
    for TransitionWidget<'a, H, F, G, S, NIndependentRelations, KB>
where
    KB: KernelBase<H, S, F, G, NIndependentRelations>,
{
    // other methods and trait implementations
    fn compute_quotient_contribution(
        &self,
        alpha_base: F,
        transcript: &Transcript<H>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> F {
        let required_polynomial_ids = KB::get_required_polynomial_ids();
        let polynomials = FFTGetterImpl::<H, F, G, S, NIndependentRelations>::get_polynomials(
            &self.key,
            &required_polynomial_ids,
        );

        let challenges = FFTGetterImpl::<H, F, G, S, NIndependentRelations>::get_challenges(
            transcript,
            alpha_base,
            KB::quotient_required_challenges(),
            rng,
        );

        let mut quotient_term;

        // TODO: hidden missing multithreading here
        for i in 0..self.key.large_domain.size {
            let mut linear_terms = CoefficientArray::default();
            KB::compute_linear_terms::<
                PolyPtrMap<F>,
                FFTGetterImpl<H, F, G, S, NIndependentRelations>,
            >(&polynomials, &challenges, &mut linear_terms, Some(i));
            let sum_of_linear_terms = KB::sum_linear_terms::<
                PolyPtrMap<F>,
                FFTGetterImpl<H, F, G, S, NIndependentRelations>,
            >(&polynomials, &challenges, &linear_terms, i);

            quotient_term = self.key.quotient_polynomial_parts
                [i >> self.key.small_domain.log2_size]
                .borrow()[i & (self.key.circuit_size - 1)];
            quotient_term += sum_of_linear_terms;
            KB::compute_non_linear_terms::<
                PolyPtrMap<F>,
                FFTGetterImpl<H, F, G, S, NIndependentRelations>,
            >(&polynomials, &challenges, &mut quotient_term, i);
        }

        FFTGetterImpl::<H, F, G, S, NIndependentRelations>::update_alpha(&challenges)
    }
}

pub(crate) trait GenericVerifierWidget<
    'a,
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    H: BarretenHasher,
    G: AffineRepr,
    Get: EvaluationGetter<H, F, G, S, NIndependentRelations>,
    NIndependentRelations,
    S: Settings<H, F, G>,
    KB,
> where
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase<H, S, F, G, NIndependentRelations>,
{
    fn compute_quotient_evaluation_contribution(
        key: &Arc<VerificationKey<'a, F>>,
        alpha_base: F,
        transcript: &Transcript<H>,
        quotient_numerator_eval: &mut F,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> F {
        let polynomial_evaluations =
            Get::get_polynomial_evaluations(&key.as_ref().polynomial_manifest, transcript);
        let challenges = Get::get_challenges(
            transcript,
            alpha_base,
            KB::quotient_required_challenges(),
            rng,
        );

        let mut linear_terms = CoefficientArray::default();
        KB::compute_linear_terms::<PolyArray<F>, Get>(
            &polynomial_evaluations,
            &challenges,
            &mut linear_terms,
            Some(0),
        );
        *quotient_numerator_eval += KB::sum_linear_terms::<PolyArray<F>, Get>(
            &polynomial_evaluations,
            &challenges,
            &linear_terms,
            0,
        );
        KB::compute_non_linear_terms::<PolyArray<F>, Get>(
            &polynomial_evaluations,
            &challenges,
            quotient_numerator_eval,
            0,
        );

        Get::update_alpha(&challenges)
    }

    fn append_scalar_multiplication_inputs(
        _key: &Arc<VerificationKey<'_, F>>,
        alpha_base: F,
        transcript: &Transcript<H>,
        _scalar_mult_inputs: &mut HashMap<String, F>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> F {
        let challenges = Get::get_challenges(
            transcript,
            alpha_base,
            KB::quotient_required_challenges() | KB::update_required_challenges(),
            rng,
        );

        Get::update_alpha(&challenges)
    }
}
