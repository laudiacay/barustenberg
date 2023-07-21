use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Debug,
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::{
    plonk::proof_system::{
        proving_key::ProvingKey, types::polynomial_manifest::PolynomialIndex,
        verification_key::VerificationKey,
    },
    transcript::{BarretenHasher, Transcript},
};

use super::{
    containers::{ChallengeArray, CoefficientArray, PolyArray},
    getters::{BaseGetter, EvaluationGetter, FFTGetter, FFTGetterImpl},
};

pub(crate) trait KernelBase {
    type Hasher: BarretenHasher;
    type Field: Field + FftField;
    type Group: AffineRepr;
    type NumIndependentRelations: generic_array::ArrayLength<Self::Field>;
    fn get_required_polynomial_ids() -> HashSet<PolynomialIndex>;
    fn quotient_required_challenges() -> u8;
    fn update_required_challenges() -> u8;
    fn compute_linear_terms<Get: BaseGetter<Fr = Self::Field>>(
        polynomials: &Get::PC,
        challenges: &ChallengeArray<Self::Field, Self::NumIndependentRelations>,
        linear_terms: &mut CoefficientArray<Self::Field>,
        index: Option<usize>,
    );
    fn sum_linear_terms<Get: BaseGetter>(
        polynomials: &Get::PC,
        challenges: &ChallengeArray<Self::Field, Self::NumIndependentRelations>,
        linear_terms: &CoefficientArray<Self::Field>,
        index: usize,
    ) -> Self::Field;
    fn compute_non_linear_terms<Get: BaseGetter<Fr = Self::Field>>(
        polynomials: &Get::PC,
        challenges: &ChallengeArray<Self::Field, Self::NumIndependentRelations>,
        quotient_term: &mut Self::Field,
        index: usize,
    );
    fn update_kate_opening_scalars(
        linear_terms: &CoefficientArray<Self::Field>,
        scalars: &mut HashMap<String, Self::Field>,
        challenges: &ChallengeArray<Self::Field, Self::NumIndependentRelations>,
    );
}

pub(crate) trait TransitionWidgetBase: std::fmt::Debug {
    type Hasher: BarretenHasher;
    type Field: Field + FftField;

    fn compute_quotient_contribution(
        &self,
        alpha_base: Self::Field,
        transcript: &Transcript<Self::Hasher>,
        rng: &mut dyn rand::RngCore,
    ) -> Self::Field;
}

#[derive(Debug)]
pub(crate) struct TransitionWidget<
    H: BarretenHasher,
    F: Field + FftField,
    G: AffineRepr,
    NIndependentRelations,
    KB,
> where
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase,
{
    key: Rc<RefCell<ProvingKey<F>>>,
    phantom: PhantomData<(H, NIndependentRelations, KB, G)>,
}

impl<H: BarretenHasher, F: Field + FftField, G: AffineRepr, NIndependentRelations, KB>
    TransitionWidget<H, F, G, NIndependentRelations, KB>
where
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase<
        Field = F,
        Group = G,
        NumIndependentRelations = NIndependentRelations,
        Hasher = H,
    >,
{
    pub(crate) fn new(key: Rc<RefCell<ProvingKey<F>>>) -> Self {
        Self {
            key,
            phantom: PhantomData,
        }
    }
}

impl<
        H: BarretenHasher,
        F: Field + FftField,
        G: AffineRepr,
        NIndependentRelations: generic_array::ArrayLength<F> + Debug,
        KB: std::fmt::Debug,
    > TransitionWidgetBase for TransitionWidget<H, F, G, NIndependentRelations, KB>
where
    KB: KernelBase<
        Field = F,
        Group = G,
        NumIndependentRelations = NIndependentRelations,
        Hasher = H,
    >,
{
    type Hasher = H;
    type Field = F;

    // other methods and trait implementations
    fn compute_quotient_contribution(
        &self,
        alpha_base: F,
        transcript: &Transcript<H>,
        rng: &mut dyn rand::RngCore,
    ) -> F {
        let required_polynomial_ids = KB::get_required_polynomial_ids();
        let polynomials = FFTGetterImpl::<H, F, G, NIndependentRelations>::get_polynomials(
            &self.key.borrow(),
            &required_polynomial_ids,
        );

        let challenges = FFTGetterImpl::<H, F, G, NIndependentRelations>::get_challenges(
            transcript,
            alpha_base,
            KB::quotient_required_challenges(),
            rng,
        );

        let mut quotient_term;

        let borrowed_key = self.key.borrow();

        // TODO: hidden missing multithreading here
        for i in 0..borrowed_key.large_domain.size {
            let mut linear_terms = CoefficientArray::default();
            KB::compute_linear_terms::<FFTGetterImpl<H, F, G, NIndependentRelations>>(
                &polynomials,
                &challenges,
                &mut linear_terms,
                Some(i),
            );
            let sum_of_linear_terms = KB::sum_linear_terms::<
                FFTGetterImpl<H, F, G, NIndependentRelations>,
            >(&polynomials, &challenges, &linear_terms, i);

            quotient_term = borrowed_key.quotient_polynomial_parts
                [i >> borrowed_key.small_domain.log2_size]
                .borrow()[i & (borrowed_key.circuit_size - 1)];
            quotient_term += sum_of_linear_terms;
            KB::compute_non_linear_terms::<FFTGetterImpl<H, F, G, NIndependentRelations>>(
                &polynomials,
                &challenges,
                &mut quotient_term,
                i,
            );
        }

        FFTGetterImpl::<H, F, G, NIndependentRelations>::update_alpha(&challenges)
    }
}

pub(crate) trait GenericVerifierWidget<'a> {
    type Hasher: BarretenHasher;
    type Field: Field + FftField;
    type Group: AffineRepr;
    type NumIndependentRelations: generic_array::ArrayLength<Self::Field>;
    type Get: EvaluationGetter<
            Hasher = Self::Hasher,
            Fr = Self::Field,
            G1 = Self::Group,
            NWidgetRelations = Self::NumIndependentRelations,
        > + BaseGetter<
            Hasher = Self::Hasher,
            Fr = Self::Field,
            G1 = Self::Group,
            NWidgetRelations = Self::NumIndependentRelations,
            PC = PolyArray<Self::Field>,
        >;
    type KB: KernelBase<
        Field = Self::Field,
        Hasher = Self::Hasher,
        Group = Self::Group,
        NumIndependentRelations = Self::NumIndependentRelations,
    >;

    fn compute_quotient_evaluation_contribution(
        key: &Arc<VerificationKey<Self::Field>>,
        alpha_base: Self::Field,
        transcript: &Transcript<Self::Hasher>,
        quotient_numerator_eval: &mut Self::Field,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field {
        let polynomial_evaluations =
            Self::Get::get_polynomial_evaluations(&key.as_ref().polynomial_manifest, transcript);
        let challenges = Self::Get::get_challenges(
            transcript,
            alpha_base,
            Self::KB::quotient_required_challenges(),
            rng,
        );

        let mut linear_terms = CoefficientArray::default();
        Self::KB::compute_linear_terms::<Self::Get>(
            &polynomial_evaluations,
            &challenges,
            &mut linear_terms,
            Some(0),
        );
        *quotient_numerator_eval += Self::KB::sum_linear_terms::<Self::Get>(
            &polynomial_evaluations,
            &challenges,
            &linear_terms,
            0,
        );
        Self::KB::compute_non_linear_terms::<Self::Get>(
            &polynomial_evaluations,
            &challenges,
            quotient_numerator_eval,
            0,
        );

        Self::Get::update_alpha(&challenges)
    }

    fn append_scalar_multiplication_inputs(
        _key: &Arc<VerificationKey<Self::Field>>,
        alpha_base: Self::Field,
        transcript: &Transcript<Self::Hasher>,
        _scalar_mult_inputs: &mut HashMap<String, Self::Field>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field {
        let challenges = Self::Get::get_challenges(
            transcript,
            alpha_base,
            Self::KB::quotient_required_challenges() | Self::KB::update_required_challenges(),
            rng,
        );

        Self::Get::update_alpha(&challenges)
    }
}
