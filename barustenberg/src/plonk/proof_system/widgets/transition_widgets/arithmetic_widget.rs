use crate::{
    plonk::proof_system::{
        types::{polynomial_manifest::PolynomialIndex, prover_settings::SettingsBase},
        widgets::transition_widgets::transition_widget::containers::{
            ChallengeArray, CoefficientArray,
        },
    },
    transcript::Transcript,
};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use super::transition_widget::{
    BaseGetter, GenericVerifierWidget, TransitionWidget, CHALLENGE_BIT_ALPHA,
};

pub trait Getters<Field, PolyContainer> {
    // ... Implement the required getter methods
}

pub struct ArithmeticKernel<Field, Getters, PolyContainer, const NUM_WIDGET_RELATIONS: usize> {
    base_getter: dyn BaseGetter<
        Field,
        Transcript<dyn SettingsBase::Hash>,
        SettingsBase,
        NUM_WIDGET_RELATIONS,
    >,
    phantom: PhantomData<(Field, Getters, PolyContainer)>,
}

impl<Field, Getters, PolyContainer> ArithmeticKernel<Field, Getters, PolyContainer>
where
    Field: ark_ff::Field,
    Getters: BaseGetter<Field, PolyContainer>,
{
    pub const NUM_INDEPENDENT_RELATIONS: usize = 1;
    pub const QUOTIENT_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA;
    pub const UPDATE_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA;

    pub fn get_required_polynomial_ids() -> &'static HashSet<PolynomialIndex> {
        // ...
    }

    pub fn compute_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<Field, Self::NUM_INDEPENDENT_RELATIONS>,
        linear_terms: &mut CoefficientArray<Field>,
        i: usize,
    ) {
        // ...
    }

    pub fn compute_non_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<Field, Self::NUM_INDEPENDENT_RELATIONS>,
        field: &mut Field,
        i: usize,
    ) {
        // ...
    }

    pub fn sum_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<Field, Self::NUM_INDEPENDENT_RELATIONS>,
        linear_terms: &mut CoefficientArray<Field>,
        i: usize,
    ) -> Field {
        // ...
    }

    pub fn update_kate_opening_scalars(
        linear_terms: &CoefficientArray<Field>,
        scalars: &mut HashMap<String, Field>,
        challenges: &ChallengeArray<Field, Self::NUM_INDEPENDENT_RELATIONS>,
    ) {
        // ...
    }
}

pub type ProverArithmeticWidget<Settings> =
    TransitionWidget<ark_bn254::Fr, Settings, ArithmeticKernel>;

pub type VerifierArithmeticWidget<Field, Group, Transcript, Settings> =
    GenericVerifierWidget<Field, Transcript, Settings, ArithmeticKernel>;
