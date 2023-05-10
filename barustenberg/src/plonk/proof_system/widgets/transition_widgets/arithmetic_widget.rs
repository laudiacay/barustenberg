use crate::{
    ecc::{curves::bn254::fr::Fr, fields::field::Field},
    plonk::proof_system::{
        types::{polynomial_manifest::PolynomialIndex, prover_settings::Settings},
        widgets::transition_widgets::transition_widget::containers::{
            ChallengeArray, CoefficientArray,
        },
    },
    transcript::{BarretenHasher, Transcript},
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

pub struct ArithmeticKernel<
    H: BarretenHasher,
    Field,
    Getters,
    PolyContainer,
    const NUM_WIDGET_RELATIONS: usize,
> {
    base_getter: dyn BaseGetter<Field, Transcript<H>, dyn Settings<H>, NUM_WIDGET_RELATIONS>,
    phantom: PhantomData<(Field, Getters, PolyContainer)>,
}

impl<H: BarretenHasher, F, Getters, PolyContainer> ArithmeticKernel<H, F, Getters, PolyContainer, 1>
where
    F: Field,
    Getters: BaseGetter<F, PolyContainer>,
{
    pub const QUOTIENT_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA;
    pub const UPDATE_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA;

    pub fn get_required_polynomial_ids() -> &'static HashSet<PolynomialIndex> {
        // ...
    }

    pub fn compute_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, Self::NUM_INDEPENDENT_RELATIONS>,
        linear_terms: &mut CoefficientArray<F>,
        i: usize,
    ) {
        // ...
    }

    pub fn compute_non_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, Self::NUM_INDEPENDENT_RELATIONS>,
        field: &mut F,
        i: usize,
    ) {
        // ...
    }

    pub fn sum_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, Self::NUM_INDEPENDENT_RELATIONS>,
        linear_terms: &mut CoefficientArray<F>,
        i: usize,
    ) -> F {
        // ...
    }

    pub fn update_kate_opening_scalars(
        linear_terms: &CoefficientArray<F>,
        scalars: &mut HashMap<String, F>,
        challenges: &ChallengeArray<F, Self::NUM_INDEPENDENT_RELATIONS>,
    ) {
        // ...
    }
}

pub type ProverArithmeticWidget<Settings> = TransitionWidget<Fr, Settings, ArithmeticKernel>;

pub type VerifierArithmeticWidget<F, Group, Transcript, Settings> =
    GenericVerifierWidget<F, Transcript, Settings, ArithmeticKernel>;
