use crate::{
    ecc::{curves::bn254::fr::Fr, fields::field::FieldParams},
    plonk::proof_system::{
        types::{polynomial_manifest::PolynomialIndex, prover_settings::Settings},
        widgets::transition_widgets::transition_widget::containers::{
            ChallengeArray, CoefficientArray,
        },
    },
    transcript::BarretenHasher,
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
    FieldParams,
    S: Settings<H>,
    Getters,
    PolyContainer,
    const NUM_WIDGET_RELATIONS: usize,
> {
    base_getter: dyn BaseGetter<H, FieldParams, dyn Settings<H>, NUM_WIDGET_RELATIONS>,
    phantom: PhantomData<(FieldParams, Getters, PolyContainer)>,
}

impl<H: BarretenHasher, F, Getters, S: Settings<H>, PolyContainer>
    ArithmeticKernel<H, F, S, Getters, PolyContainer, 1>
where
    F: FieldParams,
    Getters: BaseGetter<H, F, S, 1>,
{
    pub const QUOTIENT_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA;
    pub const UPDATE_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA;

    pub fn get_required_polynomial_ids() -> &'static HashSet<PolynomialIndex> {
        // ...
    }

    pub fn compute_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, { Self::NUM_INDEPENDENT_RELATIONS }>,
        linear_terms: &mut CoefficientArray<F>,
        i: usize,
    ) {
        // ...
    }

    pub fn compute_non_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, { Self::NUM_INDEPENDENT_RELATIONS }>,
        field: &mut F,
        i: usize,
    ) {
        // ...
    }

    pub fn sum_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, { Self::NUM_INDEPENDENT_RELATIONS }>,
        linear_terms: &mut CoefficientArray<F>,
        i: usize,
    ) -> F {
        // ...
    }

    pub fn update_kate_opening_scalars(
        linear_terms: &CoefficientArray<F>,
        scalars: &mut HashMap<String, F>,
        challenges: &ChallengeArray<F, { Self::NUM_INDEPENDENT_RELATIONS }>,
    ) {
        // ...
    }
}

pub type ProverArithmeticWidget<
    F: FieldParams,
    H: BarretenHasher,
    S: Settings<H>,
    const NUM_WIDGET_RELATIONS: usize,
    PolyContainer,
    Getters: BaseGetter<H, F, S, NUM_WIDGET_RELATIONS>,
> = TransitionWidget<
    H,
    Fr,
    S,
    PolyContainer,
    Getters,
    NUM_WIDGET_RELATIONS,
    ArithmeticKernel<H, Fr, S, Getters, PolyContainer, NUM_WIDGET_RELATIONS>,
>;

pub type VerifierArithmeticWidget<
    H: BarretenHasher,
    F: FieldParams,
    Group,
    const NUM_WIDGET_RELATIONS: usize,
    Getters: BaseGetter<H, F, S, NUM_WIDGET_RELATIONS>,
    PolyContainer,
    S: Settings<H>,
> = GenericVerifierWidget<
    F,
    H,
    S,
    ArithmeticKernel<H, Fr, S, Getters, PolyContainer, NUM_WIDGET_RELATIONS>,
>;
