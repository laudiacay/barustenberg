use typenum::U1;

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
    NIndependentRelations: typenum::Unsigned,
> {
    base_getter: Box<dyn BaseGetter<H, FieldParams, S, NIndependentRelations>>,
    phantom: PhantomData<(FieldParams, Getters, PolyContainer)>,
}

impl<H: BarretenHasher, F, Getters, S: Settings<H>, PolyContainer>
    ArithmeticKernel<H, F, S, Getters, PolyContainer, U1>
where
    F: FieldParams,
    Getters: BaseGetter<H, F, S, U1>,
{
    // TODO see all these U1s they should be a named variable but they are not :( inherent associate type problem
    pub const QUOTIENT_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA as u8;
    pub const UPDATE_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA as u8;

    pub fn get_required_polynomial_ids() -> &'static HashSet<PolynomialIndex> {
        // ...
        todo!("ArithmeticKernel::get_required_polynomial_ids")
    }

    pub fn compute_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, U1>,
        linear_terms: &mut CoefficientArray<F>,
        i: usize,
    ) {
        // ...
    }

    pub fn compute_non_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, U1>,
        field: &mut F,
        i: usize,
    ) {
        // ...
    }

    pub fn sum_linear_terms(
        polynomials: &PolyContainer,
        challenges: &ChallengeArray<F, U1>,
        linear_terms: &mut CoefficientArray<F>,
        i: usize,
    ) -> F {
        // ...
        todo!("ArithmeticKernel::sum_linear_terms")
    }

    pub fn update_kate_opening_scalars(
        linear_terms: &CoefficientArray<F>,
        scalars: &mut HashMap<String, F>,
        challenges: &ChallengeArray<F, U1>,
    ) {
        // ...
    }
}

pub type ProverArithmeticWidget<
    F: FieldParams,
    H: BarretenHasher,
    S: Settings<H>,
    NWidgetRelations: typenum::Unsigned,
    PolyContainer,
    Getters: BaseGetter<H, F, S, NWidgetRelations>,
> = TransitionWidget<
    H,
    F,
    S,
    PolyContainer,
    Getters,
    NWidgetRelations,
    ArithmeticKernel<H, Fr, S, Getters, PolyContainer, NWidgetRelations>,
>;

pub type VerifierArithmeticWidget<
    H: BarretenHasher,
    F: FieldParams,
    Group,
    NWidgetRelations: typenum::Unsigned,
    Getters: BaseGetter<H, F, S, NWidgetRelations>,
    PC,
    S: Settings<H>,
> = GenericVerifierWidget<
    F,
    H,
    PC,
    Getters,
    NWidgetRelations,
    S,
    ArithmeticKernel<H, Fr, S, Getters, PC, NWidgetRelations>,
>;
