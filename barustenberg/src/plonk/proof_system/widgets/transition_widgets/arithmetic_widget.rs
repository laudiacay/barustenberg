use ark_ec::AffineRepr;
use ark_ff::Field;
use typenum::U1;

use crate::{
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
    F: Field,
    S: Settings<H>,
    Getters,
    PolyContainer,
    NIndependentRelations: typenum::Unsigned,
> {
    _marker: PhantomData<(H, F, S, Getters, PolyContainer, NIndependentRelations)>,
}

impl<H: BarretenHasher, F, Getters, S: Settings<H>, PolyContainer>
    ArithmeticKernel<H, F, S, Getters, PolyContainer, U1>
where
    F: Field,
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
    F: Field,
    G1Affine: AffineRepr,
    H: BarretenHasher,
    S: Settings<H>,
    NWidgetRelations: generic_array::ArrayLength<F>,
    PolyContainer,
    Getters: BaseGetter<H, F, S, NWidgetRelations>,
> = TransitionWidget<
    H,
    F,
    G1Affine,
    S,
    PolyContainer,
    Getters,
    NWidgetRelations,
    ArithmeticKernel<H, F, S, Getters, PolyContainer, NWidgetRelations>,
>;

pub struct VerifierArithmeticWidget<
    H: BarretenHasher,
    F: Field,
    //Group,
    NWidgetRelations: generic_array::ArrayLength<F>,
    Getters: BaseGetter<H, F, S, NWidgetRelations>,
    PC,
    S: Settings<H>,
> {
    phantom: PhantomData<(
        H,
        F,
        //Group,
        NWidgetRelations,
        Getters,
        PC,
        S,
    )>,
}

impl<
        F: Field,
        H: BarretenHasher,
        PC,
        NWidgetRelations: generic_array::ArrayLength<F>,
        Getters: BaseGetter<H, F, S, NWidgetRelations>,
        S: Settings<H>,
    >
    GenericVerifierWidget<
        H,
        F,
        PC,
        Getters,
        NWidgetRelations,
        S,
        ArithmeticKernel<H, F, S, Getters, PC, NWidgetRelations>,
    > for VerifierArithmeticWidget<H, F, NWidgetRelations, Getters, PC, S>
{
}
