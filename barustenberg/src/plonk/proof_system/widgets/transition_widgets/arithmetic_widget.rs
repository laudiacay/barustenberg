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
    containers::PolyContainer, BaseGetter, EvaluationGetter, GenericVerifierWidget, KernelBase,
    TransitionWidget, CHALLENGE_BIT_ALPHA,
};

pub(crate) struct ArithmeticKernel<
    H: BarretenHasher,
    F: Field,
    S: Settings<H>,
    Get: BaseGetter<H, F, S, U1>,
    PolyContainer,
> {
    _marker: PhantomData<(H, F, S, Get, PolyContainer)>,
}

impl<H: BarretenHasher, F, Get: BaseGetter<H, F, S, U1>, S: Settings<H>, PolyContainer>
    ArithmeticKernel<H, F, S, Get, PolyContainer>
where
    F: Field,
{
    // TODO see all these U1s they should be a named variable but they are not :( inherent associate type problem
    pub(crate) const QUOTIENT_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA as u8;
    pub(crate) const UPDATE_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA as u8;

    pub(crate) fn get_required_polynomial_ids() -> &'static HashSet<PolynomialIndex> {
        // ...
        todo!("ArithmeticKernel::get_required_polynomial_ids")
    }

    pub(crate) fn compute_linear_terms(
        _polynomials: &PolyContainer,
        _challenges: &ChallengeArray<F, U1>,
        _linear_terms: &mut CoefficientArray<F>,
        _i: usize,
    ) {
        // ...
    }

    pub(crate) fn compute_non_linear_terms(
        _polynomials: &PolyContainer,
        _challenges: &ChallengeArray<F, U1>,
        _field: &mut F,
        _i: usize,
    ) {
        // ...
    }

    pub(crate) fn sum_linear_terms(
        _polynomials: &PolyContainer,
        _challenges: &ChallengeArray<F, U1>,
        _linear_terms: &mut CoefficientArray<F>,
        _i: usize,
    ) -> F {
        // ...
        todo!("ArithmeticKernel::sum_linear_terms")
    }

    /// Compute the scaled values of openings
    ///
    /// # Arguments
    /// - `linear_terms` - The original computed linear terms of the product and wires
    /// - `scalars` - A map where we put the values
    /// - `challenges` - Challenges where we get the alpha
    pub(crate) fn update_kate_opening_scalars(
        linear_terms: &CoefficientArray<F>,
        scalars: &mut HashMap<String, F>,
        challenges: &ChallengeArray<F, U1>,
    ) {
        let alpha: F = challenges.alpha_powers[0];
        scalars.insert(
            "Q_M".to_string(),
            *scalars.get("Q_M").unwrap() + linear_terms[0.into()] * alpha,
        );
        scalars.insert(
            "Q_1".to_string(),
            *scalars.get("Q_1").unwrap() + linear_terms[1.into()] * alpha,
        );
        scalars.insert(
            "Q_2".to_string(),
            *scalars.get("Q_2").unwrap() + linear_terms[2.into()] * alpha,
        );
        scalars.insert(
            "Q_3".to_string(),
            *scalars.get("Q_3").unwrap() + linear_terms[3.into()] * alpha,
        );
        scalars.insert("Q_C".to_string(), *scalars.get("Q_C").unwrap() + alpha);
    }
}

impl<
        H: BarretenHasher,
        F: Field,
        S: Settings<H>,
        Get: BaseGetter<H, F, S, U1>,
        PC: PolyContainer<F>,
    > KernelBase<H, S, F, PC, Get, U1> for ArithmeticKernel<H, F, S, Get, PC>
{
    fn get_required_polynomial_ids() -> HashSet<PolynomialIndex> {
        // inline static std::set<PolynomialIndex> const& get_required_polynomial_ids()
        // {
        //     static const std::set<PolynomialIndex> required_polynomial_ids = { PolynomialIndex::Q_1, PolynomialIndex::Q_2,
        //                                                                        PolynomialIndex::Q_3, PolynomialIndex::Q_M,
        //                                                                        PolynomialIndex::Q_C, PolynomialIndex::W_1,
        //                                                                        PolynomialIndex::W_2, PolynomialIndex::W_3 };
        //     return required_polynomial_ids;
        //}

        todo!()
    }

    fn quotient_required_challenges() -> u8 {
        todo!()
    }

    fn update_required_challenges() -> u8 {
        todo!()
    }

    fn compute_linear_terms(
        _polynomials: &impl PolyContainer<F>,
        _challenges: &ChallengeArray<F, U1>,
        _linear_terms: &mut CoefficientArray<F>,
        _index: usize,
    ) {
        /*
                inline static void compute_linear_terms(PolyContainer& polynomials,
                                                const challenge_array&,
                                                coefficient_array& linear_terms,
                                                const size_t i = 0)
        {
            const Field& w_1 =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::W_1>(polynomials, i);
            const Field& w_2 =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::W_2>(polynomials, i);
            const Field& w_3 =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::W_3>(polynomials, i);

            linear_terms[0] = w_1 * w_2;
            linear_terms[1] = w_1;
            linear_terms[2] = w_2;
            linear_terms[3] = w_3;
        }
             */
        todo!()
    }

    fn sum_linear_terms(
        _polynomials: &impl PolyContainer<F>,
        _challenges: &ChallengeArray<F, U1>,
        _linear_terms: &CoefficientArray<F>,
        _index: usize,
    ) -> F {
        /*
                /**
         * @brief Scale and sum the linear terms for the final equation.
         *
         * @details Multiplies the linear terms by selector values and scale the whole sum by alpha before returning
         *
         * @param polynomials Container with polynomials or their simulation
         * @param challenges A structure with various challenges
         * @param linear_terms Precomuputed linear terms to be scaled and summed
         * @param i The index at which selector/witness values are sampled
         * @return Field Scaled sum of values
         */
        inline static Field sum_linear_terms(PolyContainer& polynomials,
                                             const challenge_array& challenges,
                                             coefficient_array& linear_terms,
                                             const size_t i = 0)
        {
            const Field& alpha = challenges.alpha_powers[0];
            const Field& q_1 =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::Q_1>(polynomials, i);
            const Field& q_2 =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::Q_2>(polynomials, i);
            const Field& q_3 =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::Q_3>(polynomials, i);
            const Field& q_m =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::Q_M>(polynomials, i);
            const Field& q_c =
                Getters::template get_value<EvaluationType::NON_SHIFTED, PolynomialIndex::Q_C>(polynomials, i);

            Field result = linear_terms[0] * q_m;
            result += (linear_terms[1] * q_1);
            result += (linear_terms[2] * q_2);
            result += (linear_terms[3] * q_3);
            result += q_c;
            result *= alpha;
            return result;
        }
             */
        todo!()
    }

    /// Not being used in arithmetic_widget because there are none
    fn compute_non_linear_terms(
        _polynomials: &impl PolyContainer<F>,
        _challenges: &ChallengeArray<F, U1>,
        _quotient_term: &mut F,
        _index: usize,
    ) {
        unimplemented!(
            "ArithmeticKernel::compute_non_linear_terms- there are no non-linear terms..."
        )
    }
}

pub(crate) type ProverArithmeticWidget<'a, F, G1Affine, H, S, PolyContainer, Getters> =
    TransitionWidget<
        'a,
        H,
        F,
        G1Affine,
        S,
        PolyContainer,
        Getters,
        U1,
        ArithmeticKernel<H, F, S, Getters, PolyContainer>,
    >;

pub(crate) struct VerifierArithmeticWidget<
    H: BarretenHasher,
    F: Field,
    //Group,
    NWidgetRelations: generic_array::ArrayLength<F>,
    Getters: BaseGetter<H, F, S, NWidgetRelations> + EvaluationGetter<H, F, S, NWidgetRelations>,
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
        'a,
        H: BarretenHasher,
        S: Settings<H>,
        F: Field,
        PC: PolyContainer<F>,
        Get: BaseGetter<H, F, S, U1> + EvaluationGetter<H, F, S, U1>,
    > GenericVerifierWidget<'a, F, H, PC, Get, U1, S, ArithmeticKernel<H, F, S, Get, PC>>
    for VerifierArithmeticWidget<H, F, U1, Get, PC, S>
{
}
