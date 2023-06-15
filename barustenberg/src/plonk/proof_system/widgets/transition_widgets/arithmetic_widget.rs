use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};
use typenum::U1;

use crate::{
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::{
            polynomial_manifest::{EvaluationType, PolynomialIndex},
            prover_settings::Settings,
        },
    },
    transcript::BarretenHasher,
};

use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::Arc,
};

use super::{
    containers::{ChallengeArray, CoefficientArray, PolyContainer, CHALLENGE_BIT_ALPHA},
    getters::BaseGetter,
    transition_widget::KernelBase,
};

pub(crate) struct ArithmeticKernel<H: BarretenHasher, F: Field, S: Settings<H>> {
    _marker: PhantomData<(H, F, S)>,
}

impl<H: BarretenHasher, F, S: Settings<H>> ArithmeticKernel<H, F, S>
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
}

impl<H: BarretenHasher, F: Field, S: Settings<H>> KernelBase<H, S, F, U1>
    for ArithmeticKernel<H, F, S>
{
    #[inline]
    fn get_required_polynomial_ids() -> HashSet<PolynomialIndex> {
        HashSet::from([
            PolynomialIndex::Q1,
            PolynomialIndex::Q2,
            PolynomialIndex::Q3,
            PolynomialIndex::QM,
            PolynomialIndex::QC,
            PolynomialIndex::W1,
            PolynomialIndex::W2,
            PolynomialIndex::W3,
        ])
    }

    fn quotient_required_challenges() -> u8 {
        todo!()
    }

    fn update_required_challenges() -> u8 {
        todo!()
    }

    #[inline]
    fn compute_linear_terms<PC: PolyContainer<F>, G: BaseGetter<H, F, S, PC, U1>>(
        polynomials: &PC,
        _challenges: &ChallengeArray<F, U1>,
        linear_terms: &mut CoefficientArray<F>,
        index: Option<usize>,
    ) {
        let index = index.unwrap_or_default();

        let w_1 = G::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::W1,
            Some(index),
        );
        let w_2 = G::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::W2,
            Some(index),
        );
        let w_3 = G::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::W3,
            Some(index),
        );
        linear_terms[0.into()] = w_1 * w_2;
        linear_terms[1.into()] = w_1;
        linear_terms[2.into()] = w_2;
        linear_terms[3.into()] = w_3;
    }

    fn sum_linear_terms<PC: PolyContainer<F>, G: BaseGetter<H, F, S, PC, U1>>(
        _polynomials: &PC,
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
    fn compute_non_linear_terms<PC: PolyContainer<F>, G: BaseGetter<H, F, S, PC, U1>>(
        _polynomials: &PC,
        _challenges: &ChallengeArray<F, U1>,
        _quotient_term: &mut F,
        _index: usize,
    ) {
        unimplemented!(
            "ArithmeticKernel::compute_non_linear_terms- there are no non-linear terms..."
        )
    }

    /// Compute the scaled values of openings
    ///
    /// # Arguments
    /// - `linear_terms` - The original computed linear terms of the product and wires
    /// - `scalars` - A map where we put the values
    /// - `challenges` - Challenges where we get the alpha
    fn update_kate_opening_scalars(
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

pub(crate) struct ProverArithmeticWidget<'a, Fr: Field + FftField, G1Affine: AffineRepr, H, S> {
    key: Arc<ProvingKey<'a, Fr, G1Affine>>,
    phantom: PhantomData<(H, S)>,
}
