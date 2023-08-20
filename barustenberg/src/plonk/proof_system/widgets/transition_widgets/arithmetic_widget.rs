use ark_bn254::{Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};
use typenum::U1;

use crate::{
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::polynomial_manifest::{EvaluationType, PolynomialIndex},
    },
    transcript::BarretenHasher,
};

use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use super::{
    containers::{ChallengeArray, CoefficientArray, CHALLENGE_BIT_ALPHA},
    getters::BaseGetter,
    transition_widget::{KernelBase, TransitionWidget, TransitionWidgetBase, GenericVerifierWidget, GenericVerifierWidgetBase},
};

#[derive(Debug)]
pub(crate) struct ArithmeticKernel<H: BarretenHasher, F: Field + FftField, G: AffineRepr> {
    _marker: PhantomData<(H, F, G)>,
}

impl<H: BarretenHasher, F, G> ArithmeticKernel<H, F, G>
where
    F: Field + FftField,
    G: AffineRepr,
{
    // TODO see all these U1s they should be a named variable but they are not :( inherent associate type problem
    pub(crate) const QUOTIENT_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA as u8;
    pub(crate) const UPDATE_REQUIRED_CHALLENGES: u8 = CHALLENGE_BIT_ALPHA as u8;
}

impl<H: BarretenHasher, F: Field + FftField, G: AffineRepr> KernelBase
    for ArithmeticKernel<H, F, G>
{
    type Field = F;
    type Group = G;
    type Hasher = H;
    type NumIndependentRelations = U1;

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
        CHALLENGE_BIT_ALPHA.try_into().unwrap()
    }

    fn update_required_challenges() -> u8 {
        CHALLENGE_BIT_ALPHA.try_into().unwrap()
    }

    #[inline]
    fn compute_linear_terms<Get: BaseGetter<Fr = Self::Field>>(
        polynomials: &Get::PC,
        _challenges: &ChallengeArray<Self::Field, U1>,
        linear_terms: &mut CoefficientArray<Self::Field>,
        index: Option<usize>,
    ) {
        let index = index.unwrap_or_default();

        let w_1 = Get::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::W1,
            Some(index),
        );
        let w_2 = Get::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::W2,
            Some(index),
        );
        let w_3 = Get::get_value(
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

    /// Scales and sums the linear terms for the final equation.
    ///
    /// Multiplies the linear terms by selector values and scale the whole sum by alpha before returning.
    ///
    /// # Arguments
    ///
    /// * `polynomials` - Container with polynomials or their simulation.
    /// * `challenges` - A structure with various challenges.
    /// * `linear_terms` - Precomputed linear terms to be scaled and summed.
    /// * `i` - The index at which selector/witness values are sampled.
    ///
    /// # Returns
    ///
    /// * `FieldExt` - Scaled sum of values.
    fn sum_linear_terms<Get: BaseGetter<Fr = Self::Field>>(
        polynomials: &Get::PC,
        challenges: &ChallengeArray<Self::Field, U1>,
        linear_terms: &CoefficientArray<Self::Field>,
        index: usize,
    ) -> Self::Field {
        let alpha = challenges.alpha_powers[0];
        let q_1 = Get::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::Q1,
            Some(index),
        );
        let q_2 = Get::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::Q2,
            Some(index),
        );
        let q_3 = Get::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::Q3,
            Some(index),
        );
        let q_m = Get::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::QM,
            Some(index),
        );
        let q_c = Get::get_value(
            polynomials,
            EvaluationType::NonShifted,
            PolynomialIndex::QC,
            Some(index),
        );

        let mut result = linear_terms[0] * q_m;
        result += linear_terms[1] * q_1;
        result += linear_terms[2] * q_2;
        result += linear_terms[3] * q_3;
        result += q_c;
        result *= alpha;

        result
    }

    /// Not being used in arithmetic_widget because there are none
    fn compute_non_linear_terms<Get: BaseGetter<Fr = Self::Field>>(
        _polynomials: &Get::PC,
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
            *scalars.get("Q_M").unwrap() + linear_terms[0] * alpha,
        );
        scalars.insert(
            "Q_1".to_string(),
            *scalars.get("Q_1").unwrap() + linear_terms[1] * alpha,
        );
        scalars.insert(
            "Q_2".to_string(),
            *scalars.get("Q_2").unwrap() + linear_terms[2] * alpha,
        );
        scalars.insert(
            "Q_3".to_string(),
            *scalars.get("Q_3").unwrap() + linear_terms[3] * alpha,
        );
        scalars.insert("Q_C".to_string(), *scalars.get("Q_C").unwrap() + alpha);
    }
}

#[derive(Debug)]
pub(crate) struct ProverArithmeticWidget<H: BarretenHasher>(
    TransitionWidget<H, Fr, G1Affine, U1, ArithmeticKernel<H, Fr, G1Affine>>,
);

impl<H: BarretenHasher> TransitionWidgetBase for ProverArithmeticWidget<H> {
    type Hasher = H;
    type Field = Fr;

    fn compute_quotient_contribution(
        &self,
        alpha_base: Self::Field,
        transcript: &crate::transcript::Transcript<Self::Hasher>,
        rng: &mut dyn rand::RngCore,
    ) -> Self::Field {
        self.0
            .compute_quotient_contribution(alpha_base, transcript, rng)
    }
}

impl<H: BarretenHasher> ProverArithmeticWidget<H> {
    pub(crate) fn new(key: Arc<RwLock<ProvingKey<Fr>>>) -> Self {
        Self(TransitionWidget::new(key))
    }
}

#[derive(Debug)]
pub(crate) struct VerifierArithmeticWidget<H: BarretenHasher, F: Field + FftField, G: AffineRepr> {
    widget: GenericVerifierWidget<H, F, G, ArithmeticKernel<H, F, G>>,
}

impl <
    'a,
    H: BarretenHasher,
    F: Field + FftField,
    G: AffineRepr,
    > GenericVerifierWidgetBase<'a> for VerifierArithmeticWidget<H, F, G> {
    type Hasher = H;
    type Field = F;
    type Group = G;
    type NumIndependentRelations = typenum::U1;
    type KB = ArithmeticKernel<H, F, G>;
}