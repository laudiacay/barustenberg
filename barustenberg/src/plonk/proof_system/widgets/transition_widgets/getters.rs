// Getters are various structs that are used to retrieve/query various objects needed during the proof.
//
// You can query:
// - Challenges
// - Polynomial evaluations
// - Polynomials in monomial form
// - Polynomials in Lagrange form

use std::{collections::HashSet, marker::PhantomData};

use typenum::Unsigned;

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, Zero};

use crate::{
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::{
            polynomial_manifest::{EvaluationType, PolynomialIndex},
            PolynomialManifest,
        },
    },
    transcript::{BarretenHasher, Transcript},
};

use super::containers::{
    ChallengeArray, ChallengeIndex, PolyArray, PolyContainer, PolyPtrMap, CHALLENGE_BIT_ALPHA,
    CHALLENGE_BIT_BETA, CHALLENGE_BIT_ETA, CHALLENGE_BIT_GAMMA, CHALLENGE_BIT_ZETA,
};

/// Implements loading challenges from the transcript and computing powers of α, which are later used in widgets.
///
/// # Type Parameters
/// - `FieldExt`: Base FieldExt
/// - `Transcript`: Transcript struct
/// - `Settings`: Configuration
/// - `NUM_WIDGET_RELATIONS`: How many powers of α are needed
pub(crate) trait BaseGetter {
    type Hasher: BarretenHasher;
    type Fr: Field + FftField;
    type G1: AffineRepr;
    type PC: PolyContainer;
    type NWidgetRelations: generic_array::ArrayLength<Self::Fr>;

    /// Create a challenge array from transcript.
    /// Loads alpha, beta, gamma, eta, zeta, and nu and calculates powers of alpha.
    ///
    /// # Arguments
    /// - `transcript`: Transcript to get challenges from
    /// - `alpha_base`: α to some power (depends on previously used widgets)
    /// - `required_challenges`: Challenge bitmask, which shows when the function should fail
    ///
    /// # Returns
    /// A structure with an array of challenge values and powers of α
    fn get_challenges(
        transcript: &Transcript<Self::Hasher>,
        alpha_base: Self::Fr,
        required_challenges: u8,
        rng: &mut dyn rand::RngCore,
    ) -> ChallengeArray<Self::Fr, Self::NWidgetRelations> {
        let mut result: ChallengeArray<Self::Fr, _> = ChallengeArray::default();
        let mut add_challenge = |label: &str, tag: usize, required: bool, index: usize| {
            assert!(!required || transcript.has_challenge(label));
            if transcript.has_challenge(label) {
                assert!(index < transcript.get_num_challenges(label));
                result.elements[tag] = transcript.get_challenge_field_element(label, Some(index));
            } else {
                let mut random_bytes = vec![0u8; std::mem::size_of::<Self::Fr>()];
                // TODO should you really have an unwrap here?
                rng.fill_bytes(&mut random_bytes);
                result.elements[tag] = Self::Fr::from_random_bytes(random_bytes.as_ref())
                    .expect("random deserialization didn't work");
            }
        };
        add_challenge(
            "alpha",
            ChallengeIndex::Alpha as usize,
            required_challenges & CHALLENGE_BIT_ALPHA as u8 != 0,
            0,
        );
        add_challenge(
            "beta",
            ChallengeIndex::Beta as usize,
            required_challenges & CHALLENGE_BIT_BETA as u8 != 0,
            0,
        );
        add_challenge(
            "beta",
            ChallengeIndex::Gamma as usize,
            required_challenges & CHALLENGE_BIT_GAMMA as u8 != 0,
            1,
        );
        add_challenge(
            "eta",
            ChallengeIndex::Eta as usize,
            required_challenges & CHALLENGE_BIT_ETA as u8 != 0,
            0,
        );
        add_challenge(
            "z",
            ChallengeIndex::Zeta as usize,
            required_challenges & CHALLENGE_BIT_ZETA as u8 != 0,
            0,
        );
        result.alpha_powers[0] = alpha_base;
        for i in 1..Self::NWidgetRelations::to_usize() {
            result.alpha_powers[i] =
                result.alpha_powers[i - 1] * result.elements[ChallengeIndex::Alpha as usize];
        }
        result
    }

    fn update_alpha(challenges: &ChallengeArray<Self::Fr, Self::NWidgetRelations>) -> Self::Fr {
        if Self::NWidgetRelations::USIZE == 0 {
            challenges.alpha_powers[0]
        } else {
            challenges.alpha_powers[Self::NWidgetRelations::USIZE - 1]
                * challenges.elements[ChallengeIndex::Alpha as usize]
        }
    }

    fn get_value(
        polynomials: &Self::PC,
        evaluation_type: EvaluationType,
        id: PolynomialIndex,
        index: Option<usize>,
    ) -> Self::Fr;
}

pub(crate) struct EvaluationGetterImpl<H, F, G, NWidgetRelations>
where
    F: Field + FftField,
    G: AffineRepr,
    H: BarretenHasher,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    phantom: PhantomData<(F, H, G, NWidgetRelations)>,
}

impl<H, F: Field + FftField, G: AffineRepr, NWidgetRelations> BaseGetter
    for EvaluationGetterImpl<H, F, G, NWidgetRelations>
where
    H: BarretenHasher,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    type Hasher = H;
    type Fr = F;
    type G1 = G;
    type PC = PolyArray<F>;
    type NWidgetRelations = NWidgetRelations;

    fn get_value(
        polynomials: &PolyArray<F>,
        evaluation_type: EvaluationType,
        id: PolynomialIndex,
        index: Option<usize>,
    ) -> F {
        assert!(index.is_none());
        match evaluation_type {
            EvaluationType::NonShifted => polynomials[id].1,
            EvaluationType::Shifted => polynomials[id].0,
        }
    }
}

/// Implements loading polynomial openings from transcript in addition to BaseGetter's
/// loading challenges from the transcript and computing powers of α
pub(crate) trait EvaluationGetter {
    type Hasher: BarretenHasher;
    type Fr: Field + FftField;
    type G1: AffineRepr;
    type NWidgetRelations: generic_array::ArrayLength<Self::Fr>;

    /// Get a polynomial at offset `id`
    ///
    /// # Arguments
    ///
    /// * `polynomials` - An array of polynomials
    ///
    /// # Type Parameters
    ///
    /// * `use_shifted_evaluation` - Whether to pick first or second
    /// * `id` - Polynomial index.
    ///
    /// # Returns
    ///
    /// The chosen polynomial
    fn get_evaluation_value(
        polynomials: &PolyArray<Self::Fr>,
        evaluation_type: EvaluationType,
        id: PolynomialIndex,
        index: Option<usize>,
    ) -> &Self::Fr {
        assert!(index.is_none());
        match evaluation_type {
            EvaluationType::Shifted => &polynomials[id].1,
            EvaluationType::NonShifted => &polynomials[id].0,
        }
    }

    /// Return an array with poly
    ///
    /// # Arguments
    ///
    /// * `polynomial_manifest`
    /// * `transcript`
    ///
    /// # Returns
    ///
    /// `PolyArray`
    fn get_polynomial_evaluations(
        polynomial_manifest: &PolynomialManifest,
        transcript: &Transcript<Self::Hasher>,
    ) -> PolyArray<Self::Fr> {
        let mut result: PolyArray<Self::Fr> = Default::default();
        for i in 0..polynomial_manifest.len() {
            let info = &polynomial_manifest[i.into()];
            let label = info.polynomial_label.clone();
            result[i.into()].0 = transcript.get_field_element(&label);

            if info.requires_shifted_evaluation {
                result[info.index].1 = transcript.get_field_element(&(label + "_omega"));
            } else {
                result[info.index].1 = Self::Fr::zero();
            }
        }
        result
    }
}

impl<H, F: Field + FftField, G: AffineRepr, NWidgetRelations> EvaluationGetter
    for EvaluationGetterImpl<H, F, G, NWidgetRelations>
where
    H: BarretenHasher,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    type Hasher = H;
    type Fr = F;
    type G1 = G;
    type NWidgetRelations = NWidgetRelations;
}

pub(crate) struct FFTGetterImpl<H, F, G, NWidgetRelations>
where
    F: Field + FftField,
    H: BarretenHasher,
    G: AffineRepr,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    phantom: PhantomData<(F, H, G, NWidgetRelations)>,
}

impl<H, F, G, NWidgetRelations> BaseGetter for FFTGetterImpl<H, F, G, NWidgetRelations>
where
    F: Field + FftField,
    H: BarretenHasher,
    G: AffineRepr,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    type Hasher = H;
    type Fr = F;
    type G1 = G;
    type PC = PolyPtrMap<F>;
    type NWidgetRelations = NWidgetRelations;

    fn get_value(
        polynomials: &PolyPtrMap<F>,
        evaluation_type: EvaluationType,
        id: PolynomialIndex,
        index: Option<usize>,
    ) -> F {
        // TODO ew
        let index = index.unwrap();
        let poly = &polynomials.coefficients.get(&id).unwrap();
        if evaluation_type == EvaluationType::Shifted {
            let shifted_index = (index + polynomials.index_shift) & polynomials.block_mask;
            (**poly).read().unwrap()[shifted_index]
        } else {
            (**poly).read().unwrap()[index]
        }
    }
}

impl<H, F, G: AffineRepr, NWidgetRelations> FFTGetter for FFTGetterImpl<H, F, G, NWidgetRelations>
where
    F: Field + FftField,
    H: BarretenHasher,
    G: AffineRepr,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    type Hasher = H;
    type Fr = F;
    type G1 = G;
    type NWidgetRelations = NWidgetRelations;
}

/// Provides access to polynomials (monomial or coset FFT) for use in widgets
/// Coset FFT access is needed in quotient construction.
pub(crate) trait FFTGetter {
    type Hasher: BarretenHasher;
    type Fr: Field + FftField;
    type G1: AffineRepr;
    type NWidgetRelations: generic_array::ArrayLength<Self::Fr>;

    fn get_polynomials(
        key: &ProvingKey<Self::Fr>,
        required_polynomial_ids: &HashSet<PolynomialIndex>,
    ) -> PolyPtrMap<Self::Fr>
    where
        Self: Sized,
    {
        let mut result = PolyPtrMap::new();
        let label_suffix = "_fft";

        result.block_mask = key.large_domain.size - 1;
        result.index_shift = 4;

        // TODO sus clone
        for info in key.polynomial_manifest.clone() {
            if required_polynomial_ids.get(&info.index).is_some() {
                let label = info.polynomial_label.clone() + label_suffix;
                let poly = key.polynomial_store.get(&label).unwrap();
                result.coefficients.insert(info.index, poly);
            }
        }
        result
    }
}
