// Getters are various structs that are used to retrieve/query various objects needed during the proof.
//
// You can query:
// - Challenges
// - Polynomial evaluations
// - Polynomials in monomial form
// - Polynomials in Lagrange form

use std::{collections::HashSet, marker::PhantomData};

use ark_ec::AffineRepr;

use crate::{
    ecc::fieldext::FieldExt,
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::{
            polynomial_manifest::{EvaluationType, PolynomialIndex},
            prover_settings::Settings,
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
pub(crate) trait BaseGetter<
    H: BarretenHasher,
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    G: AffineRepr,
    S: Settings<H, F, G>,
    PC: PolyContainer<F>,
    NWidgetRelations: generic_array::ArrayLength<F>,
>
{
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
        transcript: &Transcript<H>,
        alpha_base: F,
        required_challenges: u8,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> ChallengeArray<F, NWidgetRelations> {
        let mut result: ChallengeArray<F, _> = ChallengeArray::default();
        let mut add_challenge = |label: &str, tag: usize, required: bool, index: usize| {
            assert!(!required || transcript.has_challenge(label));
            if transcript.has_challenge(label) {
                assert!(index < transcript.get_num_challenges(label));
                result.elements[tag] =
                    transcript.get_challenge_FieldExt_element(label, Some(index));
            } else {
                let mut random_bytes = vec![0u8; std::mem::size_of::<F>()];
                // TODO should you really have an unwrap here?
                rng.fill_bytes(&mut random_bytes);
                result.elements[tag] = F::from_random_bytes(random_bytes.as_ref())
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
        for i in 1..NWidgetRelations::to_usize() {
            result.alpha_powers[i] =
                result.alpha_powers[i - 1] * result.elements[ChallengeIndex::Alpha as usize];
        }
        result
    }

    fn update_alpha(challenges: &ChallengeArray<F, NWidgetRelations>) -> F {
        if NWidgetRelations::USIZE == 0 {
            challenges.alpha_powers[0]
        } else {
            challenges.alpha_powers[NWidgetRelations::USIZE - 1]
                * challenges.elements[ChallengeIndex::Alpha as usize]
        }
    }

    fn get_value(
        polynomials: &PC,
        evaluation_type: EvaluationType,
        id: PolynomialIndex,
        index: Option<usize>,
    ) -> F;
}

pub(crate) struct EvaluationGetterImpl<H, F, G, S, NWidgetRelations>
where
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    G: AffineRepr,
    H: BarretenHasher,
    S: Settings<H, F, G>,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    phantom: PhantomData<(F, H, G, S, NWidgetRelations)>,
}

impl<H, F: ark_ff::Field + ark_ff::FftField + FieldExt, G: AffineRepr, S, NWidgetRelations>
    BaseGetter<H, F, G, S, PolyArray<F>, NWidgetRelations>
    for EvaluationGetterImpl<H, F, G, S, NWidgetRelations>
where
    H: BarretenHasher,
    S: Settings<H, F, G>,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
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

impl<H, F: ark_ff::Field + ark_ff::FftField + FieldExt, G: AffineRepr, S, NWidgetRelations>
    EvaluationGetter<H, F, G, S, NWidgetRelations>
    for EvaluationGetterImpl<H, F, G, S, NWidgetRelations>
where
    H: BarretenHasher,
    S: Settings<H, F, G>,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
}

/// Implements loading polynomial openings from transcript in addition to BaseGetter's
/// loading challenges from the transcript and computing powers of α
pub(crate) trait EvaluationGetter<
    H: BarretenHasher,
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    G: AffineRepr,
    S: Settings<H, F, G>,
    NWidgetRelations: generic_array::ArrayLength<F>,
>: BaseGetter<H, F, G, S, PolyArray<F>, NWidgetRelations>
{
    // fn get_value(
    //     polynomials: &PolyArray<F>,
    //     evaluation_type: EvaluationType,
    //     id: PolynomialIndex,
    //     index: Option<usize>,
    // ) -> &F {
    //     assert!(index.is_none());
    //     match evaluation_type {
    //         EvaluationType::NonShifted => &polynomials[id].1,
    //         EvaluationType::Shifted => &polynomials[id].0,
    //     }
    // }
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
        polynomials: &PolyArray<F>,
        evaluation_type: EvaluationType,
        id: PolynomialIndex,
        index: Option<usize>,
    ) -> &F {
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
        transcript: &Transcript<H>,
    ) -> PolyArray<F> {
        let mut result: PolyArray<F> = Default::default();
        for i in 0..polynomial_manifest.len() {
            let info = &polynomial_manifest[i.into()];
            let label = info.polynomial_label.clone();
            result[i.into()].0 = transcript.get_FieldExt_element(&label);

            if info.requires_shifted_evaluation {
                result[info.index].1 = transcript.get_FieldExt_element(&(label + "_omega"));
            } else {
                result[info.index].1 = F::zero();
            }
        }
        result
    }
}

pub(crate) struct FFTGetterImpl<H, F, G, S, NWidgetRelations>
where
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    H: BarretenHasher,
    S: Settings<H, F, G>,
    G: AffineRepr,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
    phantom: PhantomData<(F, H, S, G, NWidgetRelations)>,
}

impl<H, F, G, S, NWidgetRelations> BaseGetter<H, F, G, S, PolyPtrMap<F>, NWidgetRelations>
    for FFTGetterImpl<H, F, G, S, NWidgetRelations>
where
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    H: BarretenHasher,
    S: Settings<H, F, G>,
    G: AffineRepr,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
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
            poly.borrow()[shifted_index]
        } else {
            poly.borrow()[index]
        }
    }
}

impl<'a, H, F, G, S, NWidgetRelations> FFTGetter<'a, H, F, G, S, NWidgetRelations>
    for FFTGetterImpl<H, F, G, S, NWidgetRelations>
where
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    H: BarretenHasher,
    S: Settings<H, F, G>,
    G: AffineRepr,
    NWidgetRelations: generic_array::ArrayLength<F>,
{
}

/// Provides access to polynomials (monomial or coset FFT) for use in widgets
/// Coset FFT access is needed in quotient construction.
pub(crate) trait FFTGetter<
    'a,
    H,
    F,
    G: AffineRepr,
    S,
    NWidgetRelations: generic_array::ArrayLength<F>,
>: BaseGetter<H, F, G, S, PolyPtrMap<F>, NWidgetRelations> where
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    H: BarretenHasher,
    S: Settings<H, F, G>,
{
    fn get_polynomials(
        key: &ProvingKey<'a, F, G>,
        required_polynomial_ids: &HashSet<PolynomialIndex>,
    ) -> PolyPtrMap<F> {
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

    // fn get_value(
    //     polynomials: &PolyPtrMap<F>,
    //     evaluation_type: EvaluationType,
    //     id: PolynomialIndex,
    //     index: Option<usize>,
    // ) -> &F {
    //     // TODO ew
    //     let index = index.unwrap();
    //     if evaluation_type == EvaluationType::Shifted {
    //         let shifted_index = (index + polynomials.index_shift) & polynomials.block_mask;
    //         &polynomials.coefficients.get(&id).unwrap()[shifted_index]
    //     } else {
    //         &polynomials.coefficients.get(&id).unwrap()[index]
    //     }
    // }
}
