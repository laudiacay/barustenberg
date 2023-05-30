use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use ark_bn254::G1Affine;
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};
use once_cell::sync::Lazy;
use typenum::Unsigned;

use crate::{
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::{
            polynomial_manifest::{EvaluationType, PolynomialIndex},
            prover_settings::Settings,
            PolynomialManifest,
        },
    },
    transcript::{BarretenHasher, Transcript, TranscriptKey, TranscriptWrapper},
};

use self::containers::{ChallengeArray, CoefficientArray, PolyArray, PolyContainer, PolyPtrMap};

pub(crate) enum ChallengeIndex {
    Alpha,
    Beta,
    Gamma,
    Eta,
    Zeta,
    MaxNumChallenges,
}

pub(crate) const CHALLENGE_BIT_ALPHA: usize = 1 << (ChallengeIndex::Alpha as usize);
pub(crate) const CHALLENGE_BIT_BETA: usize = 1 << (ChallengeIndex::Beta as usize);
pub(crate) const CHALLENGE_BIT_GAMMA: usize = 1 << (ChallengeIndex::Gamma as usize);
pub(crate) const CHALLENGE_BIT_ETA: usize = 1 << (ChallengeIndex::Eta as usize);
pub(crate) const CHALLENGE_BIT_ZETA: usize = 1 << (ChallengeIndex::Zeta as usize);

// need maxnumchallenges as a typenum, not just as an enum
pub(crate) type MaxNumChallengesTN = typenum::consts::U5;

// and check its correspondance with the enum before we continue...
static _MAX_NUM_CHALLENGES_CHECK: Lazy<()> = Lazy::new(|| {
    assert_eq!(
        MaxNumChallengesTN::to_usize(),
        ChallengeIndex::MaxNumChallenges as usize
    );
});

pub(crate) mod containers {
    use generic_array::GenericArray;
    use std::ops::{Index, IndexMut};

    use crate::{
        plonk::proof_system::types::polynomial_manifest::PolynomialIndex, polynomials::Polynomial,
    };

    use super::MaxNumChallengesTN;
    use ark_ff::Field;
    use std::collections::HashMap;

    pub(crate) trait PolyContainer<F: Field> {}

    #[derive(Default)]
    pub(crate) struct ChallengeArray<F: Field, NumRelations: generic_array::ArrayLength<F>> {
        pub(crate) elements: GenericArray<F, MaxNumChallengesTN>,
        pub(crate) alpha_powers: GenericArray<F, NumRelations>,
    }

    pub(crate) struct PolyArray<Field>(
        pub(crate) [(Field, Field); PolynomialIndex::MaxNumPolynomials as usize],
    );

    impl<F: Field> Index<PolynomialIndex> for PolyArray<F> {
        type Output = (F, F);

        fn index(&self, index: PolynomialIndex) -> &Self::Output {
            &self.0[index as usize]
        }
    }

    impl<F: Field> IndexMut<PolynomialIndex> for PolyArray<F> {
        fn index_mut(&mut self, index: PolynomialIndex) -> &mut Self::Output {
            &mut self.0[index as usize]
        }
    }

    impl<F: Field> Default for PolyArray<F> {
        fn default() -> Self {
            Self([(F::zero(), F::zero()); PolynomialIndex::MaxNumPolynomials as usize])
        }
    }

    impl<F: Field> PolyContainer<F> for PolyArray<F> {}

    pub(crate) struct PolyPtrMap<F: Field> {
        pub(crate) coefficients: HashMap<PolynomialIndex, Polynomial<F>>,
        pub(crate) block_mask: usize,
        pub(crate) index_shift: usize,
    }

    impl<F: Field> PolyPtrMap<F> {
        pub(crate) fn new() -> Self {
            Self {
                coefficients: HashMap::new(),
                block_mask: 0,
                index_shift: 0,
            }
        }
    }

    impl<F: Field> PolyContainer<F> for PolyPtrMap<F> {}

    pub(crate) struct CoefficientArray<F: Field>([F; PolynomialIndex::MaxNumPolynomials as usize]);
    impl<F: Field> Index<PolynomialIndex> for CoefficientArray<F> {
        type Output = F;

        fn index(&self, index: PolynomialIndex) -> &Self::Output {
            &self.0[index as usize]
        }
    }
    impl<F: Field> Default for CoefficientArray<F> {
        fn default() -> Self {
            Self([F::zero(); PolynomialIndex::MaxNumPolynomials as usize])
        }
    }
}

// Getters are various structs that are used to retrieve/query various objects needed during the proof.
//
// You can query:
// - Challenges
// - Polynomial evaluations
// - Polynomials in monomial form
// - Polynomials in Lagrange form

/// Implements loading challenges from the transcript and computing powers of α, which are later used in widgets.
///
/// # Type Parameters
/// - `Field`: Base field
/// - `Transcript`: Transcript struct
/// - `Settings`: Configuration
/// - `NUM_WIDGET_RELATIONS`: How many powers of α are needed
pub(crate) trait BaseGetter<
    H: BarretenHasher,
    F: Field,
    S: Settings<H>,
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
    fn get_challenges<G1Affine: AffineRepr>(
        transcript: &Transcript<H>,
        alpha_base: F,
        required_challenges: u8,
        rng: Arc<Mutex<dyn rand::RngCore + Send + Sync>>,
    ) -> ChallengeArray<F, NWidgetRelations> {
        let mut result: ChallengeArray<F, _> = ChallengeArray::default();
        let mut add_challenge = |label: &str, tag: usize, required: bool, index: usize| {
            assert!(!required || transcript.has_challenge(label));
            if transcript.has_challenge(label) {
                assert!(index < transcript.get_num_challenges(label));
                result.elements[tag] = <Transcript<H> as TranscriptWrapper<F, G1Affine, H>>::get_challenge_field_element(transcript, label, index);
            } else {
                let mut random_bytes = vec![0u8; std::mem::size_of::<F>()];
                // TODO should you really have an unwrap here?
                rng.lock().unwrap().fill_bytes(&mut random_bytes);
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
}

/// Implements loading polynomial openings from transcript in addition to BaseGetter's
/// loading challenges from the transcript and computing powers of α
pub(crate) trait EvaluationGetter<
    H: BarretenHasher,
    F: Field,
    S: Settings<H>,
    NWidgetRelations: generic_array::ArrayLength<F>,
>: BaseGetter<H, F, S, NWidgetRelations>
{
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
    fn get_value<const USE_SHIFTED_EVALUATION: bool, const ID: usize>(
        polynomials: &PolyArray<F>,
    ) -> &F {
        if USE_SHIFTED_EVALUATION {
            &polynomials.0[ID].1
        } else {
            &polynomials.0[ID].0
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
    fn get_polynomial_evaluations<G1Affine: AffineRepr>(
        polynomial_manifest: &PolynomialManifest,
        transcript: &Transcript<H>,
    ) -> PolyArray<F> {
        let mut result: PolyArray<F> = Default::default();
        for i in 0..polynomial_manifest.len() {
            let info = &polynomial_manifest[i.into()];
            let label = info.polynomial_label.clone();
            result[i.into()].0 =
                <Transcript<H> as TranscriptWrapper<F, G1Affine, H>>::get_field_element(
                    transcript, &label,
                );

            if info.requires_shifted_evaluation {
                result[info.index].1 =
                    <Transcript<H> as TranscriptWrapper<F, G1Affine, H>>::get_field_element(
                        transcript,
                        &(label + "_omega"),
                    );
            } else {
                result[info.index].1 = F::zero();
            }
        }
        result
    }
}

/// Provides access to polynomials (monomial or coset FFT) for use in widgets
/// Coset FFT access is needed in quotient construction.
pub(crate) trait FFTGetter<
    'a,
    H,
    F,
    G1Affine: AffineRepr,
    S,
    NWidgetRelations: generic_array::ArrayLength<F>,
>: BaseGetter<H, F, S, NWidgetRelations> where
    F: Field + FftField,
    H: BarretenHasher,
    S: Settings<H>,
{
    fn get_polynomials(
        key: &ProvingKey<'a, F, G1Affine>,
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
                let poly = key.polynomial_store.get(label).unwrap();
                result.coefficients.insert(info.index, poly);
            }
        }
        result
    }

    fn get_value(
        polynomials: &PolyPtrMap<F>,
        index: usize,
        evaluation_type: EvaluationType,
        id: PolynomialIndex,
    ) -> &F {
        if evaluation_type == EvaluationType::Shifted {
            let shifted_index = (index + polynomials.index_shift) & polynomials.block_mask;
            &polynomials.coefficients.get(&id).unwrap()[shifted_index]
        } else {
            &polynomials.coefficients.get(&id).unwrap()[index]
        }
    }
}

pub(crate) struct TransitionWidgetBase<'a, F: Field + FftField, G1Affine: AffineRepr> {
    pub(crate) key: Option<Arc<ProvingKey<'a, F, G1Affine>>>,
}

impl<'a, F: Field + FftField, G1Affine: AffineRepr> TransitionWidgetBase<'a, F, G1Affine> {
    pub(crate) fn new(key: Option<Arc<ProvingKey<'a, F, G1Affine>>>) -> Self {
        Self { key }
    }

    // other methods and trait implementations
}

pub(crate) trait KernelBase<
    H: BarretenHasher,
    S: Settings<H>,
    F: Field,
    PC: PolyContainer<F>,
    G: BaseGetter<H, F, S, NumIndependentRelations>,
    NumIndependentRelations: generic_array::ArrayLength<F>,
>
{
    fn get_required_polynomial_ids() -> HashSet<PolynomialIndex>;
    fn quotient_required_challenges() -> u8;
    fn update_required_challenges() -> u8;
    fn compute_linear_terms(
        polynomials: &impl PolyContainer<F>,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        linear_terms: &mut CoefficientArray<F>,
        index: usize,
    );
    fn sum_linear_terms(
        polynomials: &impl PolyContainer<F>,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        linear_terms: &CoefficientArray<F>,
        index: usize,
    ) -> F;
    fn compute_non_linear_terms(
        polynomials: &impl PolyContainer<F>,
        challenges: &ChallengeArray<F, NumIndependentRelations>,
        quotient_term: &mut F,
        index: usize,
    );
}

pub(crate) struct TransitionWidget<
    'a,
    H: BarretenHasher,
    F: Field + FftField,
    G1Affine: AffineRepr,
    S: Settings<H>,
    PC: PolyContainer<F>,
    G: BaseGetter<H, F, S, NIndependentRelations>,
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase<H, S, F, PC, G, NIndependentRelations>,
> {
    base: TransitionWidgetBase<'a, F, G1Affine>,
    phantom: std::marker::PhantomData<(H, S, PC, G, KB, NIndependentRelations)>,
}

impl<
        H: BarretenHasher,
        F: Field + FftField,
        G1Affine: AffineRepr,
        S: Settings<H>,
        PC: PolyContainer<F>,
        G: BaseGetter<H, F, S, NIndependentRelations>,
        NIndependentRelations: generic_array::ArrayLength<F>,
        KB: KernelBase<H, S, F, PC, G, NIndependentRelations>,
    > BaseGetter<H, F, S, NIndependentRelations>
    for TransitionWidget<'_, H, F, G1Affine, S, PC, G, NIndependentRelations, KB>
{
}

impl<
        'a,
        H: BarretenHasher,
        F: Field + FftField,
        G1Affine: AffineRepr,
        S: Settings<H>,
        PC: PolyContainer<F>,
        G: BaseGetter<H, F, S, NIndependentRelations>,
        NIndependentRelations: generic_array::ArrayLength<F>,
        KB: KernelBase<H, S, F, PC, G, NIndependentRelations>,
    > FFTGetter<'a, H, F, G1Affine, S, NIndependentRelations>
    for TransitionWidget<'a, H, F, G1Affine, S, PC, G, NIndependentRelations, KB>
{
}

impl<
        'a,
        H: BarretenHasher,
        F: Field + FftField,
        G1Affine: AffineRepr,
        S: Settings<H>,
        PC: PolyContainer<F>,
        G: BaseGetter<H, F, S, NIndependentRelations>,
        NIndependentRelations: generic_array::ArrayLength<F>,
        KB: KernelBase<H, S, F, PC, G, NIndependentRelations>,
    > TransitionWidget<'a, H, F, G1Affine, S, PC, G, NIndependentRelations, KB>
{
    pub(crate) fn new(key: Option<Arc<ProvingKey<'a, F, G1Affine>>>) -> Self {
        Self {
            base: TransitionWidgetBase::new(key),
            phantom: std::marker::PhantomData,
        }
    }

    // other methods and trait implementations
    pub(crate) fn compute_quotient_contribution(
        &self,
        alpha_base: F,
        transcript: &Transcript<H>,
        rng: Arc<Mutex<dyn rand::RngCore + Send + Sync>>,
    ) -> F {
        let key = self.base.key.as_ref().expect("Proving key is missing");

        let required_polynomial_ids = KB::get_required_polynomial_ids();
        let polynomials = Self::get_polynomials(key, &required_polynomial_ids);

        let challenges = Self::get_challenges::<G1Affine>(
            transcript,
            alpha_base,
            KB::quotient_required_challenges(),
            rng,
        );

        let mut quotient_term;

        // TODO: hidden missing multithreading here
        for i in 0..key.large_domain.size {
            let mut linear_terms = CoefficientArray::default();
            KB::compute_linear_terms(&polynomials, &challenges, &mut linear_terms, i);
            let sum_of_linear_terms =
                KB::sum_linear_terms(&polynomials, &challenges, &linear_terms, i);

            quotient_term = key.quotient_polynomial_parts[i >> key.small_domain.log2_size]
                [i & (key.circuit_size - 1)];
            quotient_term += sum_of_linear_terms;
            KB::compute_non_linear_terms(&polynomials, &challenges, &mut quotient_term, i);
        }

        Self::update_alpha(&challenges)
    }
}

// Implementations for the derived classes
impl<
        'a,
        F: Field + FftField,
        H: BarretenHasher,
        G1Affine: AffineRepr,
        S: Settings<H>,
        KB: KernelBase<H, S, F, PC, G, NIndependentRelations>,
        PC: PolyContainer<F>,
        G: BaseGetter<H, F, S, NIndependentRelations>,
        NIndependentRelations: generic_array::ArrayLength<F>,
    > From<TransitionWidget<'a, H, F, G1Affine, S, PC, G, NIndependentRelations, KB>>
    for TransitionWidgetBase<'a, F, G1Affine>
{
    fn from(
        widget: TransitionWidget<'a, H, F, G1Affine, S, PC, G, NIndependentRelations, KB>,
    ) -> Self {
        widget.base
    }
}

pub(crate) trait GenericVerifierWidget<
    'a,
    F: Field,
    H: BarretenHasher,
    PC: PolyContainer<F>,
    G: BaseGetter<H, F, S, NIndependentRelations> + EvaluationGetter<H, F, S, NIndependentRelations>,
    NIndependentRelations,
    S: Settings<H>,
    KB,
> where
    NIndependentRelations: generic_array::ArrayLength<F>,
    KB: KernelBase<H, S, F, PC, G, NIndependentRelations>,
{
    fn compute_quotient_evaluation_contribution<G1Affine: AffineRepr>(
        key: &Arc<TranscriptKey<'a>>,
        alpha_base: F,
        transcript: &Transcript<H>,
        quotient_numerator_eval: &mut F,
        rng: Arc<Mutex<dyn rand::RngCore + Send + Sync>>,
    ) -> F {
        let polynomial_evaluations = G::get_polynomial_evaluations::<G1Affine>(
            &key.as_ref().polynomial_manifest,
            transcript,
        );
        let challenges = G::get_challenges::<G1Affine>(
            transcript,
            alpha_base,
            KB::quotient_required_challenges(),
            rng,
        );

        let mut linear_terms = CoefficientArray::default();
        KB::compute_linear_terms(&polynomial_evaluations, &challenges, &mut linear_terms, 0);
        *quotient_numerator_eval +=
            KB::sum_linear_terms(&polynomial_evaluations, &challenges, &linear_terms, 0);
        KB::compute_non_linear_terms(
            &polynomial_evaluations,
            &challenges,
            quotient_numerator_eval,
            0,
        );

        G::update_alpha(&challenges)
    }

    fn append_scalar_multiplication_inputs(
        _key: &Arc<TranscriptKey<'_>>,
        alpha_base: F,
        transcript: &Transcript<H>,
        _scalar_mult_inputs: &mut HashMap<String, F>,
        rng: Arc<Mutex<dyn rand::RngCore + Send + Sync>>,
    ) -> F {
        let challenges = G::get_challenges::<G1Affine>(
            transcript,
            alpha_base,
            KB::quotient_required_challenges() | KB::update_required_challenges(),
            rng,
        );

        G::update_alpha(&challenges)
    }
}
