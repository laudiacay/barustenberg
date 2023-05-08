use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::plonk::proof_system::types::PolynomialManifest;
use crate::polynomials::polynomial_arithmetic::polynomial_arithmetic::FieldElement;
use crate::proof_system::work_queue::WorkQueue;
use std::collections::HashMap;
use crate::transcript::Transcript;

pub mod widget {
    pub enum ChallengeIndex {
        Alpha,
        Beta,
        Gamma,
        Eta,
        Zeta,
        MaxNumChallenges,
    }

    pub const CHALLENGE_BIT_ALPHA: u32 = 1 << ChallengeIndex::Alpha as u32;
    pub const CHALLENGE_BIT_BETA: u32 = 1 << ChallengeIndex::Beta as u32;
    pub const CHALLENGE_BIT_GAMMA: u32 = 1 << ChallengeIndex::Gamma as u32;
    pub const CHALLENGE_BIT_ETA: u32 = 1 << ChallengeIndex::Eta as u32;
    pub const CHALLENGE_BIT_ZETA: u32 = 1 << ChallengeIndex::Zeta as u32;

    pub mod containers {
        use std::collections::HashMap;

        use super::ChallengeIndex;

        pub struct ChallengeArray<Field, const NUM_WIDGET_RELATIONS: usize> {
            pub elements: [Field; ChallengeIndex::MaxNumChallenges as usize],
            pub alpha_powers: [Field; NUM_WIDGET_RELATIONS],
        }

        pub struct PolyArray<Field> {
            pub elements: [(Field, Field); PolynomialIndex::MAX_NUM_POLYNOMIALS as usize],
        }

        pub struct PolyPtrMap<Field> {
            pub coefficients: HashMap<PolynomialIndex, Vec<Field>>,
            pub block_mask: usize,
            pub index_shift: usize,
        }
    }
    pub struct CoefficientArray<T> {
        elements: [T; PolynomialIndex::MAX_NUM_POLYNOMIALS as usize],
    }
    
}

use self::widget::ChallengeIndex;
use self::widget::containers::{ChallengeArray, PolyArray};

pub struct BaseGetter;

impl BaseGetter {
    pub fn get_challenges<T, S, const NUM_WIDGET_RELATIONS: usize>(
        transcript: &T,
        alpha_base: FieldElement,
        required_challenges: u8,
    ) -> ChallengeArray<NUM_WIDGET_RELATIONS> {
        let mut result = ChallengeArray::default();

        let add_challenge = |label: &str, tag: usize, required: bool, index: usize| {
            assert!(!required || transcript.has_challenge(label));
            if transcript.has_challenge(label) {
                assert!(index < transcript.get_num_challenges(label));
                result.elements[tag] = transcript.get_challenge_field_element(label, index);
            } else {
                result.elements[tag] = FieldElement::random_element();
            }
        };

        add_challenge("alpha", ChallengeIndex::ALPHA as usize, required_challenges & widget::CHALLENGE_BIT_ALPHA != 0, 0);
        add_challenge("beta", ChallengeIndex::BETA as usize, required_challenges & widget::CHALLENGE_BIT_BETA != 0, 0);
        add_challenge("beta", ChallengeIndex::GAMMA as usize, required_challenges & widget::CHALLENGE_BIT_GAMMA != 0, 1);
        add_challenge("eta", ChallengeIndex::ETA as usize, required_challenges & widget::CHALLENGE_BIT_ETA != 0, 0);
        add_challenge("z", ChallengeIndex::ZETA as usize, required_challenges & widget::CHALLENGE_BIT_ZETA != 0, 0);

        result.alpha_powers[0] = alpha_base;
        for i in 1..NUM_WIDGET_RELATIONS {
            result.alpha_powers[i] = result.alpha_powers[i - 1] * result.elements[ChallengeIndex::ALPHA as usize];
        }

        result
    }
    pub fn update_alpha<const NUM_WIDGET_RELATIONS: usize>(
        challenges: &ChallengeArray<NUM_WIDGET_RELATIONS>,
        num_independent_relations: usize,
    ) -> FieldElement {
        if num_independent_relations == 0 {
            return challenges.alpha_powers[0];
        }
        challenges.alpha_powers[num_independent_relations - 1] * challenges.elements[ChallengeIndex::ALPHA as usize]
    }
}
pub struct EvaluationGetter<const NUM_WIDGET_RELATIONS: usize> {
    // Fields and methods from BaseGetter should be included here as well.
}

impl<const NUM_WIDGET_RELATIONS: usize> EvaluationGetter<NUM_WIDGET_RELATIONS> {
    pub fn get_value<const USE_SHIFTED_EVALUATION: bool, const ID: usize>(
        polynomials: &PolyArray,
        _: usize,
    ) -> &FieldElement {
        if USE_SHIFTED_EVALUATION {
            &polynomials[ID].1
        } else {
            &polynomials[ID].0
        }
    }

    pub fn get_polynomial_evaluations(
        polynomial_manifest: &PolynomialManifest,
        transcript: &Transcript,
    ) -> PolyArray {
        let mut result = [Default::default(); MAX_NUM_POLYNOMIALS];
        for info in polynomial_manifest.iter() {
            let label = info.polynomial_label.to_string();
            result[info.index as usize].0 = transcript.get_field_element(&label);

            if info.requires_shifted_evaluation {
                result[info.index as usize].1 = transcript.get_field_element(&(label + "_omega"));
            } else {
                result[info.index as usize].1 = FieldElement::zero();
            }
        }
        result
    }
}


