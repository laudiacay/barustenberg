use std::{collections::HashMap, sync::Arc};

use ark_bn254::{G1Affine, G2Affine};
use ark_ff::{FftField, Field};

use crate::{
    polynomials::evaluation_domain::EvaluationDomain,
    srs::reference_string::VerifierReferenceString,
};

use super::types::PolynomialManifest;

pub(crate) struct VerificationKey<'a, Fr: Field + FftField> {
    composer_type: u32,
    circuit_size: usize,
    log_circuit_size: usize,
    num_inputs: usize,
    pub(crate) domain: EvaluationDomain<'a, Fr>,
    reference_string: Arc<dyn VerifierReferenceString<G2Affine>>,
    commitments: HashMap<String, G1Affine>,
    pub(crate) polynomial_manifest: PolynomialManifest,
    /// This is a member variable so as to avoid recomputing it in the different places of the verifier algorithm.
    /// Note that recomputing would also have added constraints to the recursive verifier circuit.
    /// ʓ^n (ʓ being the 'evaluation challenge')
    pub(crate) z_pow_n: Fr,
    contains_recursive_proof: bool,
    recursive_proof_public_input_indices: Vec<u32>,
    pub(crate) program_width: usize,
}
