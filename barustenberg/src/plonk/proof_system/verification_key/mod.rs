use std::{collections::HashMap, sync::Arc};

use ark_bn254::{Fr, G1Affine, G2Affine};

use crate::{
    polynomials::evaluation_domain::EvaluationDomain,
    srs::reference_string::VerifierReferenceString,
};

use super::types::PolynomialManifest;

pub(crate) struct VerificationKey<'a> {
    composer_type: u32,
    circuit_size: usize,
    log_circuit_size: usize,
    num_inputs: usize,
    domain: EvaluationDomain<'a, Fr>,
    reference_string: Arc<dyn VerifierReferenceString<G2Affine>>,
    commitments: HashMap<String, G1Affine>,
    pub(crate) polynomial_manifest: PolynomialManifest,
    /// This is a member variable so as to avoid recomputing it in the different places of the verifier algorithm.
    /// Note that recomputing would also have added constraints to the recursive verifier circuit.
    /// ʓ^n (ʓ being the 'evaluation challenge')
    z_pow_n: Fr,
    contains_recursive_proof: bool,
    recursive_proof_public_input_indices: Vec<u32>,
    program_width: usize,
}
