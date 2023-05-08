// use crate::barretenberg::common::throw_or_abort;
// use crate::barretenberg::ecc::curves::bn254::{
//     fq12::Fq12, g1::AffineElement as G1AffineElement, pairing,
// };
// use crate::barretenberg::plonk::proof_system::constants::{
//     standard_verifier_settings, turbo_verifier_settings, ultra_to_standard_verifier_settings,
//     ultra_verifier_settings, ultra_with_keccak_verifier_settings, ProgramSettings,
// };
// use crate::barretenberg::plonk::proof_system::verifier::{KateVerificationScheme, Verifier};
// use crate::barretenberg::plonk::proof_system::PlonkProof;
// use crate::barretenberg::plonk::public_inputs::PublicInputs;
// use crate::barretenberg::polynomials::polynomial_arithmetic;
// use crate::barretenberg::scalar_multiplication;

use ark_bn254::G1Affine;

use crate::{
    ecc::Field,
    transcript::{HasherType, Manifest},
};

use super::{
    commitment_scheme::CommitmentScheme,
    types::{prover_settings::SettingsBase, Proof},
};

use std::collections::HashMap;
use std::sync::Arc;

use super::verification_key::VerificationKey;

#[cfg(test)]
mod test;

pub trait VerifierBase<H: HasherType, PS: SettingsBase<H>> {
    fn new(verifier_key: Option<Arc<VerificationKey>>, manifest: Manifest) -> Self;
    fn validate_commitments(&self) -> bool;
    fn validate_scalars(&self) -> bool;
    fn verify_proof(&self, proof: &Proof) -> bool;
}

impl<H: HasherType, PS: SettingsBase<H>> dyn VerifierBase<H, PS> {
    pub fn from_other(other: &Self) -> Self {
        Self {
            manifest: other.manifest.clone(),
            key: other.key.clone(),
            commitment_scheme: other.commitment_scheme.clone(),
        }
    }
}

pub struct Verifier<Fr: Field, H: HasherType, PS: SettingsBase<H>> {
    settings: PS,
    key: Option<Arc<VerificationKey>>,
    manifest: Manifest,
    kate_g1_elements: HashMap<String, G1Affine>,
    kate_fr_elements: HashMap<String, Fr>,
    commitment_scheme: Box<dyn CommitmentScheme<Fr, G1Affine, H>>,
}

impl<Fr: Field, H: HasherType, PS: SettingsBase<H>> VerifierBase<H, PS> for Verifier<Fr, H, PS> {
    fn new(verifier_key: Option<Arc<VerificationKey>>, manifest: Manifest) -> Self {
        // Implement constructor logic here.
    }

    fn validate_commitments(&self) -> bool {
        // Implement validate_commitments logic here.
    }

    fn validate_scalars(&self) -> bool {
        // Implement validate_scalars logic here.
    }

    fn verify_proof(&self, proof: &Proof) -> bool {
        // Implement verify_proof logic here.
    }
}
