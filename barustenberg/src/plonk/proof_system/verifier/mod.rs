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

use crate::{
    ecc::{
        fields::field::{Field, FieldParams},
        groups::{affine_element::Affine, GroupParams},
    },
    transcript::{BarretenHasher, Manifest},
};

use super::{
    commitment_scheme::CommitmentScheme,
    types::{prover_settings::Settings, Proof},
};

use std::collections::HashMap;
use std::sync::Arc;

use super::verification_key::VerificationKey;

#[cfg(test)]
mod test;

pub trait VerifierBase<H: BarretenHasher, PS: Settings<H>> {
    fn new(verifier_key: Option<Arc<VerificationKey>>, manifest: Manifest) -> Self;
    fn validate_commitments(&self) -> bool;
    fn validate_scalars(&self) -> bool;
    fn verify_proof(&self, proof: &Proof) -> bool;
}

pub struct Verifier<
    FqP: FieldParams,
    FrP: FieldParams,
    G1AffineP: GroupParams<FqP, FrP>,
    H: BarretenHasher,
    PS: Settings<H>,
> {
    settings: PS,
    key: Option<Arc<VerificationKey>>,
    manifest: Manifest,
    kate_g1_elements: HashMap<String, Affine<FqP, Field<FqP>, FrP, G1AffineP>>,
    kate_fr_elements: HashMap<String, Field<FrP>>,
    commitment_scheme: Box<dyn CommitmentScheme<FqP, FrP, G1AffineP, H>>,
}

impl<
        FqP: FieldParams,
        FrP: FieldParams,
        G1AffineP: GroupParams<FqP, FrP>,
        H: BarretenHasher,
        PS: Settings<H>,
    > VerifierBase<H, PS> for Verifier<FqP, FrP, G1AffineP, H, PS>
{
    fn new(verifier_key: Option<Arc<VerificationKey>>, manifest: Manifest) -> Self {
        // Implement constructor logic here.
        todo!("Verifier::new")
    }

    fn validate_commitments(&self) -> bool {
        // Implement validate_commitments logic here.
        todo!("Verifier::validate_commitments")
    }

    fn validate_scalars(&self) -> bool {
        // Implement validate_scalars logic here.
        todo!("Verifier::validate_scalars")
    }

    fn verify_proof(&self, proof: &Proof) -> bool {
        // Implement verify_proof logic here.
        todo!("Verifier::verify_proof")
    }
}
