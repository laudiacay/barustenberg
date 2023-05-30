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

use crate::transcript::{BarretenHasher, Manifest};

use ark_ec::AffineRepr;
use ark_ff::Field;

use super::{
    commitment_scheme::CommitmentScheme,
    types::{prover_settings::Settings, Proof},
};

use std::collections::HashMap;
use std::sync::Arc;

use super::verification_key::VerificationKey;

#[cfg(test)]
mod test;

pub(crate) trait VerifierBase<'a, H: BarretenHasher, PS: Settings<H>> {
    fn new(verifier_key: Option<Arc<VerificationKey<'a>>>, manifest: Manifest) -> Self;
    fn validate_commitments(&self) -> bool;
    fn validate_scalars(&self) -> bool;
    fn verify_proof(&self, proof: &Proof) -> bool;
}

pub(crate) struct Verifier<
    'a,
    Fq: Field,
    Fr: Field,
    G1Affine: AffineRepr,
    H: BarretenHasher,
    PS: Settings<H>,
> {
    settings: PS,
    key: Option<Arc<VerificationKey<'a>>>,
    manifest: Manifest,
    kate_g1_elements: HashMap<String, G1Affine>,
    kate_fr_elements: HashMap<String, Fr>,
    commitment_scheme: Box<dyn CommitmentScheme<Fq, Fr, G1Affine, H>>,
}

impl<'a, Fq: Field, Fr: Field, G1Affine: AffineRepr, H: BarretenHasher, PS: Settings<H>>
    VerifierBase<'a, H, PS> for Verifier<'a, Fq, Fr, G1Affine, H, PS>
{
    fn new(_verifier_key: Option<Arc<VerificationKey<'a>>>, _manifest: Manifest) -> Self {
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

    fn verify_proof(&self, _proof: &Proof) -> bool {
        // Implement verify_proof logic here.
        todo!("Verifier::verify_proof")
    }
}
