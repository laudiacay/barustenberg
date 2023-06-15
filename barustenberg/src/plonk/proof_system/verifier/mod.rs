use crate::transcript::{BarretenHasher, Manifest};

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use super::{
    commitment_scheme::CommitmentScheme,
    types::{prover_settings::Settings, Proof},
};

use std::collections::HashMap;
use std::sync::Arc;

use super::verification_key::VerificationKey;

#[cfg(test)]
mod test;

pub(crate) trait VerifierBase<'a, Fr: Field + FftField, H: BarretenHasher, PS: Settings<H>> {
    fn new(verifier_key: Option<Arc<VerificationKey<'a, Fr>>>, manifest: Manifest) -> Self;
    fn validate_commitments(&self) -> bool;
    fn validate_scalars(&self) -> bool;
    fn verify_proof(&self, proof: &Proof) -> bool;
}

pub(crate) struct Verifier<
    'a,
    Fq: Field,
    Fr: Field + FftField,
    G1Affine: AffineRepr,
    H: BarretenHasher,
    PS: Settings<H>,
> {
    settings: PS,
    key: Option<Arc<VerificationKey<'a, Fr>>>,
    manifest: Manifest,
    kate_g1_elements: HashMap<String, G1Affine>,
    kate_fr_elements: HashMap<String, Fr>,
    commitment_scheme: Box<dyn CommitmentScheme<Fq, Fr, G1Affine, H>>,
}

impl<
        'a,
        Fq: Field,
        Fr: Field + FftField,
        G1Affine: AffineRepr,
        H: BarretenHasher,
        PS: Settings<H>,
    > VerifierBase<'a, Fr, H, PS> for Verifier<'a, Fq, Fr, G1Affine, H, PS>
{
    fn new(_verifier_key: Option<Arc<VerificationKey<'a, Fr>>>, _manifest: Manifest) -> Self {
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
