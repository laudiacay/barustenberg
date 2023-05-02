use std::collections::HashMap;
use std::sync::Arc;

use ark_bn254::G1Affine;

use crate::ecc::Field;
use crate::proof_system::work_queue::WorkQueue;
use crate::transcript::{HasherType, Transcript};

use super::proving_key::ProvingKey;
use super::types::proof::CommitmentOpenProof;
use super::types::prover_settings::SettingsBase;
use super::verification_key::VerificationKey;

pub trait CommitmentScheme<Fr: Field, H: HasherType> {
    fn commit(
        &mut self,
        coefficients: &mut [Self::Fr],
        tag: &str,
        item_constant: Self::Fr,
        queue: &mut WorkQueue,
    );

    fn compute_opening_polynomial(
        &self,
        src: &[Self::Fr],
        dest: &mut [Self::Fr],
        z: &Self::Fr,
        n: usize,
    );

    fn generic_batch_open(
        &self,
        src: &[Self::Fr],
        dest: &mut [Self::Fr],
        num_polynomials: usize,
        z_points: &[Self::Fr],
        num_z_points: usize,
        challenges: &[Self::Fr],
        n: usize,
        tags: &[String],
        item_constants: &[Self::Fr],
        queue: &mut WorkQueue,
    );

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue,
        input_key: Option<Arc<ProvingKey>>,
    );

    fn batch_verify(
        &self,
        transcript: &Transcript<H>,
        kate_g1_elements: &mut HashMap<String, G1Affine>,
        kate_fr_elements: &mut HashMap<String, Self::Fr>,
        input_key: Option<Arc<VerificationKey>>,
    );

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<ProvingKey>>,
        in_lagrange_form: bool,
    );
}

pub(crate) struct KateCommitmentScheme<H: HasherType, S: SettingsBase<H>> {
    kate_open_proof: CommitmentOpenProof,
}

impl<Fr: Field, H: HasherType, S: SettingsBase<H>> CommitmentScheme<Fr, H>
    for KateCommitmentScheme<H, S>
{
    fn commit(
        &mut self,
        coefficients: &mut [Self::Fr],
        tag: &str,
        item_constant: Self::Fr,
        queue: &mut WorkQueue,
    ) {
        todo!()
    }

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<ProvingKey>>,
        in_lagrange_form: bool,
    ) {
        todo!()
    }

    fn compute_opening_polynomial(
        &self,
        src: &[Self::Fr],
        dest: &mut [Self::Fr],
        z: &Self::Fr,
        n: usize,
    ) {
        todo!()
    }

    fn generic_batch_open(
        &self,
        src: &[Self::Fr],
        dest: &mut [Self::Fr],
        num_polynomials: usize,
        z_points: &[Self::Fr],
        num_z_points: usize,
        challenges: &[Self::Fr],
        n: usize,
        tags: &[String],
        item_constants: &[Self::Fr],
        queue: &mut WorkQueue,
    ) {
        todo!()
    }

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue,
        input_key: Option<Arc<ProvingKey>>,
    ) {
        todo!()
    }

    fn batch_verify(
        &self,
        transcript: &Transcript<H>,
        kate_g1_elements: &mut HashMap<String, G1Affine>,
        kate_fr_elements: &mut HashMap<String, Self::Fr>,
        input_key: Option<Arc<VerificationKey>>,
    ) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_commitment_scheme() {
        todo!("see commitment_scheme.test.cpp")
    }
}
