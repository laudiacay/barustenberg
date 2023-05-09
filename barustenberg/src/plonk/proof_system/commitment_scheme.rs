use std::collections::HashMap;
use std::sync::Arc;

use ark_bn254::{Fr, G1Affine};

use crate::ecc::Field;
use crate::proof_system::work_queue::WorkQueue;
use crate::transcript::{BarretenHasher, Transcript};

use super::proving_key::ProvingKey;
use super::types::proof::CommitmentOpenProof;
use super::types::prover_settings::Settings;
use super::verification_key::VerificationKey;

pub trait CommitmentScheme<Fr: Field, G1Affine: Field, H: BarretenHasher> {
    fn commit(
        &mut self,
        coefficients: &mut [Fr],
        tag: &str,
        item_constant: Fr,
        queue: &mut WorkQueue<Fr>,
    );

    fn compute_opening_polynomial(&self, src: &[Fr], dest: &mut [Fr], z: &Fr, n: usize);

    fn generic_batch_open(
        &self,
        src: &[Fr],
        dest: &mut [Fr],
        num_polynomials: usize,
        z_points: &[Fr],
        num_z_points: usize,
        challenges: &[Fr],
        n: usize,
        tags: &[String],
        item_constants: &[Fr],
        queue: &mut WorkQueue<Fr>,
    );

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue<Fr>,
        input_key: Option<Arc<ProvingKey<Fr>>>,
    );

    fn batch_verify(
        &self,
        transcript: &Transcript<H>,
        kate_g1_elements: &mut HashMap<String, G1Affine>,
        kate_fr_elements: &mut HashMap<String, Fr>,
        input_key: Option<Arc<VerificationKey>>,
    );

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<ProvingKey<Fr>>>,
        in_lagrange_form: bool,
    );
}

pub(crate) struct KateCommitmentScheme<H: BarretenHasher, S: Settings<H>> {
    kate_open_proof: CommitmentOpenProof,
}

impl<H: BarretenHasher, S: Settings<H>> CommitmentScheme<Fr, G1Affine, H>
    for KateCommitmentScheme<H, S>
{
    fn commit(
        &mut self,
        coefficients: &mut [Fr],
        tag: &str,
        item_constant: Fr,
        queue: &mut WorkQueue<Fr>,
    ) {
        todo!()
    }

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<ProvingKey<Fr>>>,
        in_lagrange_form: bool,
    ) {
        todo!()
    }

    fn compute_opening_polynomial(&self, src: &[Fr], dest: &mut [Fr], z: &Fr, n: usize) {
        todo!()
    }

    fn generic_batch_open(
        &self,
        src: &[Fr],
        dest: &mut [Fr],
        num_polynomials: usize,
        z_points: &[Fr],
        num_z_points: usize,
        challenges: &[Fr],
        n: usize,
        tags: &[String],
        item_constants: &[Fr],
        queue: &mut WorkQueue<Fr>,
    ) {
        todo!()
    }

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue<Fr>,
        input_key: Option<Arc<ProvingKey<Fr>>>,
    ) {
        todo!()
    }

    fn batch_verify(
        &self,
        transcript: &Transcript<H>,
        kate_g1_elements: &mut HashMap<String, G1Affine>,
        kate_fr_elements: &mut HashMap<String, Fr>,
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
