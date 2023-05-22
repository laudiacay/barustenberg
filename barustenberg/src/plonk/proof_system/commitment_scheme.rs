use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use crate::proof_system::work_queue::WorkQueue;
use crate::transcript::{BarretenHasher, Transcript};

use super::proving_key::ProvingKey;
use super::types::proof::CommitmentOpenProof;
use super::types::prover_settings::Settings;
use super::verification_key::VerificationKey;

pub trait CommitmentScheme<Fq: Field, Fr: Field + FftField, G1Affine: AffineRepr, H: BarretenHasher>
{
    fn commit(
        &mut self,
        coefficients: &mut [Fr],
        tag: &str,
        item_constant: Fr,
        queue: &mut WorkQueue<H, Fr, G1Affine>,
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
        queue: &mut WorkQueue<H, Fr, G1Affine>,
    );

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue<H, Fr, G1Affine>,
        input_key: Option<Arc<ProvingKey<Fr, G1Affine>>>,
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
        input_key: Option<Arc<ProvingKey<Fr, G1Affine>>>,
        in_lagrange_form: bool,
    );
}

pub(crate) struct KateCommitmentScheme<H: BarretenHasher, S: Settings<H>> {
    kate_open_proof: CommitmentOpenProof,
    phantom: PhantomData<(H, S)>,
}

impl<Fq: Field, Fr: Field + FftField, G1Affine: AffineRepr, H: BarretenHasher, S: Settings<H>>
    CommitmentScheme<Fq, Fr, G1Affine, H> for KateCommitmentScheme<H, S>
{
    fn commit(
        &mut self,
        coefficients: &mut [Fr],
        tag: &str,
        item_constant: Fr,
        queue: &mut WorkQueue<H, Fr, G1Affine>,
    ) {
        todo!()
    }

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<ProvingKey<Fr, G1Affine>>>,
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
        queue: &mut WorkQueue<H, Fr, G1Affine>,
    ) {
        todo!()
    }

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue<H, Fr, G1Affine>,
        input_key: Option<Arc<ProvingKey<Fr, G1Affine>>>,
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
