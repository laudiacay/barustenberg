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

pub(crate) trait CommitmentScheme<
    Fq: Field,
    Fr: Field + FftField,
    G1Affine: AffineRepr,
    H: BarretenHasher,
>
{
    fn commit<'a>(
        &mut self,
        coefficients: &mut [Fr],
        tag: &str,
        item_constant: Fr,
        queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
    );

    fn compute_opening_polynomial(&self, src: &[Fr], dest: &mut [Fr], z: &Fr, n: usize);

    fn generic_batch_open<'a>(
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
        queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
    );

    fn batch_open<'a>(
        &mut self,
        transcript: &Transcript<H, Fr, G1Affine>,
        queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
        input_key: Option<Arc<ProvingKey<'a, Fr, G1Affine>>>,
    );

    fn batch_verify<'a>(
        &self,
        transcript: &Transcript<H, Fr, G1Affine>,
        kate_g1_elements: &mut HashMap<String, G1Affine>,
        kate_fr_elements: &mut HashMap<String, Fr>,
        input_key: Option<Arc<VerificationKey<'a, Fr>>>,
    );

    fn add_opening_evaluations_to_transcript<'a>(
        &self,
        transcript: &mut Transcript<H, Fr, G1Affine>,
        input_key: Option<Arc<ProvingKey<'a, Fr, G1Affine>>>,
        in_lagrange_form: bool,
    );
}

#[derive(Default)]
pub(crate) struct KateCommitmentScheme<H: BarretenHasher, S: Settings<H>> {
    kate_open_proof: CommitmentOpenProof,
    phantom: PhantomData<(H, S)>,
}

impl<Fq: Field, Fr: Field + FftField, G1Affine: AffineRepr, H: BarretenHasher, S: Settings<H>>
    CommitmentScheme<Fq, Fr, G1Affine, H> for KateCommitmentScheme<H, S>
{
    fn commit<'a>(
        &mut self,
        _coefficients: &mut [Fr],
        _tag: &str,
        _item_constant: Fr,
        _queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
    ) {
        todo!()
    }

    fn add_opening_evaluations_to_transcript(
        &self,
        _transcript: &mut Transcript<H, Fr, G1Affine>,
        _input_key: Option<Arc<ProvingKey<'_, Fr, G1Affine>>>,
        _in_lagrange_form: bool,
    ) {
        todo!()
    }

    fn compute_opening_polynomial(&self, _src: &[Fr], _dest: &mut [Fr], _z: &Fr, _n: usize) {
        todo!()
    }

    fn generic_batch_open<'a>(
        &self,
        _src: &[Fr],
        _dest: &mut [Fr],
        _num_polynomials: usize,
        _z_points: &[Fr],
        _num_z_points: usize,
        _challenges: &[Fr],
        _n: usize,
        _tags: &[String],
        _item_constants: &[Fr],
        _queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
    ) {
        todo!()
    }

    fn batch_open<'a>(
        &mut self,
        _transcript: &Transcript<H, Fr, G1Affine>,
        _queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
        _input_key: Option<Arc<ProvingKey<'a, Fr, G1Affine>>>,
    ) {
        todo!()
    }

    fn batch_verify<'a>(
        &self,
        _transcript: &Transcript<H, Fr, G1Affine>,
        _kate_g1_elements: &mut HashMap<String, G1Affine>,
        _kate_fr_elements: &mut HashMap<String, Fr>,
        _input_key: Option<Arc<VerificationKey<'a, Fr>>>,
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
