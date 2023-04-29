use std::collections::HashMap;
use std::sync::Arc;

use crate::ecc::Field;
use crate::proof_system::work_queue::WorkQueue;
use crate::srs::reference_string::g1;
use crate::transcript::{HasherType, Transcript};

use super::proving_key::ProvingKey;
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
        kate_g1_elements: &mut HashMap<String, g1::AffineElement>,
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
