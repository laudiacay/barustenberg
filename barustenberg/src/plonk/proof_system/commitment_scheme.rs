use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::ecc::curves::bn254::fr::Fr;
use crate::ecc::curves::bn254::g1::G1Affine;
use crate::ecc::fields::field::{Field, FieldParams};
use crate::ecc::groups::affine_element::Affine;
use crate::ecc::groups::GroupParams;
use crate::proof_system::work_queue::WorkQueue;
use crate::transcript::{BarretenHasher, Transcript};

use super::proving_key::ProvingKey;
use super::types::proof::CommitmentOpenProof;
use super::types::prover_settings::Settings;
use super::verification_key::VerificationKey;

pub trait CommitmentScheme<
    FqP: FieldParams,
    FrP: FieldParams,
    G1AffineP: GroupParams<FqP, FrP>,
    H: BarretenHasher,
>
{
    fn commit(
        &mut self,
        coefficients: &mut [Field<FrP>],
        tag: &str,
        item_constant: Field<FrP>,
        queue: &mut WorkQueue<H, FrP>,
    );

    fn compute_opening_polynomial(
        &self,
        src: &[Field<FrP>],
        dest: &mut [Field<FrP>],
        z: &Field<FrP>,
        n: usize,
    );

    fn generic_batch_open(
        &self,
        src: &[Field<FrP>],
        dest: &mut [Field<FrP>],
        num_polynomials: usize,
        z_points: &[Field<FrP>],
        num_z_points: usize,
        challenges: &[Field<FrP>],
        n: usize,
        tags: &[String],
        item_constants: &[Field<FrP>],
        queue: &mut WorkQueue<H, FrP>,
    );

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue<H, FrP>,
        input_key: Option<Arc<ProvingKey<FrP>>>,
    );

    fn batch_verify(
        &self,
        transcript: &Transcript<H>,
        kate_g1_elements: &mut HashMap<String, Affine<FqP, Field<FqP>, FrP, G1AffineP>>,
        kate_fr_elements: &mut HashMap<String, Field<FrP>>,
        input_key: Option<Arc<VerificationKey>>,
    );

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<ProvingKey<FrP>>>,
        in_lagrange_form: bool,
    );
}

pub(crate) struct KateCommitmentScheme<H: BarretenHasher, S: Settings<H>> {
    kate_open_proof: CommitmentOpenProof,
    phantom: PhantomData<(H, S)>,
}

impl<
        FqP: FieldParams,
        FrP: FieldParams,
        G1AffineP: GroupParams<FqP, FrP>,
        H: BarretenHasher,
        S: Settings<H>,
    > CommitmentScheme<FqP, FrP, G1AffineP, H> for KateCommitmentScheme<H, S>
{
    fn commit(
        &mut self,
        coefficients: &mut [Field<FrP>],
        tag: &str,
        item_constant: Field<FrP>,
        queue: &mut WorkQueue<H, FrP>,
    ) {
        todo!()
    }

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<ProvingKey<FrP>>>,
        in_lagrange_form: bool,
    ) {
        todo!()
    }

    fn compute_opening_polynomial(
        &self,
        src: &[Field<FrP>],
        dest: &mut [Field<FrP>],
        z: &Field<FrP>,
        n: usize,
    ) {
        todo!()
    }

    fn generic_batch_open(
        &self,
        src: &[Field<FrP>],
        dest: &mut [Field<FrP>],
        num_polynomials: usize,
        z_points: &[Field<FrP>],
        num_z_points: usize,
        challenges: &[Field<FrP>],
        n: usize,
        tags: &[String],
        item_constants: &[Field<FrP>],
        queue: &mut WorkQueue<H, FrP>,
    ) {
        todo!()
    }

    fn batch_open(
        &mut self,
        transcript: &Transcript<H>,
        queue: &mut WorkQueue<H, FrP>,
        input_key: Option<Arc<ProvingKey<FrP>>>,
    ) {
        todo!()
    }

    fn batch_verify(
        &self,
        transcript: &Transcript<H>,
        kate_g1_elements: &mut HashMap<String, Affine<FqP, Field<FqP>, FrP, G1AffineP>>,
        kate_fr_elements: &mut HashMap<String, Field<FrP>>,
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
