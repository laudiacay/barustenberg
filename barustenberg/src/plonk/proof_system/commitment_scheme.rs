use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};
use ff::BatchInverter;

use crate::polynomials::polynomial_arithmetic;
use crate::proof_system::work_queue::{WorkItem, WorkQueue, WorkType};
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
        tag: String,
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
        coefficients: &mut [Fr],
        tag: String,
        item_constant: Fr,
        queue: &mut WorkQueue<'a, H, Fr, G1Affine>,
    ) {
        queue.add_to_queue(WorkItem {
            work_type: WorkType::ScalarMultiplication,
            mul_scalars: coefficients,
            tag,
            constant: item_constant,
            index: 0,
        })
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
    ) {
        // In this function, we compute the opening polynomials using Kate scheme for multiple input
        // polynomials with multiple evaluation points. The input polynomials are separated according
        // to the point at which they need to be opened at, viz.
        //
        // z_1 -> [F_{1,1},  F_{1,2},  F_{1, 3},  ...,  F_{1, m}]
        // z_2 -> [F_{2,1},  F_{2,2},  F_{2, 3},  ...,  F_{2, m}]
        // ...
        // z_t -> [F_{t,1},  F_{t,2},  F_{t, 3},  ...,  F_{t, m}]
        //
        // Note that the input polynomials are assumed to be stored in their coefficient forms
        // in a single array `src` in the same order as above. Polynomials which are to be opened at a
        // same point `z_i` are combined linearly using the powers of the challenge `γ_i = challenges[i]`.
        //
        // The output opened polynomials [W_{1},  W_{2}, ...,  W_{t}] are saved in the array `dest`.
        //             1
        // W_{i} = ---------- * \sum_{j=1}^{m} (γ_i)^{j-1} * [ F_{i,j}(X) - F_{i,j}(z_i) ]
        //           X - z_i
        //
        // P.S. This function isn't actually used anywhere in PLONK but was written as a generic batch
        // opening test case.

        // compute [-z, -z', ... ]
        let mut divisors = vec![Fr::zero(); num_z_points];
        for i in 0..num_z_points {
            divisors[i] = -z_points[i];
        }
        // invert them all
        divisors
            .iter_mut()
            .map(|x| *x = x.inverse().unwrap())
            .for_each(drop);

        for i in 0..num_z_points {
            let challenge = challenges[i];
            let divisor = divisors[i];
            let src_offset = i * n * num_polynomials;
            let dest_offset = i * n;

            // compute i-th linear combination polynomial
            // F_i(X) = \sum_{j = 1, 2, ..., num_poly} \gamma^{j - 1} * f_{i, j}(X)
            for k in 0..n {
                let mut coeff_sum = Fr::zero();
                let mut challenge_pow = Fr::one();
                for j in 0..num_polynomials {
                    coeff_sum += challenge_pow * src[src_offset + (j * n) + k];
                    challenge_pow *= challenge;
                }
                dest[dest_offset + k] = coeff_sum;
            }

            // evaluation of the i-th linear combination polynomial F_i(X) at z
            let d_i_eval = polynomial_arithmetic::evaluate(&dest[dest_offset..], &z_points[i], n);

            // compute coefficients of h_i(X) = (F_i(X) - F_i(z))/(X - z) as done in the previous function
            dest[dest_offset] -= d_i_eval;
            dest[dest_offset] *= divisor;
            for k in 1..n {
                dest[dest_offset + k] -= dest[dest_offset + k - 1];
                dest[dest_offset + k] *= divisor;
            }

            // commit to the i-th opened polynomial
            self.commit(&mut dest[dest_offset..], tags[i], item_constants[i], queue);
        }
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
