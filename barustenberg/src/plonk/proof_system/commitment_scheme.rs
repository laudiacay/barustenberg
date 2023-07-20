use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use ark_bn254::{Fq, Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, One, Zero};

use crate::polynomials::{polynomial_arithmetic, Polynomial};
use crate::proof_system::work_queue::{Work, WorkItem, WorkQueue};
use crate::transcript::{BarretenHasher, Transcript};

use super::proving_key::ProvingKey;
use super::types::proof::CommitmentOpenProof;
use super::verification_key::VerificationKey;

/// A polynomial commitment scheme defined over two FieldExts, a group, a hash function.
/// kate commitments are one example
pub(crate) trait CommitmentScheme {
    type Fq: Field + FftField;
    type Fr: Field + FftField;
    type Group: AffineRepr;
    type Hasher: BarretenHasher;
    fn commit(
        &mut self,
        coefficients: Arc<RwLock<Polynomial<Self::Fr>>>,
        tag: String,
        item_constant: Self::Fr,
        queue: &mut WorkQueue<Self::Hasher>,
    );

    fn compute_opening_polynomial(
        &self,
        src: &[Self::Fr],
        dest: &mut [Self::Fr],
        z: &Self::Fr,
        n: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn generic_batch_open(
        &mut self,
        src: &[Self::Fr],
        dest: Arc<RwLock<Polynomial<Self::Fr>>>,
        num_polynomials: usize,
        z_points: &[Self::Fr],
        num_z_points: usize,
        challenges: &[Self::Fr],
        n: usize,
        tags: &[String],
        item_constants: &[Self::Fr],
        queue: &mut WorkQueue<Self::Hasher>,
    );

    fn batch_open(
        &mut self,
        transcript: &Transcript<Self::Hasher>,
        queue: &mut WorkQueue<Self::Hasher>,
        input_key: Option<Arc<RwLock<ProvingKey<Self::Fr>>>>,
    );

    fn batch_verify(
        &self,
        transcript: &Transcript<Self::Hasher>,
        kate_g1_elements: &mut HashMap<String, Self::Group>,
        kate_fr_elements: &mut HashMap<String, Self::Fr>,
        input_key: Option<&VerificationKey<Self::Fr>>,
    );

    fn add_opening_evaluations_to_transcript(
        &self,
        transcript: &mut Transcript<Self::Hasher>,
        input_key: Option<Arc<RwLock<ProvingKey<Self::Fr>>>>,
        in_lagrange_form: bool,
    );
}

#[derive(Default, Debug)]
pub(crate) struct KateCommitmentScheme<
    H: BarretenHasher,
    Fq: Field + FftField,
    Fr: Field + FftField,
    G: AffineRepr,
> {
    _kate_open_proof: CommitmentOpenProof,
    phantom: PhantomData<(H, Fr, G, Fq)>,
}

impl<H: BarretenHasher, Fq: Field + FftField, Fr: Field + FftField, G: AffineRepr>
    KateCommitmentScheme<H, Fq, Fr, G>
{
    pub(crate) fn new() -> Self {
        Self {
            _kate_open_proof: CommitmentOpenProof::default(),
            phantom: PhantomData,
        }
    }
}

impl<H: BarretenHasher> CommitmentScheme for KateCommitmentScheme<H, Fq, Fr, G1Affine> {
    type Fq = Fq;
    type Fr = Fr;
    type Group = G1Affine;
    type Hasher = H;

    fn commit(
        &mut self,
        coefficients: Arc<RwLock<Polynomial<Fr>>>,
        tag: String,
        item_constant: Fr,
        queue: &mut WorkQueue<H>,
    ) {
        queue.add_to_queue(WorkItem {
            work: Work::ScalarMultiplication {
                mul_scalars: coefficients,
                constant: item_constant,
            },
            tag,
        })
    }

    fn add_opening_evaluations_to_transcript(
        &self,
        _transcript: &mut Transcript<H>,
        _input_key: Option<Arc<RwLock<ProvingKey<Self::Fr>>>>,
        _in_lagrange_form: bool,
    ) {
        todo!()
    }

    fn compute_opening_polynomial(&self, _src: &[Fr], _dest: &mut [Fr], _z: &Fr, _n: usize) {
        todo!()
    }

    fn generic_batch_open(
        &mut self,
        src: &[Fr],
        dest: Arc<RwLock<Polynomial<Fr>>>,
        num_polynomials: usize,
        z_points: &[Fr],
        num_z_points: usize,
        challenges: &[Fr],
        n: usize,
        tags: &[String],
        item_constants: &[Fr],
        queue: &mut WorkQueue<H>,
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
            {
                let mut dest_mut = dest.write().unwrap();
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
                    dest_mut[dest_offset + k] = coeff_sum;
                }

                // evaluation of the i-th linear combination polynomial F_i(X) at z
                let d_i_eval =
                    polynomial_arithmetic::evaluate(&dest_mut[dest_offset..], &z_points[i], n);

                // compute coefficients of h_i(X) = (F_i(X) - F_i(z))/(X - z) as done in the previous function
                dest_mut[dest_offset] -= d_i_eval;
                dest_mut[dest_offset] *= divisor;
                for k in 1..n {
                    let sub = dest_mut[dest_offset + k - 1];
                    dest_mut[dest_offset + k] -= sub;
                    dest_mut[dest_offset + k] *= divisor;
                }
            }
            // commit to the i-th opened polynomial
            Self::commit(
                //<KateCommitmentScheme<H, Fq, Fr, G, S> as CommitmentScheme>::commit(
                self,
                dest.clone(),
                tags[i].clone(),
                item_constants[i],
                queue,
            );
        }
    }

    fn batch_open(
        &mut self,
        _transcript: &Transcript<H>,
        _queue: &mut WorkQueue<H>,
        _input_key: Option<Arc<RwLock<ProvingKey<Self::Fr>>>>,
    ) {
        todo!()
    }

    fn batch_verify(
        &self,
        _transcript: &Transcript<H>,
        _kate_g1_elements: &mut HashMap<String, G1Affine>,
        _kate_fr_elements: &mut HashMap<String, Fr>,
        _input_key: Option<&VerificationKey<Fr>>,
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
