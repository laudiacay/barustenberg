use crate::ecc::curves::coset_generator;
use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::plonk::proof_system::public_inputs::compute_public_input_delta;
use crate::plonk::proof_system::verification_key::VerificationKey;
use crate::plonk::proof_system::widgets::random_widgets::random_widget::ProverRandomWidget;
use crate::polynomials::Polynomial;
use crate::proof_system::work_queue::Work::{Fft, ScalarMultiplication};
use crate::proof_system::work_queue::{WorkItem, WorkQueue};
use crate::transcript::{BarretenHasher, Transcript};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, One, UniformRand, Zero};

use ark_bn254::{Fr, G1Affine};
use rand::thread_rng;
use rayon::prelude::*;

use anyhow::Result;

pub(crate) struct VerifierPermutationWidget<
    H: BarretenHasher,
    F: Field + FftField,
    Group: AffineRepr,
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
> {
    transcript: Transcript<H>,
    phantom: PhantomData<(F, Group)>,
}

impl<H, F, G1Affine, const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize>
    VerifierPermutationWidget<H, F, G1Affine, NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL>
where
    H: BarretenHasher,
    F: Field + FftField,
    G1Affine: AffineRepr,
{
    pub(crate) fn new() -> Self {
        Self {
            transcript: Transcript::<H>::default(),
            phantom: PhantomData,
        }
    }

    pub(crate) fn compute_quotient_evaluation_contribution(
        key: &VerificationKey<F>,
        alpha: F,
        transcript: &Transcript<H>,
        quotient_numerator_eval: &mut F,
        idpolys: Option<bool>,
    ) -> F {
        let idpolys = idpolys.unwrap_or(false);
        let alpha_squared: F = alpha.square();
        let alpha_cubed = alpha_squared * alpha;
        // a.k.a. zeta or ʓ
        let z: F = transcript.get_challenge_field_element("z", None);
        let beta: F = transcript.get_challenge_field_element("beta", Some(0));
        let gamma: F = transcript.get_challenge_field_element("beta", Some(1));
        let z_beta: F = z * beta;

        // We need wire polynomials' and sigma polynomials' evaluations at zeta which we fetch from the transcript.
        // Fetch a_eval, b_eval, c_eval, sigma1_eval, sigma2_eval
        let mut wire_evaluations = Vec::<F>::new();
        let mut sigma_evaluations = Vec::<F>::new();

        for i in 0..key.program_width {
            let index = (i + 1).to_string();
            // S_σ_i(ʓ)
            sigma_evaluations
                .push(transcript.get_field_element(format!("sigma_{}", &index).as_str()));
        }

        for i in 0..key.program_width {
            // w_i(ʓ)
            // (Note: in the Plonk paper, these polys are called a, b, c. We interchangeably call
            // them a,b,c or w_l, w_r, w_o, or w_1, w_2, w_3,... depending on the context).
            wire_evaluations.push(transcript.get_field_element(format!("w_{}", i + 1).as_str()));
        }

        // Compute evaluations of lagrange polynomials L_1(X) and L_{n-k} at ʓ.
        // Recall, k is the number of roots cut out of the vanishing polynomial Z_H(X), to yield Z_H*(X).
        // Note that
        //                                  X^n - 1
        // L_i(X) = L_1(X.ω^{-i + 1}) = -----------------
        //                               X.ω^{-i + 1} - 1
        //
        // ʓ^n - 1
        let mut numerator = key.z_pow_n - F::one();
        numerator *= key.domain.domain_inverse;
        // [ʓ^n - 1] / [n.(ʓ - 1)] =: L_1(ʓ)
        let l_start: F = numerator / (z - F::one());

        // Compute ω^{num_roots_cut_out_of_vanishing_polynomial + 1}
        let mut l_end_root = if NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL & 1 != 0 {
            key.domain.root.square()
        } else {
            key.domain.root
        };
        for _ in 0..NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL / 2 {
            l_end_root *= key.domain.root.square();
        }
        // [ʓ^n - 1] / [n.(ʓ.ω^{k+1} - 1)] =: L_{n-k}(ʓ)
        let l_end: F = numerator / ((z * l_end_root) - F::one());

        let z_1_shifted_eval: F = transcript.get_field_element("z_perm_omega");

        // Recall that the full quotient numerator is the polynomial
        // t(X) =
        //         [   a(X).b(X).qm(X) + a(X).ql(X) + b(X).qr(X) + c(X).qo(X) + qc(X) ]
        //   +   α.[
        //             [ a(X) + β.X + γ)(b(X) + β.k_1.X + γ)(c(X) + β.k_2.X + γ).z(X) ]
        //           - [ a(X) + β.Sσ1(X) + γ)(b(X) + β.Sσ2(X) + γ)(c(X) + β.Sσ3(X) + γ).z(X.ω) ]
        //         ]
        //   + α^3.[ (z(X) - 1).L_1(X) ]
        //   + α^2.[ (z(X.ω) - ∆_PI).L_{n-k}(X) ]
        //
        // This function computes the copy constraint pair, i.e., the sum of the α^1, α^2 and α^3 terms.

        // Part 1: compute the sigma contribution, i.e.
        //
        // sigma_contribution = (a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ)(c_eval + γ).z(ʓ.ω).α
        //
        let mut sigma_contribution = F::one();
        let mut t0;
        let mut t1: F;
        for i in 0..key.program_width - 1 {
            t0 = sigma_evaluations[i] * beta;
            t1 = wire_evaluations[i] + gamma;
            t0 += t1;
            sigma_contribution *= t0;
        }

        t0 = wire_evaluations[key.program_width - 1] + gamma;
        sigma_contribution *= t0;
        sigma_contribution *= z_1_shifted_eval;
        sigma_contribution *= alpha;
        // Part 2: compute the public-inputs term, i.e.
        //
        // (z(ʓ.ω) - ∆_{PI}).L_{n-k}(ʓ).α^2
        //
        // (See the separate paper which alters the 'public inputs' component of the plonk protocol)
        let public_inputs = transcript.get_field_element_vector("public_inputs");
        let public_input_delta: F =
            compute_public_input_delta(&public_inputs, beta, gamma, key.domain.root);

        t1 = (z_1_shifted_eval - public_input_delta) * l_end * alpha_squared;
        // Part 3: compute starting lagrange polynomial term, i.e.
        //
        // L_1(ʓ).α^3
        //
        let t2: F = l_start * alpha_cubed;

        // Combine parts 1, 2, 3.
        //  quotient_numerator_eval =
        //         α^2.(z(ʓ.ω) - ∆_{PI}).L_{n-k}(ʓ)
        //       - α^3.L_1(ʓ)
        //       - α.(a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ)(c_eval + γ).z(ʓ.ω)
        //

        t1 -= t2;
        t1 -= sigma_contribution;
        *quotient_numerator_eval += t1;

        // If we were using the linearization trick, we would return here. Instead we proceed to fully construct
        // the permutation part of a purported quotient numerator value.

        // Part 4: compute multiplicand of last sigma polynomial S_{sigma3}(X), i.e.
        //
        // - α.(a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ).β.z(ʓ.ω)
        //
        sigma_contribution = F::one();
        for i in 0..key.program_width - 1 {
            t0 = sigma_evaluations[i] * beta;
            t0 += wire_evaluations[i];
            t0 += gamma;
            sigma_contribution *= t0;
        }
        sigma_contribution *= z_1_shifted_eval;
        let mut sigma_last_multiplicand = -(sigma_contribution * alpha);
        sigma_last_multiplicand *= beta;

        // Add up part 4 to the  quotient_numerator_eval term
        //
        // At this intermediate stage,  quotient_numerator_eval will be:
        //
        //  quotient_numerator_eval =   α^2.(z(ʓ.ω) - ∆_{PI}).L_{n-k}(ʓ) |
        //       - α^3.L_1(ʓ)                                                                            |->
        //       quotient_numerator_eval from
        //       -   α.(a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ)(c_eval + γ).z(ʓ.ω)       |   before
        //
        //       -   α.(a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ).β.z(ʓ.ω).S_{sigma3}(ʓ)
        //                                                                               ^^^^^^^^^^^^^
        //                                                                               Evaluated at X=ʓ
        //     =   α^2.(z(ʓ.ω) - ∆_{PI}).L_{n-k}(ʓ)
        //       - α^3.L_1(ʓ)
        //       -   α.(a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ)(c_eval + β.S_{sigma3}(ʓ) + γ).z(ʓ.ω)
        //
        *quotient_numerator_eval +=
            sigma_last_multiplicand * sigma_evaluations[key.program_width - 1];

        let z_eval: F = transcript.get_field_element("z_perm");
        if idpolys {
            // Part 5.1: If idpolys = true, it indicates that we are not using the identity polynomials to
            // represent identity permutations. In that case, we need to use the pre-defined values for
            // representing identity permutations and then compute the coefficient of the z(X) component of r(X):
            //
            // [
            //       α.(a_eval + β.id_1 + γ)(b_eval + β.id_2 + γ)(c_eval + β.id_3 + γ)
            //   + α^3.L_1(ʓ)
            // ].z(X)
            //

            let mut id_contribution = F::one();
            for (i, eval_i) in wire_evaluations.iter().enumerate().take(key.program_width) {
                let id_evaluation: F =
                    transcript.get_field_element(format!("id_{}", i + 1).as_str());
                t0 = id_evaluation * beta;
                t0 += eval_i;
                t0 += gamma;
                id_contribution *= t0;
            }
            let mut id_last_multiplicand = id_contribution * alpha;
            t0 = l_start * alpha_cubed;
            id_last_multiplicand += t0;

            // Add up part 5.1 to the  quotient_numerator_eval term, so  quotient_numerator_eval will be:
            //
            //  quotient_numerator_eval = α^2.(z(ʓ.ω) - ∆_{PI}).L_{n-k}(ʓ)
            //     - α^3.L_1(ʓ)
            //     -   α.(a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ)(c_eval + β.S_{sigma3}(ʓ) + γ).z(ʓ.ω)
            //     + [
            //             α.(a_eval + β.id_1 + γ)(b_eval + β.id_2 + γ)(c_eval + β.id_3 + γ)
            //         + α^3.L_1(ʓ)
            //       ].z(ʓ)
            //         ^^^^
            //         Evaluated at X=ʓ

            *quotient_numerator_eval += id_last_multiplicand * z_eval;
        } else {
            // Part 5.2: If idpolys is false, the identity permutations are identity polynomials.
            // So we need to compute the following term
            //
            // [
            //       α.(a_eval + β.ʓ + γ)(b_eval + β.k_1.ʓ + γ)(c_eval + β.k_2.ʓ + γ)
            //   + α^3.L_1(ʓ)
            // ].z(ʓ)
            //
            let mut z_contribution = F::one();
            for (i, eval_i) in wire_evaluations.iter().enumerate().take(key.program_width) {
                let coset_generator = if i == 0 {
                    F::one()
                } else {
                    coset_generator(i - 1)
                };
                t0 = z_beta * coset_generator;
                t0 += eval_i;
                t0 += gamma;
                z_contribution *= t0;
            }
            let mut z_1_multiplicand = z_contribution * alpha;
            t0 = l_start * alpha_cubed;
            z_1_multiplicand += t0;

            // add up part 5.2 to the  quotient_numerator_eval term
            *quotient_numerator_eval += z_1_multiplicand * z_eval;
        }
        alpha_squared.square()
    }

    pub(crate) fn append_scalar_multiplication_inputs(
        alpha_base: F,
        transcript: &Transcript<H>,
    ) -> F {
        let alpha_step: F = transcript.get_challenge_field_element("alpha", None);
        alpha_base * alpha_step.square() * alpha_step
    }
}

#[derive(Debug)]
pub(crate) struct ProverPermutationWidget<
    Hash: BarretenHasher + Sync + Send,
    const PROGRAM_WIDTH: usize,
    const IDPOLYS: bool,
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
> {
    pub(crate) key: Arc<RwLock<ProvingKey<ark_bn254::Fr>>>,
    phantom: PhantomData<Hash>,
}

impl<
        Hash: BarretenHasher + Sync + Send,
        const PROGRAM_WIDTH: usize,
        const IDPOLYS: bool,
        const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
    > ProverRandomWidget
    for ProverPermutationWidget<
        Hash,
        PROGRAM_WIDTH,
        IDPOLYS,
        NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL,
    >
{
    type Hasher = Hash;

    type Fr = Fr;
    type G1 = G1Affine;

    /// Computes the permutation polynomial z(X)
    /// Commits to z(X)
    /// Computes & stores the coset form of z(X) for later use in quotient polynomial calculation.
    fn compute_round_commitments(
        &self,
        transcript: &mut Transcript<Hash>,
        round_number: usize,
        work_queue: &mut WorkQueue<Hash>,
    ) -> Result<()> {
        if round_number != 3 {
            return Ok(());
        }

        // Allocate scratch space in memory for computation of lagrange form of permutation polynomial
        // 'z_perm'. Elements 2,...,n of z_perm are constructed in place in accumulators[0]. (The first
        // element of z_perm is one, i.e. z_perm[0] == 1). The remaining accumulators are used only as scratch
        // space. All memory allocated for the accumulators is freed before termination of this function.
        let num_accumulators = if PROGRAM_WIDTH == 1 {
            3
        } else {
            PROGRAM_WIDTH * 2
        };

        let circuit_size = (*self.key).read().unwrap().circuit_size;
        let accumulators: Vec<Vec<Fr>> = vec![vec![Fr::zero(); circuit_size]; num_accumulators];

        let beta: Fr = transcript.get_challenge_field_element("beta", None);
        let gamma: Fr = transcript.get_challenge_field_element("beta", Some(1));

        let mut lagrange_base_wires: Vec<Arc<RwLock<Polynomial<Fr>>>> = Vec::new();
        let mut lagrange_base_sigmas: Vec<Arc<RwLock<Polynomial<Fr>>>> = Vec::new();
        let mut lagrange_base_ids: Vec<Arc<RwLock<Polynomial<Fr>>>> = Vec::new();

        for i in 0..PROGRAM_WIDTH {
            lagrange_base_wires.push(
                (*self.key)
                    .read()
                    .unwrap()
                    .polynomial_store
                    .get(&format!("w_{}_lagrange", i + 1))?,
            );
            lagrange_base_sigmas.push(
                (*self.key)
                    .read()
                    .unwrap()
                    .polynomial_store
                    .get(&format!("sigma_{}_lagrange", i + 1))?,
            );

            // If idpolys = true, it implies that we do NOT use the identity permutation
            // S_ID1(X) = X, S_ID2(X) = k_1X, S_ID3(X) = k_2X.
            if IDPOLYS {
                lagrange_base_ids.push(
                    (*self.key)
                        .read()
                        .unwrap()
                        .polynomial_store
                        .get(&format!("id_{}_lagrange", i + 1))?,
                );
            }
        }

        // When we write w_i it means the evaluation of witness polynomial at i-th index.
        // When we write w^{i} it means the generator of the subgroup to the i-th power.
        //
        // step 1: compute the individual terms in the permutation poylnomial.
        //
        // Consider the case in which we use identity permutation polynomials and let program width = 3.
        // (extending it to the case when the permutation polynomials is not identity is trivial).
        //
        // coefficient of L_1: 1
        //
        // coefficient of L_2:
        //
        //  coeff_of_L1 *   (w_1 + γ + β.ω^{0}) . (w_{n+1} + γ + β.k_1.ω^{0}) . (w_{2n+1} + γ + β.k_2.ω^{0})
        //                  ---------------------------------------------------------------------------------
        //                  (w_1 + γ + β.σ(1) ) . (w_{n+1} + γ + β.σ(n+1)   ) . (w_{2n+1} + γ + β.σ(2n+1)  )
        //
        // coefficient of L_3:
        //
        //  coeff_of_L2 *   (w_2 + γ + β.ω^{1}) . (w_{n+2} + γ + β.k_1.ω^{1}) . (w_{2n+2} + γ + β.k_2.ω^{1})
        //                  --------------------------------------------------------------------------------
        //                  (w_2 + γ + β.σ(2) ) . (w_{n+2} + γ + β.σ(n+2)   ) . (w_{2n+2} + γ + β.σ(2n+2)  )
        // and so on...
        //
        // accumulator data structure:
        // numerators are stored in accumulator[0: program_width-1],
        // denominators are stored in accumulator[program_width:]
        //
        //      0                                1                                      (n-1)
        // 0 -> (w_1      + γ + β.ω^{0}    ),    (w_2      + γ + β.ω^{1}    ),    ...., (w_n      + γ + β.ω^{n-1}    )
        // 1 -> (w_{n+1}  + γ + β.k_1.ω^{0}),    (w_{n+1}  + γ + β.k_1.ω^{2}),    ...., (w_{n+1}  + γ + β.k_1.ω^{n-1})
        // 2 -> (w_{2n+1} + γ + β.k_2.ω^{0}),    (w_{2n+1} + γ + β.k_2.ω^{0}),    ...., (w_{2n+1} + γ + β.k_2.ω^{n-1})
        //
        // 3 -> (w_1      + γ + β.σ(1)     ),    (w_2      + γ + β.σ(2)     ),    ...., (w_n      + γ + β.σ(n)       )
        // 4 -> (w_{n+1}  + γ + β.σ(n+1)   ),    (w_{n+1}  + γ + β.σ{n+2}   ),    ...., (w_{n+1}  + γ + β.σ{n+n}     )
        // 5 -> (w_{2n+1} + γ + β.σ(2n+1)  ),    (w_{2n+1} + γ + β.σ(2n+2)  ),    ...., (w_{2n+1} + γ + β.σ(2n+n)    )
        //
        // Thus, to obtain coefficient_of_L2, we need to use accumulators[:][0]:
        //    acc[0][0]*acc[1][0]*acc[2][0] / acc[program_width][0]*acc[program_width+1][0]*acc[program_width+2][0]
        //
        // To obtain coefficient_of_L3, we need to use accumulator[:][0] and accumulator[:][1]
        // and so on upto coefficient_of_Ln.

        // Recall: In a domain: num_threads * thread_size = size (= subgroup_size)
        //        |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 | <-- n = 16
        //    j:  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    | num_threads = 8
        //    i:     0    1    0    1    0    1    0    1    0    1    0    1    0    1    0    1   thread_size = 2
        // So i will access a different element from 0..(n-1) each time.
        // Commented maths notation mirrors the indexing from the giant comment immediately above.
        // Assuming `small_domain_thread_size` and `small_domain_num_threads` are usize fields in the key
        let small_domain = &self.key.read().unwrap().small_domain;

        let accumulators = Arc::new(RwLock::new(accumulators));

        // TODO this is the worst global mutex ever- needs cleaning seriously.

        (0..small_domain.num_threads).into_par_iter().for_each(|j| {
            let thread_root = small_domain
                .root
                .pow([(j * small_domain.thread_size) as u64]); // effectively ω^{i} in inner loop
            let mut cur_root_times_beta: Fr = thread_root * beta; // β.ω^{i}
            let mut t0;
            let mut wire_plus_gamma;
            let start = j * small_domain.thread_size;
            let end = (j + 1) * small_domain.thread_size;

            for i in start..end {
                wire_plus_gamma = gamma + lagrange_base_wires[0].read().unwrap()[i]; // w_{i + 1} + γ

                if !IDPOLYS {
                    accumulators.write().unwrap()[0][i] = wire_plus_gamma + cur_root_times_beta;
                    // w_{i + 1} + γ + β.ω^{i}
                }
                if IDPOLYS {
                    t0 = lagrange_base_ids[0].read().unwrap()[i] * beta; // β.id(i + 1)
                    accumulators.write().unwrap()[0][i] = t0 + wire_plus_gamma; // w_{i + 1} + γ + β.id(i + 1)
                }

                t0 = lagrange_base_sigmas[0].read().unwrap()[i] * beta; // β.σ(i + 1)
                accumulators.write().unwrap()[PROGRAM_WIDTH][i] = t0 + wire_plus_gamma; // w_{i + 1} + γ + β.σ(i + 1)

                for k in 1..PROGRAM_WIDTH {
                    wire_plus_gamma = gamma + lagrange_base_wires[k].read().unwrap()[i]; // w_{k.n + i + 1} + γ

                    if IDPOLYS {
                        t0 = lagrange_base_ids[k].read().unwrap()[i] * beta; // β.id(k.n + i + 1)
                    } else {
                        t0 = coset_generator::<Fr>(k - 1) * cur_root_times_beta;
                        // β.k_{k}.ω^{i}
                    }
                    accumulators.write().unwrap()[k][i] = t0 + wire_plus_gamma; // w_{k.n + i + 1} + γ + β.id(k.n + i + 1)

                    t0 = lagrange_base_sigmas[k].read().unwrap()[i] * beta; // β.σ(k.n + i + 1)
                    accumulators.write().unwrap()[k + PROGRAM_WIDTH][i] = t0 + wire_plus_gamma;
                    // w_{k.n + i + 1} + γ + β.σ(k.n + i + 1)
                }
                if !IDPOLYS {
                    cur_root_times_beta *= small_domain.root; // β.ω^{i + 1}
                }
            }
        });
        // Step 2: compute the constituent components of z(X). This is a small multithreading bottleneck, as we have
        // program_width * 2 non-parallelizable processes
        //
        // Update the accumulator matrix a[:][:] to contain the left products like so:
        //      0           1                     2                          (n-1)
        // 0 -> (a[0][0]),  (a[0][1] * a[0][0]),  (a[0][2] * a[0][1]), ...,  (a[0][n-1] * a[0][n-2])
        // 1 -> (a[1][0]),  (a[1][1] * a[1][0]),  (a[1][2] * a[1][1]), ...,  (a[1][n-1] * a[1][n-2])
        // 2 -> (a[2][0]),  (a[2][1] * a[2][0]),  (a[2][2] * a[2][1]), ...,  (a[2][n-1] * a[2][n-2])
        //
        // 3 -> (a[3][0]),  (a[3][1] * a[3][0]),  (a[3][2] * a[3][1]), ...,  (a[3][n-1] * a[3][n-2])
        // 4 -> (a[4][0]),  (a[4][1] * a[4][0]),  (a[4][2] * a[4][1]), ...,  (a[4][n-1] * a[4][n-2])
        // 5 -> (a[5][0]),  (a[5][1] * a[5][0]),  (a[5][2] * a[5][1]), ...,  (a[5][n-1] * a[5][n-2])
        //
        // and so on...

        (0..(PROGRAM_WIDTH * 2)).into_par_iter().for_each(|i| {
            // TODO big lock...
            let coeffs = &mut accumulators.write().unwrap()[i];
            for j in 0..(small_domain.size - 1) {
                let coeffs_j = coeffs[j];
                coeffs[j + 1] *= coeffs_j; // iteratively update elements in subsequent columns
            }
        });

        // step 3: concatenate together the accumulator elements into z(X)
        //
        // Update each element of the accumulator row a[0] to be the product of itself with the 'numerator' rows beneath
        // it, and update each element of a[program_width] to be the product of itself with the 'denominator' rows
        // beneath it.
        //
        //       0                                     1                                           (n-1)
        // 0 ->  (a[0][0] * a[1][0] * a[2][0]),        (a[0][1] * a[1][1] * a[2][1]),        ...., (a[0][n-1] *
        // a[1][n-1] * a[2][n-1])
        //
        // pw -> (a[pw][0] * a[pw+1][0] * a[pw+2][0]), (a[pw][1] * a[pw+1][1] * a[pw+2][1]), ...., (a[pw][n-1] *
        // a[pw+1][n-1] * a[pw+2][n-1])
        //
        // Note that pw = program_width
        //
        // Hereafter, we can compute
        // coefficient_Lj = a[0][j]/a[pw][j]
        //
        // Naive way of computing these coefficients would result in n inversions, which is pretty expensive.
        // Instead we use Montgomery's trick for batch inversion.
        // Montgomery's trick documentation:
        // ./src/barretenberg/ecc/curves/bn254/scalar_multiplication/scalar_multiplication.hpp/L286

        (0..small_domain.num_threads).into_par_iter().for_each(|j| {
            let start = j * small_domain.thread_size;
            let end = ((j + 1) * small_domain.thread_size)
                - if j == small_domain.num_threads - 1 {
                    1
                } else {
                    0
                };

            let mut inversion_accumulator = Fr::one();
            let inversion_index = if PROGRAM_WIDTH == 1 {
                2
            } else {
                PROGRAM_WIDTH * 2 - 1
            };

            for i in start..end {
                let mut accumulators = accumulators.write().unwrap();
                for k in 1..PROGRAM_WIDTH {
                    let ac_k_i = accumulators[k][i];
                    accumulators[0][i] *= ac_k_i;
                    let ac_pwk_i = accumulators[PROGRAM_WIDTH + k][i];
                    accumulators[PROGRAM_WIDTH][i] *= ac_pwk_i;
                }
                accumulators[inversion_index][i] = accumulators[0][i] * inversion_accumulator;
                inversion_accumulator *= accumulators[PROGRAM_WIDTH][i];
            }

            // todo is this unwrap ok? not sure
            inversion_accumulator = inversion_accumulator.inverse().unwrap();

            (start..end).rev().for_each(|i| {
                let acc_inv = accumulators.read().unwrap()[inversion_index][i];
                accumulators.write().unwrap()[0][i] = inversion_accumulator * acc_inv;
                inversion_accumulator *= accumulators.read().unwrap()[PROGRAM_WIDTH][i];
            });
        });

        // Construct permutation polynomial 'z' in lagrange form as:
        // z = [1 accumulators[0][0] accumulators[0][1] ... accumulators[0][n-2]]
        let mut z_perm = Polynomial::new(circuit_size);
        z_perm[0] = Fr::one();

        // Use `copy_from_slice` for copying elements from one vector to another
        z_perm[1..circuit_size]
            .copy_from_slice(&accumulators.read().unwrap()[0][..circuit_size - 1]);

        /*
        Adding zero knowledge to the permutation polynomial.
        */
        // To ensure that PLONK is honest-verifier zero-knowledge, we need to ensure that the witness polynomials
        // and the permutation polynomial look uniformly random to an adversary. To make the witness polynomials
        // a(X), b(X) and c(X) uniformly random, we need to add 2 random blinding factors into each of them.
        // i.e. a'(X) = a(X) + (r_1X + r_2)
        // where r_1 and r_2 are uniformly random scalar field elements. A natural question is:
        // Why do we need 2 random scalars in witness polynomials? The reason is: our witness polynomials are
        // evaluated at only 1 point (\scripted{z}), so adding a random degree-1 polynomial suffices.
        //
        // NOTE: In TurboPlonk and UltraPlonk, the witness polynomials are evaluated at 2 points and thus
        // we need to add 3 random scalars in them.
        //
        // On the other hand, permutation polynomial z(X) is evaluated at two points, namely \scripted{z} and
        // \scripted{z}.\omega. Hence, we need to add a random polynomial of degree 2 to ensure that the permutation
        // polynomial looks uniformly random.
        // z'(X) = z(X) + (r_3.X^2 + r_4.X + r_5)
        // where r_3, r_4, r_5 are uniformly random scalar field elements.
        //
        // Furthermore, instead of adding random polynomials, we could directly add random scalars in the lagrange-
        // basis forms of the witness and permutation polynomials. This is because we are using a modified vanishing
        // polynomial of the form
        //                           (X^n - 1)
        // Z*_H(X) = ------------------------------------------
        //           (X - ω^{n-1}).(X - ω^{n-2})...(X - ω^{n-k})
        // where ω = n-th root of unity, k = num_roots_cut_out_of_vanishing_polynomials.
        // Thus, the last k places in the lagrange basis form of z(X) are empty. We can therefore utilise them and
        // add random scalars as coefficients of L_{n-1}, L_{n-2},... and so on.
        //
        // Note: The number of coefficients in the permutation polynomial z(X) is (n - k + 1) DOCTODO: elaborate on why.
        // (refer to Round 2 in the PLONK paper). Hence, if we cut 3 roots out of the vanishing polynomial,
        // we are left with only 2 places (coefficients) in the z array to add randomness. To have the last 3 places
        // available for adding random scalars, we therefore need to cut at least 4 roots out of the vanishing polynomial.
        //
        // Since we have valid z coefficients in positions from 0 to (n - k), we can start adding random scalars
        // from position (n - k + 1) upto (n - k + 3).
        //
        // NOTE: If in future there is a need to cut off more zeros off the vanishing polynomial, this method
        // will not change. This must be changed only if the number of evaluations of permutation polynomial
        // changes.

        let z_randomness = 3;
        assert!(z_randomness < NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL);
        let mut rng = thread_rng();
        for k in 0..z_randomness {
            z_perm[(circuit_size - NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL) + 1 + k] =
                Fr::rand(&mut rng);
        }

        small_domain.ifft_inplace(&mut z_perm.coefficients);

        let z_perm = Arc::new(RwLock::new(z_perm));

        // Commit to z:
        work_queue.add_to_queue(WorkItem {
            work: ScalarMultiplication {
                constant: Fr::from(circuit_size as i64),
                mul_scalars: z_perm.clone(),
            },
            tag: "Z_PERM".to_string(),
        });

        // Compute coset-form of z:
        work_queue.add_to_queue(WorkItem {
            work: Fft { index: 0 },
            tag: "z_perm".to_string(),
        });

        (*self.key)
            .write()
            .unwrap()
            .polynomial_store
            .put_owned("z_perm".to_string(), z_perm);

        Ok(())
    }

    fn compute_quotient_contribution(
        &self,
        alpha_base: Fr,
        transcript: &Transcript<Hash>,
    ) -> Result<Fr> {
        let z_perm_fft = (*self.key)
            .read()
            .unwrap()
            .polynomial_store
            .get(&"z_perm_fft".to_string())?;

        let alpha_squared = alpha_base.square();
        let beta = transcript.get_challenge_field_element("beta", None);
        let gamma = transcript.get_challenge_field_element("beta", Some(1));

        // Initialise the (n + 1)th coefficients of quotient parts so that reuse of proving
        // keys does not use some residual data from another proof.a
        let circuit_size = (*self.key).read().unwrap().circuit_size;
        (*(*self.key).read().unwrap().quotient_polynomial_parts[0])
            .write()
            .unwrap()[circuit_size] = Fr::zero();
        (*(*self.key).read().unwrap().quotient_polynomial_parts[1])
            .write()
            .unwrap()[circuit_size] = Fr::zero();
        (*(*self.key).read().unwrap().quotient_polynomial_parts[2])
            .write()
            .unwrap()[circuit_size] = Fr::zero();

        // Information about the permutation check is available in the comments in the original C++ code

        let mut wire_ffts: Vec<Arc<RwLock<Polynomial<Fr>>>> = Vec::new();
        let mut sigma_ffts: Vec<Arc<RwLock<Polynomial<Fr>>>> = Vec::new();
        let mut id_ffts: Vec<Arc<RwLock<Polynomial<Fr>>>> = Vec::new();

        for i in 0..PROGRAM_WIDTH {
            // wire_fft[0] contains the fft of the wire polynomial w_1
            // sigma_fft[0] contains the fft of the permutation selector polynomial \sigma_1
            wire_ffts.push(
                (*self.key)
                    .read()
                    .unwrap()
                    .polynomial_store
                    .get(&format!("w_{}_fft", i + 1))?,
            );
            sigma_ffts.push(
                (*self.key)
                    .read()
                    .unwrap()
                    .polynomial_store
                    .get(&format!("sigma_{}_fft", i + 1))?,
            );

            // idpolys is FALSE iff the "identity permutation" is used as a monomial
            // as a part of the permutation polynomial
            // <=> idpolys = FALSE
            if IDPOLYS {
                id_ffts.push(
                    (*self.key)
                        .read()
                        .unwrap()
                        .polynomial_store
                        .get(&format!("id_{}_fft", i + 1))?,
                );
            }
        }

        // we start with lagrange polynomial L_1(X)
        let l_start = (*self.key)
            .read()
            .unwrap()
            .polynomial_store
            .get(&"lagrange_1_fft".to_string())?;

        // Compute our public input component
        let public_inputs: Vec<Fr> = transcript.get_field_element_vector("public_inputs");

        let public_input_delta = compute_public_input_delta(
            &public_inputs,
            beta,
            gamma,
            (*self.key).read().unwrap().small_domain.root,
        );

        let block_mask = (*self.key).read().unwrap().large_domain.size - 1;
        // Step 4: Set the quotient polynomial to be equal to
        let large_domain_num_threads = (*self.key).read().unwrap().large_domain.num_threads;
        let large_domain_thread_size = (*self.key).read().unwrap().large_domain.thread_size;
        let large_domain_root = (*self.key).read().unwrap().large_domain.root;
        let small_domain_log2_size = (*self.key).read().unwrap().small_domain.log2_size;
        let small_domain_generator = (*self.key).read().unwrap().small_domain.generator;
        (0..large_domain_num_threads).into_par_iter().for_each(|j| {
            let start = j * large_domain_thread_size;
            let end = (j + 1) * large_domain_thread_size;

            // Leverage multi-threading by computing quotient polynomial at points
            // (ω^{j * num_threads}, ω^{j * num_threads + 1}, ..., ω^{j * num_threads + num_threads})
            //
            // curr_root = ω^{j * num_threads} * g_{small} * β
            // curr_root will be used in denominator
            let mut cur_root_times_beta =
                large_domain_root.pow([(j * large_domain_thread_size) as u64]);
            cur_root_times_beta *= small_domain_generator;
            cur_root_times_beta *= beta;

            let mut wire_plus_gamma;
            let mut t0;
            let mut denominator;
            let mut numerator;
            let wire_ffts_0 = wire_ffts[0].read().unwrap();
            let sigma_ffts_0 = sigma_ffts[0].read().unwrap();
            for i in start..end {
                wire_plus_gamma = gamma + wire_ffts_0[i];
                // Numerator computation
                if !IDPOLYS {
                    // identity polynomial used as a monomial: S_{id1} = x, S_{id2} = k_1.x, S_{id3} = k_2.x
                    // start with (w_l(X) + β.X + γ)
                    numerator = cur_root_times_beta + wire_plus_gamma;
                } else {
                    numerator = id_ffts[0].read().unwrap()[i] * beta + wire_plus_gamma;
                }

                // Denominator computation
                // start with (w_l(X) + β.σ_1(X) + γ)
                denominator = sigma_ffts_0[i] * beta;
                denominator += wire_plus_gamma;
                for k in 1..PROGRAM_WIDTH {
                    wire_plus_gamma = gamma + wire_ffts[k].read().unwrap()[i];
                    if !IDPOLYS {
                        t0 = coset_generator::<Fr>(k - 1) * cur_root_times_beta;
                    } else {
                        t0 = id_ffts[k].read().unwrap()[i] * beta;
                    }
                    t0 += wire_plus_gamma;
                    numerator *= t0;

                    // (w_r(X) + β.σ_{k}(X) + γ)
                    t0 = sigma_ffts[k].read().unwrap()[i] * beta;
                    t0 += wire_plus_gamma;
                    denominator *= t0;
                }
                numerator *= (*z_perm_fft).read().unwrap()[i];
                denominator *= (*z_perm_fft).read().unwrap()[(i + 4) & block_mask];
                /*
                 * Permutation bounds check
                 * (z(X.w) - 1).(α^3).L_{end}(X) = T(X).Z*_H(X)
                 *
                 * where Z*_H(X) = (X^n - 1)/[(X - ω^{n-1})...(X - ω^{n - num_roots_cut_out_of_vanishing_polynomial})]
                 * i.e. we remove some roots from the true vanishing polynomial to ensure that the overall degree
                 * of the permutation polynomial is <= n.
                 * Read more on this here: https://hackmd.io/1DaroFVfQwySwZPHMoMdBg
                 *
                 * Therefore, L_{end} = L_{n - num_roots_cut_out_of_vanishing_polynomial}
                 */
                // The α^3 term is so that we can subsume this polynomial into the quotient polynomial,
                // whilst ensuring the term is linearly independent form the other terms in the quotient polynomial

                // We want to verify that z(X) equals `1` when evaluated at `ω_n`, the 'last' element of our multiplicative
                // subgroup H. But PLONK's 'vanishing polynomial', Z*_H(X), isn't the true vanishing polynomial of subgroup
                // H. We need to cut a root of unity out of Z*_H(X), specifically `ω_n`, for our grand product argument.
                // When evaluating z(X) has been constructed correctly, we verify that z(X.ω).(identity permutation product)
                // = z(X).(sigma permutation product), for all X \in H. But this relationship breaks down for X = ω_n,
                // because z(X.ω) will evaluate to the *first* element of our grand product argument. The last element of
                // z(X) has a dependency on the first element, so the first element cannot have a dependency on the last
                // element.

                // TODO: With the reduction from 2 z polynomials to a single z(X), the above no longer applies
                // TODO: Fix this to remove the (z(X.ω) - 1).L_{n-1}(X) check

                // To summarise, we can't verify claims about z(X) when evaluated at `ω_n`.
                // But we can verify claims about z(X.ω) when evaluated at `ω_{n-1}`, which is the same thing

                // To summarise the summary: If z(ω_n) = 1, then (z(X.ω) - 1).L_{n-1}(X) will be divisible by Z_H*(X)
                // => add linearly independent term (z(X.ω) - 1).(α^3).L{n-1}(X) into the quotient polynomial to check
                // this

                // z_perm_fft already contains evaluations of Z(X).(\alpha^2)
                // at the (4n)'th roots of unity
                // => to get Z(X.w) instead of Z(X), index element (i+4) instead of i
                t0 = (*z_perm_fft).read().unwrap()[(i + 4) & block_mask] - public_input_delta; // T0 = (Z(X.w) - (delta)).(\alpha^2)
                t0 *= alpha_base; // T0 = (Z(X.w) - (delta)).(\alpha^3)

                // T0 = (z(X.ω) - Δ).(α^3).L_{end}
                // where L_{end} = L{n - num_roots_cut_out_of_vanishing_polynomial}.
                //
                // Note that L_j(X) = L_1(X . ω^{-j}) = L_1(X . ω^{n-j})
                // => L_{end}= L_1(X . ω^{num_roots_cut_out_of_vanishing_polynomial + 1})
                // => fetch the value at index (i + (num_roots_cut_out_of_vanishing_polynomial + 1) * 4) in l_1
                // the factor of 4 is because l_1 is a 4n-size fft.
                //
                // Recall, we use l_start for l_1 for consistency in notation.
                t0 *= (*l_start).read().unwrap()
                    [(i + 4 + 4 * NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL) & block_mask];
                numerator += t0;

                // Step 2: Compute (z(X) - 1).(α^4).L1(X)
                // We need to verify that z(X) equals `1` when evaluated at the first element of our subgroup H
                // i.e. z(X) starts at 1 and ends at 1
                // The `alpha^4` term is so that we can add this as a linearly independent term in our quotient polynomial
                t0 = (*z_perm_fft).read().unwrap()[i] - Fr::one(); // T0 = (Z(X) - 1).(\alpha^2)
                t0 *= alpha_squared; // T0 = (Z(X) - 1).(\alpha^4)
                t0 *= (*l_start).read().unwrap()[i]; // T0 = (Z(X) - 1).(\alpha^2).L1(X)
                numerator += t0;
                // Combine into quotient polynomial
                t0 = numerator - denominator;

                (*self.key).read().unwrap().quotient_polynomial_parts
                    [i >> small_domain_log2_size]
                    .write()
                    .unwrap()[i & (circuit_size - 1)] = t0 * alpha_base;

                // Update our working root of unity
                cur_root_times_beta *= large_domain_root;
            }
        });

        Ok(alpha_base.square().square())
    }
}

impl<
        Hash: BarretenHasher + Sync + Send,
        const PROGRAM_WIDTH: usize,
        const IDPOLYS: bool,
        const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
    >
    ProverPermutationWidget<Hash, PROGRAM_WIDTH, IDPOLYS, NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL>
{
    pub(crate) fn new(proving_key: Arc<RwLock<ProvingKey<Fr>>>) -> Self {
        Self {
            key: proving_key,
            phantom: PhantomData,
        }
    }
}
