use crate::ecc::curves::coset_generator;
use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::plonk::proof_system::public_inputs::compute_public_input_delta;
use crate::plonk::proof_system::verification_key::VerificationKey;
use crate::plonk::proof_system::widgets::random_widgets::random_widget::ProverRandomWidget;
use crate::proof_system::work_queue::WorkQueue;
use crate::transcript::{BarretenHasher, Transcript};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

pub(crate) struct VerifierPermutationWidget<
    H: BarretenHasher,
    F: Field,
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
        key: &Arc<VerificationKey<F>>,
        alpha: F,
        transcript: &Transcript<H>,
        quotient_numerator_eval: &mut F,
        idpolys: bool,
    ) -> F {
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
    Fr: Field + FftField,
    Hash: BarretenHasher,
    G1Affine: AffineRepr,
    const PROGRAM_WIDTH: usize,
    const IDPOLYS: bool,
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
> {
    pub(crate) key: Rc<RefCell<ProvingKey<Fr>>>,
    phantom: PhantomData<(Hash, Fr, G1Affine)>,
}

impl<
        'a,
        Fr: Field + FftField,
        Hash: BarretenHasher,
        G1Affine: AffineRepr,
        const PROGRAM_WIDTH: usize,
        const IDPOLYS: bool,
        const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
    > ProverRandomWidget
    for ProverPermutationWidget<
        Fr,
        Hash,
        G1Affine,
        PROGRAM_WIDTH,
        IDPOLYS,
        NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL,
    >
{
    type Hasher = Hash;

    type Fr = Fr;
    type G1 = G1Affine;

    fn compute_round_commitments(
        &self,
        _transcript: &mut Transcript<Hash>,
        _size: usize,
        _work_queue: &mut WorkQueue<Hash, Fr, G1Affine>,
    ) {
        todo!()
    }

    fn compute_quotient_contribution(&self, _alpha_base: Fr, _transcript: &Transcript<Hash>) -> Fr {
        todo!()
    }
}

impl<
        'a,
        Fr: Field + FftField,
        G1Affine: AffineRepr,
        Hash: BarretenHasher,
        const PROGRAM_WIDTH: usize,
        const IDPOLYS: bool,
        const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize,
    >
    ProverPermutationWidget<
        Fr,
        Hash,
        G1Affine,
        PROGRAM_WIDTH,
        IDPOLYS,
        NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL,
    >
{
    pub(crate) fn new(proving_key: Rc<RefCell<ProvingKey<Fr>>>) -> Self {
        Self {
            key: proving_key,
            phantom: PhantomData,
        }
    }

    pub(crate) fn compute_round_commitments(
        &mut self,
        _transcript: &mut Transcript<Hash>,
        _round_number: usize,
        _queue: &mut WorkQueue<Hash, Fr, G1Affine>,
    ) {
        // ...
    }

    pub(crate) fn compute_quotient_contribution(
        &self,
        _alpha_base: Fr,
        _transcript: &Transcript<Hash>,
    ) -> Fr {
        // ...
        todo!("ProverPermutationWidget::compute_quotient_contribution")
    }
}
