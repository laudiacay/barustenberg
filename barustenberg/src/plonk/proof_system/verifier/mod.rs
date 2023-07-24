use crate::{
    ecc::{
        curves::bn254_scalar_multiplication::{
            generate_pippenger_point_table, PippengerRuntimeState,
        },
        reduced_ate_pairing_batch_precomputed,
    },
    plonk::proof_system::constants::NUM_LIMB_BITS_IN_FIELD_SIMULATION,
    transcript::{BarretenHasher, Manifest, Transcript},
};

use ark_bn254::{Fq, Fq12, Fr, G1Affine, G1Projective};
use ark_ec::AffineRepr;
use ark_ec::CurveGroup;
use ark_ff::{BigInteger, Field, One, Zero};

use super::{
    commitment_scheme::{CommitmentScheme, KateCommitmentScheme},
    types::{prover_settings::Settings, Proof},
};

use std::cell::RefCell;
use std::{collections::HashMap, rc::Rc};

use super::verification_key::VerificationKey;

use anyhow::{anyhow, Result};

#[cfg(test)]
mod test;

/// Verifier struct 
#[derive(Debug)]
pub struct Verifier<H: BarretenHasher, S: Settings<Hasher = H, Field = Fr, Group = G1Affine>> {
    settings: S,
    key: Rc<RefCell<VerificationKey<Fr>>>,
    manifest: Manifest,
    kate_g1_elements: HashMap<String, G1Affine>,
    kate_fr_elements: HashMap<String, Fr>,
    pub(crate) commitment_scheme: Box<KateCommitmentScheme<H, Fq, Fr, G1Affine>>,
}

/// verifier interface 
impl<H: BarretenHasher, S: Settings<Hasher = H, Field = Fr, Group = G1Affine>> Verifier<H, S> {

    /// Constructor
    pub fn new(
        _verifier_key: Option<Rc<RefCell<VerificationKey<Fr>>>>,
        _manifest: Manifest,
    ) -> Self {
        // Implement constructor logic here.
        todo!("Verifier::new")
    }

    /// Validate commitements
    fn validate_commitments(&self) -> bool {
        // Implement validate_commitments logic here.
        todo!("Verifier::validate_commitments")
    }

    fn validate_scalars(&self) -> bool {
        // Implement validate_scalars logic here.
        todo!("Verifier::validate_scalars")
    }

    /// Verify Proof
    pub fn verify_proof(&mut self, proof: &Proof) -> Result<bool> {
        // This function verifies a PLONK proof for given program settings.
        // A PLONK proof for standard PLONK is of the form:
        //
        // π_SNARK =   { [a]_1,[b]_1,[c]_1,[z]_1,[t_{low}]_1,[t_{mid}]_1,[t_{high}]_1,[W_z]_1,[W_zω]_1 \in G,
        //                a_eval, b_eval, c_eval, sigma1_eval, sigma2_eval, sigma3_eval,
        //                  q_l_eval, q_r_eval, q_o_eval, q_m_eval, q_c_eval, z_eval_omega \in F }
        //
        // Proof π_SNARK must first be added to the transcript with the other program_settings.
        (*self.key).borrow_mut().program_width = self.settings.program_width();

        // Add the proof data to the transcript, according to the manifest. Also initialise the transcript's hash type and
        // challenge bytes.
        let mut transcript: Transcript<_> = Transcript::new_from_transcript(
            proof.proof_data.as_ref(),
            self.manifest.clone(),
            self.settings.num_challenge_bytes(),
        );

        // From the verification key, also add n & l (the circuit size and the number of public inputs) to the transcript.
        transcript.add_element(
            "circuit_size",
            (*self.key).borrow().circuit_size.to_le_bytes().to_vec(),
        );
        transcript.add_element(
            "public_input_size",
            (*self.key)
                .borrow()
                .num_public_inputs
                .to_le_bytes()
                .to_vec(),
        );

        // Compute challenges from the proof data, based on the manifest, using the Fiat-Shamir heuristic
        transcript.apply_fiat_shamir("init");
        transcript.apply_fiat_shamir("eta");
        transcript.apply_fiat_shamir("beta");
        transcript.apply_fiat_shamir("alpha");
        // `zeta` is the name being given to the "Fraktur" (gothic) z from the plonk paper, so as not to
        // confuse us with the z permutation polynomial and Z_H vanishing polynomial.
        // You could use a unicode "latin small letter ezh with curl" (ʓ) to get close, if you wanted.
        transcript.apply_fiat_shamir("z");

        // Deserialize alpha and zeta from the transcript.
        let alpha = transcript.get_challenge_field_element("alpha", None);
        let zeta = transcript.get_challenge_field_element("z", None);

        // Compute the evaluations of the lagrange polynomials L_1(X) and L_{n - k}(X) at X = ʓ.
        // Also computes the evaluation of the vanishing polynomial Z_H*(X) at X = ʓ.
        // Here k = num_roots_cut_out_of_the_vanishing_polynomial and n is the size of the evaluation domain.
        // TODO: can we add these lagrange evaluations to the transcript? They get recalcualted after this multiple times,
        // by each widget.
        let lagrange_evals = (*self.key)
            .borrow()
            .domain
            .get_lagrange_evaluations(&zeta, None);

        // Step 8: Compute quotient polynomial evaluation at zeta
        //           r_eval − (a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval + γ)(c_eval + γ).z_eval_omega.α −
        //           L_1(ʓ).α^3 + (z_eval_omega - ∆_{PI}).L_{n-k}(ʓ).α^2
        // t_eval =  -----------------------------------------------------------------------------------------------
        //                                                    Z_H*(ʓ)
        //
        // where Z_H*(X) is the modified vanishing polynomial.

        // Compute ʓ^n.
        let z_pow_n = zeta.pow([(*self.key).borrow().domain.size as u64]);
        (*self.key).borrow_mut().z_pow_n = z_pow_n;

        // compute the quotient polynomial numerator contribution
        let mut t_numerator_eval = Fr::zero();
        S::compute_quotient_evaluation_contribution(
            &(*self.key).borrow(),
            &alpha,
            &transcript,
            &mut t_numerator_eval,
        );
        let t_eval = t_numerator_eval * lagrange_evals.vanishing_poly.inverse().unwrap();
        transcript.add_field_element("t", &t_eval);

        // Compute nu and separator challenges.
        transcript.apply_fiat_shamir("nu");
        transcript.apply_fiat_shamir("separator");
        // a.k.a. `u` in the plonk paper
        let separator_challenge: Fr = transcript.get_challenge_field_element("separator", None);

        // In the following function, we do the following computation.
        // Step 10: Compute batch opening commitment [F]_1
        //          [F]  :=  [t_{low}]_1 + \zeta^{n}.[tmid]1 + \zeta^{2n}.[t_{high}]_1
        //                   + [D]_1 + \nu_{a}.[a]_1 + \nu_{b}.[b]_1 + \nu_{c}.[c]_1
        //                   + \nu_{\sigma1}.[s_{\sigma_1}]1 + \nu_{\sigma2}.[s_{\sigma_2}]1
        //
        // We do not compute [D]_1 term in this method as the information required to compute [D]_1
        // in inadequate as far as this KateCommitmentScheme class is concerned.
        //
        // Step 11: Compute batch evaluation commitment [E]_1
        //          [E]_1  :=  (t_eval + \nu_{r}.r_eval + \nu_{a}.a_eval + \nu_{b}.b_eval
        //                      \nu_{c}.c_eval + \nu_{\sigma1}.sigma1_eval + \nu_{\sigma2}.sigma2_eval +
        //                      nu_z_omega.separator.z_eval_omega) . [1]_1
        //
        // Note that we do not actually compute the scalar multiplications but just accumulate the scalars
        // and the group elements in different vectors.
        self.commitment_scheme.batch_verify(
            &transcript,
            &mut self.kate_g1_elements,
            &mut self.kate_fr_elements,
            Some(&(*self.key).borrow()),
        );

        // Step 9: Compute the partial opening batch commitment [D]_1:
        //         [D]_1 = (a_eval.b_eval.[qM]_1 + a_eval.[qL]_1 + b_eval.[qR]_1 + c_eval.[qO]_1 + [qC]_1) * nu_{linear} * α
        //         >> selector polynomials
        //                  + [(a_eval + β.z + γ)(b_eval + β.k_1.z + γ)(c_eval + β.k_2.z + γ).α +
        //                  L_1(z).α^{3}].nu_{linear}.[z]_1 >> grand product perm polynomial
        //                  - (a_eval + β.sigma1_eval + γ)(b_eval + β.sigma2_eval +
        //                  γ)α.β.nu_{linear}.z_omega_eval.[sigma3]_1     >> last perm polynomial
        //
        // Again, we dont actually compute the MSMs and just accumulate scalars and group elements and postpone MSM to last
        // step.
        //
        S::append_scalar_multiplication_inputs(
            &(*self.key).borrow(),
            &alpha,
            &transcript,
            &self.kate_fr_elements,
        );

        // Fetch the group elements [W_z]_1,[W_zω]_1 from the transcript
        let pi_z: G1Affine = transcript.get_group_element("PI_Z");
        let pi_z_omega: G1Affine = transcript.get_group_element("PI_Z_OMEGA");

        // Validate PI_Z, PI_Z_OMEGA are valid ecc points.
        // N.B. we check that witness commitments are valid points in KateCommitmentScheme<settings>::batch_verify
        // note that we don't check that PI_Z/PI_Z_OMEGA is not the point at infinity, because it's an affine repr, so it can't be.
        if !pi_z.is_on_curve() {
            return Err(anyhow!(
                "opening proof group element PI_Z not a valid point"
            ));
        }
        if !pi_z_omega.is_on_curve() {
            return Err(anyhow!(
                "opening proof group element PI_Z_OMEGA not a valid point"
            ));
        }

        // Accumulate pairs of scalars and group elements which would be used in the final pairing check.
        self.kate_g1_elements
            .insert("PI_Z_OMEGA".to_string(), pi_z_omega);
        self.kate_fr_elements.insert(
            "PI_Z_OMEGA".to_string(),
            zeta * (*self.key).borrow().domain.root * separator_challenge,
        );

        self.kate_g1_elements.insert("PI_Z".to_owned(), pi_z);
        self.kate_fr_elements.insert("PI_Z".to_owned(), zeta);

        // Initialize vectors for scalars and elements
        let mut scalars: Vec<Fr> = Vec::new();
        let mut elements: Vec<G1Affine> = Vec::new();

        // Iterate through the kate_g1_elements and accumulate scalars and elements
        for (key, element) in &self.kate_g1_elements {
            // this used to contain a check for whether the element was the point at infinity
            // but seeing as it's an affine repr, it can't be. so i removed it.
            if element.is_on_curve() {
                // TODO: perhaps we should throw if not on curve or if infinity?
                scalars.push(self.kate_fr_elements[key]);
                elements.push(*element);
            }
        }

        let n = elements.len();
        elements.resize(2 * n, G1Affine::zero());

        // Generate Pippenger point table
        //     this was: barretenberg::scalar_multiplication::generate_pippenger_point_table(&elements[0], &elements[0], num_elements);
        let mut elements_clone = elements.clone();
        let elements_len = elements.len();
        generate_pippenger_point_table(&mut elements_clone[..], &mut elements[..], elements_len);
        let mut state = PippengerRuntimeState::new(n);

        let mut p: [G1Affine; 2] = [G1Affine::zero(); 2];
        p[0] = state.pippenger(&mut [scalars[0]], &[elements[0]], n, false);
        p[1] =
            <ark_ec::short_weierstrass::Projective<ark_bn254::g1::Config> as std::convert::Into<
                G1Affine,
            >>::into(-(G1Affine::identity() * separator_challenge + pi_z));

        let key_borrowed = (*self.key).borrow();
        if key_borrowed.contains_recursive_proof {
            assert!(key_borrowed.recursive_proof_public_input_indices.len() == 16);
            let inputs: Vec<Fr> = transcript.get_field_element_vector("public_inputs");
            let recover_fq_from_public_inputs =
                |idx0: usize, idx1: usize, idx2: usize, idx3: usize| {
                    // TODO THIS BADLY NEEDS CHECKING!!
                    let mut l0: ark_ff::BigInt<4> = inputs[idx0].into();
                    let mut l1: ark_ff::BigInt<4> = inputs[idx1].into();
                    let mut l2: ark_ff::BigInt<4> = inputs[idx2].into();
                    let mut l3: ark_ff::BigInt<4> = inputs[idx3].into();
                    l1.muln(NUM_LIMB_BITS_IN_FIELD_SIMULATION.try_into().unwrap());
                    l2.muln((NUM_LIMB_BITS_IN_FIELD_SIMULATION * 2).try_into().unwrap());
                    l3.muln((NUM_LIMB_BITS_IN_FIELD_SIMULATION * 3).try_into().unwrap());
                    let _ = l0.add_with_carry(&l1);
                    let _ = l0.add_with_carry(&l2);
                    let _ = l0.add_with_carry(&l3);
                    Fq::new(l0)
                };

            let recursion_separator_challenge = transcript
                .get_challenge_field_element::<Fr>("separator", None)
                .square();

            let x0 = recover_fq_from_public_inputs(
                key_borrowed.recursive_proof_public_input_indices[0] as usize,
                key_borrowed.recursive_proof_public_input_indices[1] as usize,
                key_borrowed.recursive_proof_public_input_indices[2] as usize,
                key_borrowed.recursive_proof_public_input_indices[3] as usize,
            );
            let y0 = recover_fq_from_public_inputs(
                key_borrowed.recursive_proof_public_input_indices[4] as usize,
                key_borrowed.recursive_proof_public_input_indices[5] as usize,
                key_borrowed.recursive_proof_public_input_indices[6] as usize,
                key_borrowed.recursive_proof_public_input_indices[7] as usize,
            );
            let x1 = recover_fq_from_public_inputs(
                key_borrowed.recursive_proof_public_input_indices[8] as usize,
                key_borrowed.recursive_proof_public_input_indices[9] as usize,
                key_borrowed.recursive_proof_public_input_indices[10] as usize,
                key_borrowed.recursive_proof_public_input_indices[11] as usize,
            );
            let y1 = recover_fq_from_public_inputs(
                key_borrowed.recursive_proof_public_input_indices[12] as usize,
                key_borrowed.recursive_proof_public_input_indices[13] as usize,
                key_borrowed.recursive_proof_public_input_indices[14] as usize,
                key_borrowed.recursive_proof_public_input_indices[15] as usize,
            );
            p[0] = (p[0] + G1Projective::new(x0, y0, Fq::one()) * recursion_separator_challenge)
                .into_affine();
            p[1] = (p[1] + G1Projective::new(x1, y1, Fq::one()) * recursion_separator_challenge)
                .into_affine();
        }

        // The final pairing check of step 12.
        let result: Fq12 = reduced_ate_pairing_batch_precomputed(
            &p,
            (*self.key)
                .borrow()
                .reference_string
                .borrow()
                .get_precomputed_g2_lines()
                .as_ref(),
            2,
        );

        Ok(result == Fq12::one())
    }
}
