use ark_bn254::Fq12;

use super::*;

impl<Fq: Field, Fr: Field + FftField, G1Affine: AffineRepr, H: BarretenHasher, PS: Settings<H>>
    Verifier<'_, Fq, Fr, G1Affine, H, PS>
{
    pub fn generate_verifier(circuit_proving_key: Arc<ProvingKey<'_, Fr, G1Affine>>) -> Self {
        let mut poly_coefficients = vec![[]; 8];
        poly_coefficients[0] = circuit_proving_key
            .polynomial_store
            .get("q_1".to_owned())?
            .map(|p| p.coefficients());
        poly_coefficients[1] = circuit_proving_key
            .polynomial_store
            .get("q_2".to_owned())?
            .map(|p| p.coefficients());
        poly_coefficients[2] = circuit_proving_key
            .polynomial_store
            .get("q_3".to_owned())?
            .map(|p| p.coefficients());
        poly_coefficients[3] = circuit_proving_key
            .polynomial_store
            .get("q_m".to_owned())?
            .borrow_mut()
            .coefficients;
        poly_coefficients[4] = circuit_proving_key
            .polynomial_store
            .get("q_c".to_owned())?
            .map(|p| p.coefficients());
        poly_coefficients[5] = circuit_proving_key
            .polynomial_store
            .get("sigma_1".to_owned())?
            .map(|p| p.coefficients());
        poly_coefficients[6] = circuit_proving_key
            .polynomial_store
            .get("sigma_2".to_owned())?
            .map(|p| p.coefficients());
        poly_coefficients[7] = circuit_proving_key
            .polynomial_store
            .get("sigma_3".to_owned())?
            .map(|p| p.coefficients());

        let mut commitments = vec![G1Affine::default(); 8];
        let mut state = PippengerRuntimeState::new(circuit_proving_key.circuit_size);

        for i in 0..8 {
            if let Some(poly_coeffs) = &poly_coefficients[i] {
                commitments[i] = G1Affine::from_projective(state.pippenger(
                    poly_coeffs,
                    circuit_proving_key.reference_string.monomial_points(),
                    circuit_proving_key.circuit_size,
                ));
            }
        }

        // TODOL: this number of points in arbitrary and needs to be checked with the reference string
        let crs = Arc::new(FileReferenceString::new(32, "../srs_db/ignition"));
        let circuit_verification_key = Arc::new(VerificationKey::new(
            circuit_proving_key.circuit_size,
            circuit_proving_key.num_public_inputs,
            crs,
            circuit_proving_key.composer_type,
        ));

        circuit_verification_key
            .commitments
            .insert("Q_1", commitments[0]);
        circuit_verification_key
            .commitments
            .insert("Q_2", commitments[1]);
        circuit_verification_key
            .commitments
            .insert("Q_3", commitments[2]);
        circuit_verification_key
            .commitments
            .insert("Q_M", commitments[3]);
        circuit_verification_key
            .commitments
            .insert("Q_C", commitments[4]);

        circuit_verification_key
            .commitments
            .insert("SIGMA_1", commitments[5]);
        circuit_verification_key
            .commitments
            .insert("SIGMA_2", commitments[6]);
        circuit_verification_key
            .commitments
            .insert("SIGMA_3", commitments[7]);

        let verifier = Verifier::new(
            Some(circuit_verification_key),
            ComposerType::StandardComposer::create_manifest(0),
        );

        let kate_commitment_scheme = Box::new(KateCommitmentScheme::<
            H,
            crate::plonk::proof_system::types::polynomial_manifest::PolynomialIndex,
        >::new());
        verifier.commitment_scheme = kate_commitment_scheme;
        verifier
    }

    pub fn verify_proof(self, proof: &Proof) -> Result<bool, &'static str> {
        // This function verifies a PLONK proof for given program settings.
        // A PLONK proof for standard PLONK is of the form:
        //
        // π_SNARK =   { [a]_1,[b]_1,[c]_1,[z]_1,[t_{low}]_1,[t_{mid}]_1,[t_{high}]_1,[W_z]_1,[W_zω]_1 \in G,
        //                a_eval, b_eval, c_eval, sigma1_eval, sigma2_eval, sigma3_eval,
        //                  q_l_eval, q_r_eval, q_o_eval, q_m_eval, q_c_eval, z_eval_omega \in F }
        //
        // Proof π_SNARK must first be added to the transcript with the other program_settings.
        self.key.program_width = S::PROGRAM_WIDTH;

        // Initialize the transcript.
        let mut transcript = Transcript::StandardTranscript::new(
            proof.proof_data.clone(),
            self.manifest.clone(),
            S::HASH_TYPE,
            S::NUM_CHALLENGE_BYTES,
        );

        // Add circuit size and public input size to the transcript.
        transcript.add_element("circuit_size", proof.key.circuit_size.to_be_bytes());
        transcript.add_element(
            "public_input_size",
            proof.key.num_public_inputs.to_be_bytes(),
        );

        // Compute challenges using Fiat-Shamir heuristic.
        transcript.apply_fiat_shamir("init");
        transcript.apply_fiat_shamir("eta");
        transcript.apply_fiat_shamir("beta");
        transcript.apply_fiat_shamir("alpha");
        transcript.apply_fiat_shamir("z");

        // Deserialize alpha and zeta from the transcript.
        let alpha = Fr::deserialize_from_buffer(transcript.get_challenge("alpha"));
        let zeta = Fr::deserialize_from_buffer(transcript.get_challenge("z"));

        todo!("fail here- are you sure this is the right function?");
        // Compute the evaluations of the Lagrange polynomials and the vanishing polynomial.
        let lagrange_evals = &self.key.domain.evaluate_all_lagrange_coefficients(zeta);

        // Compute quotient polynomial evaluation at zeta.
        let mut t_numerator_eval = Fr::default();
        S::compute_quotient_evaluation_contribution(
            &self.key,
            alpha,
            &transcript,
            &mut t_numerator_eval,
        );
        let t_eval = t_numerator_eval * lagrange_evals.vanishing_poly.inverse();
        transcript.add_element("t", t_eval.to_buffer());

        // Compute nu and separator challenges.
        transcript.apply_fiat_shamir("nu");
        transcript.apply_fiat_shamir("separator");
        let separator_challenge =
            Fr::deserialize_from_buffer(transcript.get_challenge("separator"));

        // Verify the commitments using Kate commitment scheme.
        self.commitment_scheme.batch_verify(
            &transcript,
            &mut self.kate_g1_elements,
            &mut self.kate_fr_elements,
            &self.key,
        )?;

        // Append scalar multiplication inputs.
        S::append_scalar_multiplication_inputs(
            &self.key,
            alpha,
            &transcript,
            &mut self.kate_fr_elements,
        );

        // Get PI_Z and PI_Z_OMEGA from the transcript.
        let pi_z = G1Affine::deserialize_from_buffer(transcript.get_element("PI_Z"));
        let pi_z_omega = G1Affine::deserialize_from_buffer(transcript.get_element("PI_Z_OMEGA"));

        // Check if PI_Z and PI_Z_OMEGA are valid points.
        if !pi_z.on_curve() || pi_z.is_point_at_infinity() {
            return Err("opening proof group element PI_Z not a valid point".into());
        }
        if !pi_z_omega.on_curve() || pi_z_omega.is_point_at_infinity() {
            return Err("opening proof group element PI_Z_OMEGA not a valid point".into());
        }

        // get kate_g1_elements: HashMap<u64, G1Affine> and kate_fr_elements: HashMap<u64, Fr>
        let mut kate_g1_elements: HashMap<String, G1Affine> = HashMap::new();
        let mut kate_fr_elements: HashMap<String, Fr> = HashMap::new();

        // Initialize vectors for scalars and elements
        let mut scalars: Vec<Fr> = Vec::new();
        let mut elements: Vec<G1Affine> = Vec::new();

        // Iterate through the kate_g1_elements and accumulate scalars and elements
        for (key, element) in &kate_g1_elements {
            if element.is_on_curve() && !element.is_zero() {
                if let Some(scalar) = kate_fr_elements.get(key) {
                    scalars.push(*scalar);
                    elements.push(*element);
                }
            }
        }

        // Resize elements vector to make room for Pippenger point table
        let n = elements.len();
        elements.resize(2 * n, G1Affine::zero());

        // Generate Pippenger point table
        generate_pippenger_point_table(&mut elements[..]);

        // Create Pippenger runtime state
        let mut state = PippengerRuntimeState::new(n);

        // Perform Pippenger multi-scalar multiplication
        let p0 = state.pippenger(&scalars, &elements);

        // Calculate P[1]
        let p1 =
            -((G1Projective::from(pi_z_omega) * separator_challenge) + G1Projective::from(pi_z));

        // Check if recursive proof is present
        if let Some(recursive_proof_indices) = self.key.recursive_proof_public_input_indices {
            assert_eq!(recursive_proof_indices.len(), 16);

            let inputs = transcript.get_field_element_vector("public_inputs");

            //  Recover Fq values from public inputs
            let recover_fq_from_public_inputs =
                |idx0: usize, idx1: usize, idx2: usize, idx3: usize| {
                    let l0 = inputs[idx0];
                    let l1 = inputs[idx1];
                    let l2 = inputs[idx2];
                    let l3 = inputs[idx3];

                    let limb = l0
                        + (l1 << NUM_LIMB_BITS_IN_FIELD_SIMULATION)
                        + (l2 << (NUM_LIMB_BITS_IN_FIELD_SIMULATION * 2))
                        + (l3 << (NUM_LIMB_BITS_IN_FIELD_SIMULATION * 3));
                    Fq::from(limb)
                };

            // Get recursion_separator_challenge
            let recursion_separator_challenge =
                transcript.get_challenge_field_element("separator").square();

            // Recover x0, y0, x1, and y1
            let x0 = recover_fq_from_public_inputs(
                recursive_proof_indices[0],
                recursive_proof_indices[1],
                recursive_proof_indices[2],
                recursive_proof_indices[3],
            );
            let y0 = recover_fq_from_public_inputs(
                recursive_proof_indices[4],
                recursive_proof_indices[5],
                recursive_proof_indices[6],
                recursive_proof_indices[7],
            );
            let x1 = recover_fq_from_public_inputs(
                recursive_proof_indices[8],
                recursive_proof_indices[9],
                recursive_proof_indices[10],
                recursive_proof_indices[11],
            );
            let y1 = recover_fq_from_public_inputs(
                recursive_proof_indices[12],
                recursive_proof_indices[13],
                recursive_proof_indices[14],
                recursive_proof_indices[15],
            );

            // Update P[0] and P[1] with recursive proof values
            let p0 = p0 + (G1Projective::new(x0, y0, Fq::one()) * recursion_separator_challenge);
            let p1 = p1 + (G1Projective::new(x1, y1, Fq::one()) * recursion_separator_challenge);
        }

        // Normalize P[0] and P[1]
        let p_affine = [G1Affine::from(p0), G1Affine::from(p1)];

        // Perform final pairing check
        let result = reduced_ate_pairing_batch_precomputed(
            &p_affine,
            &self.key.reference_string.get_precomputed_g2_lines(),
            // TODO this num_points was NOT provided in the original code.
            p_affine.len(),
        );

        // Check if result equals Fq12::one()
        Ok(result == Fq12::one())
        // Err("opening proof group element PI_Z not a valid point".into());
    }
}

fn generate_test_data<'a>(n: usize) -> Prover<'a, Fr, StandardSettings> {
    // create some constraints that satisfy our arithmetic circuit relation
    let crs = Rc::new(FileReferenceString::new(n + 1, "../srs_db/ignition"));
    let key = Rc::new(ProvingKey::new(n, 0, crs, ComposerType::Standard));

    let mut w_l = Polynomial::new(n);
    let mut w_r = Polynomial::new(n);
    let mut w_o = Polynomial::new(n);
    let mut q_l = Polynomial::new(n);
    let mut q_r = Polynomial::new(n);
    let mut q_o = Polynomial::new(n);
    let mut q_c = Polynomial::new(n);
    let mut q_m = Polynomial::new(n);

    let mut t0;
    for i in 0..n / 4 {
        w_l.coeffs[2 * i] = Fr::random_element();
        w_r.coeffs[2 * i] = Fr::random_element();
        w_o.coeffs[2 * i] = w_l.coeffs[2 * i] * w_r.coeffs[2 * i];
        w_o.coeffs[2 * i] += w_l.coeffs[2 * i];
        w_o.coeffs[2 * i] += w_r.coeffs[2 * i];
        w_o.coeffs[2 * i] += Fr::one();
        q_l.coeffs[2 * i] = Fr::one();
        q_r.coeffs[2 * i] = Fr::one();
        q_o.coeffs[2 * i] = Fr::neg_one();
        q_c.coeffs[2 * i] = Fr::one();
        q_m.coeffs[2 * i] = Fr::one();

        w_l.coeffs[2 * i + 1] = Fr::random_element();
        w_r.coeffs[2 * i + 1] = Fr::random_element();
        w_o.coeffs[2 * i + 1] = Fr::random_element();

        t0 = w_l.coeffs[2 * i + 1] + w_r.coeffs[2 * i + 1];
        q_c[2 * i + 1] = t0 + w_o[2 * i + 1];
        q_c[2 * i + 1].self_neg();
        q_l[2 * i + 1] = Fr::one();
        q_r[2 * i + 1] = Fr::one();
        q_o[2 * i + 1] = Fr::one();
        q_m[2 * i + 1] = Fr::zero();
    }

    let shift = n / 2;
    w_l.coeffs[shift..].copy_from_slice(&w_l.coeffs[..shift]);
    w_r.coeffs[shift..].copy_from_slice(&w_r.coeffs[..shift]);
    w_o.coeffs[shift..].copy_from_slice(&w_o.coeffs[..shift]);
    q_m.coeffs[shift..].copy_from_slice(&q_m.coeffs[..shift]);
    q_l.coeffs[shift..].copy_from_slice(&q_l.coeffs[..shift]);
    q_r.coeffs[shift..].copy_from_slice(&q_r.coeffs[..shift]);
    q_o.coeffs[shift..].copy_from_slice(&q_o.coeffs[..shift]);
    q_c.coeffs[shift..].copy_from_slice(&q_c.coeffs[..shift]);

    let mut sigma_1_mapping: Vec<u32> = vec![0; n];
    let mut sigma_2_mapping: Vec<u32> = vec![0; n];
    let mut sigma_3_mapping: Vec<u32> = vec![0; n];

    // create basic permutation - second half of witness vector is a copy of the first half
    for i in 0..(n / 2) {
        sigma_1_mapping[shift + i] = i as u32;
        sigma_2_mapping[shift + i] = (i as u32) + (1 << 30);
        sigma_3_mapping[shift + i] = (i as u32) + (1 << 31);
        sigma_1_mapping[i] = (i + shift) as u32;
        sigma_2_mapping[i] = ((i + shift) as u32) + (1 << 30);
        sigma_3_mapping[i] = ((i + shift) as u32) + (1 << 31);
    }

    // make last permutation the same as identity permutation
    // we are setting the permutation in the last 4 gates as identity permutation since
    // we are cutting out 4 roots as of now.

    let num_roots_cut_out_of_the_vanishing_polynomial = 4;
    for j in 0..num_roots_cut_out_of_the_vanishing_polynomial {
        let index = (shift - 1 - j) as u32;
        sigma_1_mapping[shift - 1 - j] = index;
        sigma_2_mapping[shift - 1 - j] = index + (1 << 30);
        sigma_3_mapping[shift - 1 - j] = index + (1 << 31);
        sigma_1_mapping[n - 1 - j] = (n - 1 - j) as u32;
        sigma_2_mapping[n - 1 - j] = ((n - 1 - j) as u32) + (1 << 30);
        sigma_3_mapping[n - 1 - j] = ((n - 1 - j) as u32) + (1 << 31);
    }

    let mut sigma_1 = Polynomial::new(key.circuit_size);
    let mut sigma_2 = Polynomial::new(key.circuit_size);
    let mut sigma_3 = Polynomial::new(key.circuit_size);

    compute_permutation_lagrange_base_single(&mut sigma_1, &sigma_1_mapping, &key.small_domain);
    compute_permutation_lagrange_base_single(&mut sigma_2, &sigma_2_mapping, &key.small_domain);
    compute_permutation_lagrange_base_single(&mut sigma_3, &sigma_3_mapping, &key.small_domain);

    let sigma_1_lagrange_base = Polynomial::new_from(sigma_1, key.circuit_size);
    let sigma_2_lagrange_base = Polynomial::new_from(sigma_2, key.circuit_size);
    let sigma_3_lagrange_base = Polynomial::new_from(sigma_3, key.circuit_size);

    key.polynomial_store
        .insert("sigma_1_lagrange", sigma_1_lagrange_base);
    key.polynomial_store
        .insert("sigma_2_lagrange", sigma_2_lagrange_base);
    key.polynomial_store
        .insert("sigma_3_lagrange", sigma_3_lagrange_base);

    sigma_1.ifft(&key.small_domain);
    sigma_2.ifft(&key.small_domain);
    sigma_3.ifft(&key.small_domain);

    const WIDTH: usize = 4;
    let sigma_1_fft = Polynomial::new_from(sigma_1, key.circuit_size * WIDTH);
    let sigma_2_fft = Polynomial::new_from(sigma_2, key.circuit_size * WIDTH);
    let sigma_3_fft = Polynomial::new_from(sigma_3, key.circuit_size * WIDTH);

    sigma_1_fft.coset_fft(&key.large_domain);
    sigma_2_fft.coset_fft(&key.large_domain);
    sigma_3_fft.coset_fft(&key.large_domain);

    key.polynomial_store.insert("sigma_1", sigma_1);
    key.polynomial_store.insert("sigma_2", sigma_2);
    key.polynomial_store.insert("sigma_3", sigma_3);

    key.polynomial_store.insert("sigma_1_fft", sigma_1_fft);
    key.polynomial_store.insert("sigma_2_fft", sigma_2_fft);
    key.polynomial_store.insert("sigma_3_fft", sigma_3_fft);

    key.polynomial_store.insert("w_1_lagrange", w_l);
    key.polynomial_store.insert("w_2_lagrange", w_r);
    key.polynomial_store.insert("w_3_lagrange", w_o);

    q_l.ifft(&key.small_domain);
    q_r.ifft(&key.small_domain);
    q_o.ifft(&key.small_domain);
    q_m.ifft(&key.small_domain);
    q_c.ifft(&key.small_domain);

    let q_1_fft = Polynomial::new_from(q_l, n_times_4);
    let q_2_fft = Polynomial::new_from(q_r, n_times_4);
    let q_3_fft = Polynomial::new_from(q_o, n_times_4);
    let q_m_fft = Polynomial::new_from(q_m, n_times_4);
    let q_c_fft = Polynomial::new_from(q_c, n_times_4);

    q_1_fft.coset_fft(&key.large_domain);
    q_2_fft.coset_fft(&key.large_domain);
    q_3_fft.coset_fft(&key.large_domain);
    q_m_fft.coset_fft(&key.large_domain);
    q_c_fft.coset_fft(&key.large_domain);

    key.polynomial_store.insert("q_1", q_l);
    key.polynomial_store.insert("q_2", q_r);
    key.polynomial_store.insert("q_3", q_o);
    key.polynomial_store.insert("q_m", q_m);
    key.polynomial_store.insert("q_c", q_c);

    key.polynomial_store.insert("q_1_fft", q_1_fft);
    key.polynomial_store.insert("q_2_fft", q_2_fft);
    key.polynomial_store.insert("q_3_fft", q_3_fft);
    key.polynomial_store.insert("q_m_fft", q_m_fft);
    key.polynomial_store.insert("q_c_fft", q_c_fft);

    let permutation_widget: Box<ProverPermutationWidget> =
        Box::new(ProverPermutationWidget::<3>::new(key.clone()));

    let widget: Box<ProverArithmeticWidget<StandardSettings>> =
        Box::new(ProverArithmeticWidget::<StandardSettings>::new(key.clone()));

    let kate_commitment_scheme = Box::new(KateCommitmentScheme::<StandardSettings>::new());

    let state = Prover::new(
        key,
        Some(ComposerType::StandardComposer::create_manifest(0)),
        None,
    );
    state.random_widgets.push(permutation_widget);
    state.transition_widgets.push(widget);
    state.commitment_scheme = kate_commitment_scheme;
    state
}

use std::rc::Rc;

use crate::{
    ecc::{
        curves::bn254::{
            fq::Fq,
            fr::Fr,
            g1::{G1Affine, G1},
            scalar_multiplication::{
                runtime_states::PippengerRuntimeState,
                scalar_multiplication::generate_pippenger_point_table,
            },
        },
        reduced_ate_pairing_batch_precomputed,
    },
    plonk::{
        composer::composer_base::ComposerType,
        proof_system::{
            commitment_scheme::KateCommitmentScheme,
            constants::NUM_LIMB_BITS_IN_FIELD_SIMULATION,
            prover::Prover,
            proving_key::ProvingKey,
            types::prover_settings::StandardSettings,
            widgets::{
                random_widgets::permutation_widget::ProverPermutationWidget,
                transition_widgets::arithmetic_widget::ProverArithmeticWidget,
            },
        },
    },
    polynomials::Polynomial,
    srs::reference_string::file_reference_string::FileReferenceString,
    transcript::Transcript,
};

#[test]
fn verify_arithmetic_proof_small() {
    let n = 8;

    let state = generate_test_data(n);
    let verifier = Verifier::generate_verifier(&state.key);

    // Construct proof
    let proof = state.construct_proof();

    // Verify proof
    let result = verifier.verify_proof(&proof).unwrap();

    assert!(result);
}

#[test]
fn verify_arithmetic_proof() {
    let n = 1 << 14;

    let state = generate_test_data(n);
    let verifier = Verifier::generate_verifier(&state.key);

    // Construct proof
    let proof = state.construct_proof();

    // Verify proof
    let result = verifier.verify_proof(&proof).unwrap();

    assert!(result);
}

#[test]
#[should_panic]
fn verify_damaged_proof() {
    let n = 8;

    let state = generate_test_data(n);
    let verifier = Verifier::generate_verifier(&state.key);

    // Create empty proof
    let proof = Proof::default();

    // Verify proof
    verifier.verify_proof(&proof).unwrap();
}
