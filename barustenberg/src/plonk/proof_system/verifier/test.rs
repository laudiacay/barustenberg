use ark_ff::{FftField};

use super::*;

impl<H: BarretenHasher, S: Settings<Hasher = H, Field = Fr, Group = G1Affine>> Verifier<'_, H, S> {
    pub fn generate_verifier(circuit_proving_key: Rc<RefCell<ProvingKey<'_, Fr, G>>>) -> Self {
        let mut poly_coefficients: Vec<&mut [Fr]> = vec![&mut []; 8];
        poly_coefficients[0] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"q_1".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();
        poly_coefficients[1] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"q_2".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();
        poly_coefficients[2] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"q_3".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();
        poly_coefficients[3] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"q_m".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();
        poly_coefficients[4] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"q_c".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();
        poly_coefficients[5] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"sigma_1".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();
        poly_coefficients[6] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"sigma_2".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();
        poly_coefficients[7] = circuit_proving_key
            .borrow()
            .polynomial_store
            .get(&"sigma_3".to_owned())
            .unwrap()
            .borrow_mut()
            .coefficients
            .as_mut_slice();

        let mut commitments = vec![G1Affine::default(); 8];
        let mut state = PippengerRuntimeState::new(circuit_proving_key.borrow().circuit_size);

        for i in 0..8 {
            commitments[i] = G1Affine::from(
                state.pippenger(
                    poly_coefficients[i],
                    *(circuit_proving_key
                        .borrow()
                        .reference_string
                        .borrow()
                        .get_monomial_points())[..],
                    circuit_proving_key.borrow().circuit_size,
                    false,
                ),
            );
        }

        // TODOL: this number of points in arbitrary and needs to be checked with the reference string
        let crs =Rc::new(FileReferenceString::new(32, "../srs_db/ignition"));
        let circuit_verification_key = Rc::new(VerificationKey::new(
            circuit_proving_key.borrow().circuit_size,
            circuit_proving_key.borrow().num_public_inputs,
            crs,
            circuit_proving_key.borrow().composer_type,
        ));

        circuit_verification_key
            .commitments
            .insert("Q_1".to_string(), commitments[0]);
        circuit_verification_key
            .commitments
            .insert("Q_2".to_string(), commitments[1]);
        circuit_verification_key
            .commitments
            .insert("Q_3".to_string(), commitments[2]);
        circuit_verification_key
            .commitments
            .insert("Q_M".to_string(), commitments[3]);
        circuit_verification_key
            .commitments
            .insert("Q_C".to_string(), commitments[4]);

        circuit_verification_key
            .commitments
            .insert("SIGMA_1".to_string(), commitments[5]);
        circuit_verification_key
            .commitments
            .insert("SIGMA_2".to_string(), commitments[6]);
        circuit_verification_key
            .commitments
            .insert("SIGMA_3".to_string(), commitments[7]);

        let verifier = Verifier::new(
            Some(circuit_verification_key),
            ComposerType::Standard.create_manifest(0),
        );

        let kate_commitment_scheme = Box::new(KateCommitmentScheme::<
            H,
            Fq, Fr, G1Affine,
        >::new());
        verifier.commitment_scheme = kate_commitment_scheme;
        verifier
    }
}

fn generate_test_data<
    'a,
    Fq: Field + FftField,
    Fr: Field + FftField,
    G: AffineRepr,
    H: BarretenHasher + Default,
>(
    n: usize,
) -> Prover<'a, H, StandardSettings<H>> {
    // create some constraints that satisfy our arithmetic circuit relation
    let crs = Rc::new(FileReferenceString::new(n + 1, "../srs_db/ignition"));
    let key = Rc::new(ProvingKey::new(n, 0, crs, ComposerType::Standard));

    let mut rand = rand::thread_rng();

    let mut w_l = Polynomial::new(n);
    let mut w_r = Polynomial::new(n);
    let mut w_o = Polynomial::new(n);
    let mut q_l = Polynomial::new(n);
    let mut q_r = Polynomial::new(n);
    let mut q_o = Polynomial::new(n);
    let mut q_c: Polynomial<Fr> = Polynomial::new(n);
    let mut q_m = Polynomial::new(n);

    let mut t0;
    for i in 0..n / 4 {
        w_l.coefficients[2 * i] = Fr::rand(&mut rand);
        w_r.coefficients[2 * i] = Fr::rand(&mut rand);
        w_o.coefficients[2 * i] = w_l.coefficients[2 * i] * w_r.coefficients[2 * i];
        w_o.coefficients[2 * i] += w_l.coefficients[2 * i];
        w_o.coefficients[2 * i] += w_r.coefficients[2 * i];
        w_o.coefficients[2 * i] += Fr::one();
        q_l.coefficients[2 * i] = Fr::one();
        q_r.coefficients[2 * i] = Fr::one();
        q_o.coefficients[2 * i] = -Fr::one();
        q_c.coefficients[2 * i] = Fr::one();
        q_m.coefficients[2 * i] = Fr::one();

        w_l.coefficients[2 * i + 1] = Fr::rand(&mut rand);
        w_r.coefficients[2 * i + 1] = Fr::rand(&mut rand);
        w_o.coefficients[2 * i + 1] = Fr::rand(&mut rand);

        t0 = w_l.coefficients[2 * i + 1] + w_r.coefficients[2 * i + 1];
        q_c[2 * i + 1] = t0 + w_o[2 * i + 1];
        q_c[2 * i + 1] = -q_c[2 * i + 1];
        q_l[2 * i + 1] = Fr::one();
        q_r[2 * i + 1] = Fr::one();
        q_o[2 * i + 1] = Fr::one();
        q_m[2 * i + 1] = Fr::zero();
    }

    let shift = n / 2;
    w_l.coefficients[shift..].copy_from_slice(&w_l.coefficients[..shift]);
    w_r.coefficients[shift..].copy_from_slice(&w_r.coefficients[..shift]);
    w_o.coefficients[shift..].copy_from_slice(&w_o.coefficients[..shift]);
    q_m.coefficients[shift..].copy_from_slice(&q_m.coefficients[..shift]);
    q_l.coefficients[shift..].copy_from_slice(&q_l.coefficients[..shift]);
    q_r.coefficients[shift..].copy_from_slice(&q_r.coefficients[..shift]);
    q_o.coefficients[shift..].copy_from_slice(&q_o.coefficients[..shift]);
    q_c.coefficients[shift..].copy_from_slice(&q_c.coefficients[..shift]);

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

    let sigma_1_lagrange_base = sigma_1.clone();
    let sigma_2_lagrange_base = sigma_2.clone();
    let sigma_3_lagrange_base = sigma_3.clone();

    key.polynomial_store
        .insert(&"sigma_1_lagrange".to_string(), sigma_1_lagrange_base);
    key.polynomial_store
        .insert(&"sigma_2_lagrange".to_string(), sigma_2_lagrange_base);
    key.polynomial_store
        .insert(&"sigma_3_lagrange".to_string(), sigma_3_lagrange_base);

    key.small_domain.ifft_inplace(&mut sigma_1);
    key.small_domain.ifft_inplace(&mut sigma_2);
    key.small_domain.ifft_inplace(&mut sigma_3);

    const WIDTH: usize = 4;
    let mut sigma_1_fft = sigma_1.clone();
    sigma_1_fft.resize(key.circuit_size * WIDTH, Fr::zero());
    let mut sigma_2_fft = sigma_2.clone();
    sigma_2_fft.resize(key.circuit_size * WIDTH, Fr::zero());
    let mut sigma_3_fft = sigma_3.clone();
    sigma_3_fft.resize(key.circuit_size * WIDTH, Fr::zero());

    key.large_domain.coset_fft_inplace(&mut sigma_1_fft.coefficients[..]);
    key.large_domain.coset_fft_inplace(&mut sigma_2_fft.coefficients[..]);
    key.large_domain.coset_fft_inplace(&mut sigma_3_fft.coefficients[..]);

    key.polynomial_store.insert(&"sigma_1".to_string(), sigma_1);
    key.polynomial_store.insert(&"sigma_2".to_string(), sigma_2);
    key.polynomial_store.insert(&"sigma_3".to_string(), sigma_3);

    key.polynomial_store
        .insert(&"sigma_1_fft".to_string(), sigma_1_fft);
    key.polynomial_store
        .insert(&"sigma_2_fft".to_string(), sigma_2_fft);
    key.polynomial_store
        .insert(&"sigma_3_fft".to_string(), sigma_3_fft);

    key.polynomial_store
        .insert(&"w_1_lagrange".to_string(), w_l);
    key.polynomial_store
        .insert(&"w_2_lagrange".to_string(), w_r);
    key.polynomial_store
        .insert(&"w_3_lagrange".to_string(), w_o);

    key.small_domain.ifft_inplace(&mut q_l);
    key.small_domain.ifft_inplace(&mut q_r);
    key.small_domain.ifft_inplace(&mut q_o);
    key.small_domain.ifft_inplace(&mut q_m);
    key.small_domain.ifft_inplace(&mut q_c);

    let mut q_1_fft = q_l.clone();
    q_1_fft.resize(n*4, Fr::zero());
    let mut q_2_fft = q_r.clone();
    q_2_fft.resize(n*4, Fr::zero());
    let mut q_3_fft = q_o.clone();
    q_3_fft.resize(n*4, Fr::zero());
    let mut q_m_fft = q_m.clone();
    q_m_fft.resize(n*4, Fr::zero());
    let mut q_c_fft = q_c.clone();
    q_c_fft.resize(n*4, Fr::zero());

    key.large_domain.coset_fft_inplace(&mut q_1_fft.coefficients[..]);
    key.large_domain.coset_fft_inplace(&mut q_2_fft.coefficients[..]);
    key.large_domain.coset_fft_inplace(&mut q_3_fft.coefficients[..]);
    key.large_domain.coset_fft_inplace(&mut q_m_fft.coefficients[..]);
    key.large_domain.coset_fft_inplace(&mut q_c_fft.coefficients[..]);

    key.polynomial_store.insert(&"q_1".to_string(), q_l);
    key.polynomial_store.insert(&"q_2".to_string(), q_r);
    key.polynomial_store.insert(&"q_3".to_string(), q_o);
    key.polynomial_store.insert(&"q_m".to_string(), q_m);
    key.polynomial_store.insert(&"q_c".to_string(), q_c);

    key.polynomial_store.insert(&"q_1_fft".to_string(), q_1_fft);
    key.polynomial_store.insert(&"q_2_fft".to_string(), q_2_fft);
    key.polynomial_store.insert(&"q_3_fft".to_string(), q_3_fft);
    key.polynomial_store.insert(&"q_m_fft".to_string(), q_m_fft);
    key.polynomial_store.insert(&"q_c_fft".to_string(), q_c_fft);

    let permutation_widget: Box<ProverPermutationWidget<'_, Fr, H, G>> =
        Box::new(ProverPermutationWidget::<3>::new(key.clone()));

    let widget: Box<ProverArithmeticWidget<'_, StandardSettings>> = Box::new(
        ProverArithmeticWidget::<_, StandardSettings>::new(key.clone()),
    );

    let kate_commitment_scheme = KateCommitmentScheme::<H, Fq, Fr, G1Affine>::new();

    let state : Prover<'_, H, StandardSettings<H>> = Prover::new(
        Some(key),
        Some(ComposerType::StandardComposer::create_manifest(0)),
        None,
    );
    state.random_widgets.push(permutation_widget);
    state.transition_widgets.push(widget);
    state.commitment_scheme = kate_commitment_scheme;
    state
}

use std::{cell::RefCell, rc::Rc};

use crate::{
    plonk::{
        composer::composer_base::ComposerType,
        proof_system::{
            commitment_scheme::KateCommitmentScheme,
            prover::Prover,
            proving_key::ProvingKey,
            types::prover_settings::StandardSettings,
            utils::permutation::compute_permutation_lagrange_base_single,
            widgets::{
                random_widgets::permutation_widget::ProverPermutationWidget,
                transition_widgets::arithmetic_widget::ProverArithmeticWidget,
            },
        },
    },
    polynomials::Polynomial,
    srs::reference_string::file_reference_string::FileReferenceString, transcript::Keccak256,
};

#[test]
fn verify_arithmetic_proof_small() {
    let n = 8;

    let state = generate_test_data::<Fq, Fr, G1Affine, Keccak256>(n);
    let verifier : Verifier<'_, Keccak256, StandardSettings<Keccak256>>= Verifier::generate_verifier(&state.key);
    
    // Construct proof
    let proof = state.construct_proof().unwrap();

    // Verify proof
    let result = verifier.verify_proof(&proof).unwrap();

    assert!(result);
}

#[test]
fn verify_arithmetic_proof() {
    let n = 1 << 14;

    let state = generate_test_data::<Fq, Fr, G1Affine, Keccak256>(n);
    let verifier : Verifier<'_, Keccak256, StandardSettings<Keccak256>>= Verifier::generate_verifier(&state.key);

    // Construct proof
    let proof = state.construct_proof().unwrap();

    // Verify proof
    let result = verifier.verify_proof(&proof).unwrap();

    assert!(result);
}

#[test]
#[should_panic]
fn verify_damaged_proof() {
    let n = 8;

    let state = generate_test_data::<Fq, Fr, G1Affine, Keccak256>(n);
    let verifier : Verifier<'_, Keccak256, StandardSettings<Keccak256>>= Verifier::generate_verifier(&state.key);

    // Create empty proof
    let proof = Proof::default();

    // Verify proof
    verifier.verify_proof(&proof).unwrap();
}
