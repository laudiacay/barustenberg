use std::sync::{Arc, RwLock};

use crate::{
    plonk::{
        composer::composer_base::ComposerType,
        proof_system::{
            commitment_scheme::KateCommitmentScheme,
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
    srs::reference_string::file_reference_string::FileReferenceString,
    transcript::{BarretenHasher, Keccak256, Manifest, ManifestEntry, RoundManifest},
};
use ark_bn254::{Fq, Fr, G1Affine};
use ark_ff::{One, UniformRand, Zero};

use super::Prover;

/*
```
elliptic curve point addition on a short weierstrass curve.

circuit has 9 gates, I've added 7 dummy gates so that the polynomial degrees are a power of 2

input points: (x_1, y_1), (x_2, y_2)
output point: (x_3, y_3)
intermediate variables: (t_1, t_2, t_3, t_4, t_5, t_6, t_7)

Variable assignments:
t_1 = (y_2 - y_1)
t_2 = (x_2 - x_1)
t_3 = (y_2 - y_1) / (x_2 - x_1)
x_3 = t_3*t_3 - x_2 - x_1
y_3 = t_3*(x_1 - x_3) - y_1
t_4 = (x_3 + x_1)
t_5 = (t_4 + x_2)
t_6 = (y_3 + y_1)
t_7 = (x_1 - x_3)

Constraints:
(y_2 - y_1) - t_1 = 0
(x_2 - x_1) - t_2 = 0
(x_1 + x_2) - t_4 = 0
(t_4 + x_3) - t_5 = 0
(y_3 + y_1) - t_6 = 0
(x_1 - x_3) - t_7 = 0
 (t_3 * t_2) - t_1 = 0
-(t_3 * t_3) + t_5 = 0
-(t_3 * t_7) + t_6 = 0

Wire polynomials:
w_l = [y_2, x_2, x_1, t_4, y_3, x_1, t_3, t_3, t_3, 0, 0, 0, 0, 0, 0, 0]
w_r = [y_1, x_1, x_2, x_3, y_1, x_3, t_2, t_3, t_7, 0, 0, 0, 0, 0, 0, 0]
w_o = [t_1, t_2, t_4, t_5, t_6, t_7, t_1, t_5, t_6, 0, 0, 0, 0, 0, 0, 0]

Gate polynomials:
q_m = [ 0,  0,  0,  0,  0,  0,  1, -1, -1, 0, 0, 0, 0, 0, 0, 0]
q_l = [ 1,  1,  1,  1,  1,  1,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0]
q_r = [-1, -1,  1,  1,  1, -1,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0]
q_o = [-1, -1, -1, -1, -1, -1, -1,  1,  1, 0, 0, 0, 0, 0, 0, 0]
q_c = [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0]

Permutation polynomials:
s_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sigma_1 = [1, 3+n, 6, 3+2n, 5, 2+n, 8, 9, 8+n, 10, 11, 12, 13, 14, 15, 16]
sigma_2 = [5+n, 3, 2, 6+n, 1+n, 4+n, 2+2n, 7, 6+2n, 10+n, 11+n, 12+n, 13+n, 14+n, 15+n, 16+n]
sigma_3 = [7+2n, 7+n, 4, 8+2n, 9+2n, 9+n, 1+2n, 4+2n, 5+2n, 10+2n, 11+2n, 12+2n, 13+2n, 14+2n, 15+2n]

(for n = 16, permutation polynomials are)
sigma_1 = [1, 19, 6, 35, 5, 18, 8, 9, 24, 10, 11, 12, 13, 14, 15, 16]
sigma_2 = [21, 3, 2, 22, 17, 20, 34, 7, 38, 26, 27, 28, 29, 30, 31, 32]
sigma_3 = [39, 23, 4, 40, 41, 25, 33, 36, 37, 42, 43, 44, 45, 46, 47, 48]
```
*/

fn create_manifest(num_public_inputs: usize) -> Manifest {
    const G1_SIZE: usize = 64;
    const FR_SIZE: usize = 32;
    let public_input_size: usize = FR_SIZE * num_public_inputs;
    Manifest::new(vec![
        RoundManifest {
            elements: vec![ManifestEntry {
                name: "circuit_size".to_string(),
                num_bytes: 4,
                derived_by_verifier: false,
                challenge_map_index: 0,
            }],
            challenge: "init".to_string(),
            num_challenges: 1,
            map_challenges: false,
        },
        RoundManifest {
            elements: vec![],
            challenge: "eta".to_string(),
            num_challenges: 0,
            map_challenges: false,
        },
        RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "public_inputs".to_string(),
                    num_bytes: public_input_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "W_1".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "W_2".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "W_3".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
            ],
            challenge: "beta".to_string(),
            num_challenges: 2,
            map_challenges: false,
        },
        RoundManifest {
            elements: vec![ManifestEntry {
                name: "Z_PERM".to_string(),
                num_bytes: G1_SIZE,
                derived_by_verifier: false,
                challenge_map_index: 0,
            }],
            challenge: "alpha".to_string(),
            num_challenges: 1,
            map_challenges: false,
        },
        RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "T_1".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "T_2".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "T_3".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
            ],
            challenge: "z".to_string(),
            num_challenges: 1,
            map_challenges: false,
        },
        RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "w_1".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 1,
                },
                ManifestEntry {
                    name: "w_2".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 2,
                },
                ManifestEntry {
                    name: "w_3".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 3,
                },
                ManifestEntry {
                    name: "w_3_omega".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 4,
                },
                ManifestEntry {
                    name: "z_perm_omega".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 5,
                },
                ManifestEntry {
                    name: "sigma_1".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 6,
                },
                ManifestEntry {
                    name: "sigma_2".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 7,
                },
                ManifestEntry {
                    name: "r".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 8,
                },
                ManifestEntry {
                    name: "t".to_string(),
                    num_bytes: FR_SIZE,
                    derived_by_verifier: true,
                    challenge_map_index: -1,
                },
            ],
            challenge: "nu".to_string(),
            num_challenges: 10,
            map_challenges: true,
        },
        RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "PI_Z".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "PI_Z_OMEGA".to_string(),
                    num_bytes: G1_SIZE,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
            ],
            challenge: "separator".to_string(),
            num_challenges: 1,
            map_challenges: false,
        },
    ])
}

fn generate_test_data<H: BarretenHasher + Default + 'static>(
    n: usize,
) -> Prover<H, StandardSettings<H>> {
    // create some constraints that satisfy our arithmetic circuit relation
    let reference_string = Arc::new(RwLock::new(
        FileReferenceString::new(n + 1, "./src/srs_db/ignition").unwrap(),
    ));
    let key = Arc::new(RwLock::new(ProvingKey::new(
        n,
        0,
        reference_string,
        ComposerType::Standard,
    )));
    let mut w_l = Polynomial::new(n);
    let mut w_r = Polynomial::new(n);
    let mut w_o = Polynomial::new(n);
    let mut q_l = Polynomial::new(n);
    let mut q_r = Polynomial::new(n);
    let mut q_o = Polynomial::new(n);
    let mut q_c = Polynomial::new(n);
    let mut q_m = Polynomial::new(n);

    let mut rand = rand::thread_rng();
    for i in 0..n / 4 {
        w_l.coefficients[i * 2] = Fr::rand(&mut rand);
        w_r.coefficients[i * 2] = Fr::rand(&mut rand);
        w_o.coefficients[i * 2] = w_l.coefficients[i * 2] * w_r.coefficients[i * 2];
        w_o.coefficients[i * 2] += w_l.coefficients[i * 2];
        w_o.coefficients[i * 2] += w_r.coefficients[i * 2];
        w_o.coefficients[i * 2] = Fr::one() + w_o.coefficients[i * 2];
        q_l.coefficients[i * 2] = Fr::one();
        q_r.coefficients[i * 2] = Fr::one();
        q_o.coefficients[i * 2] = -Fr::one();
        q_c.coefficients[i * 2] = Fr::one();
        q_m.coefficients[i * 2] = Fr::one();

        w_l.coefficients[i * 2 + 1] = Fr::rand(&mut rand);
        w_r.coefficients[i * 2 + 1] = Fr::rand(&mut rand);
        w_o.coefficients[i * 2 + 1] = Fr::rand(&mut rand);

        let t0 = w_l.coefficients[i * 2 + 1] + w_r.coefficients[i * 2 + 1];
        q_c.coefficients[i * 2 + 1] = t0 + w_o.coefficients[i * 2 + 1];
        q_c.coefficients[i * 2 + 1] = -q_c.coefficients[i * 2 + 1];
        q_l.coefficients[i * 2 + 1] = Fr::one();
        q_r.coefficients[i * 2 + 1] = Fr::one();
        q_o.coefficients[i * 2 + 1] = Fr::one();
        q_m.coefficients[i * 2 + 1] = Fr::zero();
    }

    let shift = n / 2;
    w_l.coefficients[shift..].copy_within(..shift, shift);
    w_r.coefficients[shift..].copy_within(..shift, shift);
    w_o.coefficients[shift..].copy_within(..shift, shift);
    q_m.coefficients[shift..].copy_within(..shift, shift);
    q_l.coefficients[shift..].copy_within(..shift, shift);
    q_r.coefficients[shift..].copy_within(..shift, shift);
    q_o.coefficients[shift..].copy_within(..shift, shift);
    q_c.coefficients[shift..].copy_within(..shift, shift);

    // create basic permutation - second half of witness vector is a copy of the first half
    let mut sigma_1_mapping = vec![0; n];
    let mut sigma_2_mapping = vec![0; n];
    let mut sigma_3_mapping = vec![0; n];

    for i in 0..(n / 2) {
        sigma_1_mapping[shift + i] = i as u32;
        sigma_2_mapping[shift + i] = (i + (1 << 30)) as u32;
        sigma_3_mapping[shift + i] = (i + (1 << 31)) as u32;
        sigma_1_mapping[i] = (i + shift) as u32;
        sigma_2_mapping[i] = (i + shift + (1 << 30)) as u32;
        sigma_3_mapping[i] = (i + shift + (1 << 31)) as u32;
    }

    // make last permutation the same as identity permutation
    sigma_1_mapping[shift - 1] = (shift - 1) as u32;
    sigma_2_mapping[shift - 1] = (shift - 1 + (1 << 30)) as u32;
    sigma_3_mapping[shift - 1] = (shift - 1 + (1 << 31)) as u32;
    sigma_1_mapping[n - 1] = (n - 1) as u32;
    sigma_2_mapping[n - 1] = (n - 1 + (1 << 30)) as u32;
    sigma_3_mapping[n - 1] = (n - 1 + (1 << 31)) as u32;

    {
        let mut key_locked = key.write().unwrap();

        let mut sigma_1 = Polynomial::new(key_locked.circuit_size);
        let mut sigma_2 = Polynomial::new(key_locked.circuit_size);
        let mut sigma_3 = Polynomial::new(key_locked.circuit_size);

        compute_permutation_lagrange_base_single::<H, Fr>(
            &mut sigma_1,
            &sigma_1_mapping,
            &key_locked.small_domain,
        );
        compute_permutation_lagrange_base_single::<H, Fr>(
            &mut sigma_2,
            &sigma_2_mapping,
            &key_locked.small_domain,
        );
        compute_permutation_lagrange_base_single::<H, Fr>(
            &mut sigma_3,
            &sigma_3_mapping,
            &key_locked.small_domain,
        );

        let sigma_1_lagrange_base = sigma_1.clone();
        let sigma_2_lagrange_base = sigma_2.clone();
        let sigma_3_lagrange_base = sigma_3.clone();

        key_locked
            .polynomial_store
            .insert(&"sigma_1_lagrange".to_string(), sigma_1_lagrange_base);
        key_locked
            .polynomial_store
            .insert(&"sigma_2_lagrange".to_string(), sigma_2_lagrange_base);
        key_locked
            .polynomial_store
            .insert(&"sigma_3_lagrange".to_string(), sigma_3_lagrange_base);

        key_locked
            .small_domain
            .ifft_inplace(&mut sigma_1.coefficients);
        key_locked
            .small_domain
            .ifft_inplace(&mut sigma_2.coefficients);
        key_locked
            .small_domain
            .ifft_inplace(&mut sigma_3.coefficients);

        const WIDTH: usize = 4;
        let mut sigma_1_fft = sigma_1.clone();
        sigma_1_fft.resize(key_locked.circuit_size * WIDTH, Fr::zero());
        let mut sigma_2_fft = sigma_2.clone();
        sigma_2_fft.resize(key_locked.circuit_size * WIDTH, Fr::zero());
        let mut sigma_3_fft = sigma_3.clone();
        sigma_3_fft.resize(key_locked.circuit_size * WIDTH, Fr::zero());

        key_locked
            .large_domain
            .coset_fft_inplace(&mut sigma_1_fft.coefficients[..]);
        key_locked
            .large_domain
            .coset_fft_inplace(&mut sigma_2_fft.coefficients[..]);
        key_locked
            .large_domain
            .coset_fft_inplace(&mut sigma_3_fft.coefficients[..]);

        key_locked
            .polynomial_store
            .insert(&"sigma_1".to_string(), sigma_1);
        key_locked
            .polynomial_store
            .insert(&"sigma_2".to_string(), sigma_2);
        key_locked
            .polynomial_store
            .insert(&"sigma_3".to_string(), sigma_3);

        key_locked
            .polynomial_store
            .insert(&"sigma_1_fft".to_string(), sigma_1_fft);
        key_locked
            .polynomial_store
            .insert(&"sigma_2_fft".to_string(), sigma_2_fft);
        key_locked
            .polynomial_store
            .insert(&"sigma_3_fft".to_string(), sigma_3_fft);

        w_l.coefficients[n - 1] = Fr::zero();
        w_r.coefficients[n - 1] = Fr::zero();
        w_o.coefficients[n - 1] = Fr::zero();
        q_c.coefficients[n - 1] = Fr::zero();
        q_l.coefficients[n - 1] = Fr::zero();
        q_r.coefficients[n - 1] = Fr::zero();
        q_o.coefficients[n - 1] = Fr::zero();
        q_m.coefficients[n - 1] = Fr::zero();

        w_l.coefficients[shift - 1] = Fr::zero();
        w_r.coefficients[shift - 1] = Fr::zero();
        w_o.coefficients[shift - 1] = Fr::zero();
        q_c.coefficients[shift - 1] = Fr::zero();

        key_locked
            .polynomial_store
            .insert(&"w_1_lagrange".to_string(), w_l);
        key_locked
            .polynomial_store
            .insert(&"w_2_lagrange".to_string(), w_r);
        key_locked
            .polynomial_store
            .insert(&"w_3_lagrange".to_string(), w_o);

        key_locked
            .small_domain
            .ifft_inplace(&mut q_l.coefficients[..]);
        key_locked
            .small_domain
            .ifft_inplace(&mut q_r.coefficients[..]);
        key_locked
            .small_domain
            .ifft_inplace(&mut q_o.coefficients[..]);
        key_locked
            .small_domain
            .ifft_inplace(&mut q_m.coefficients[..]);
        key_locked
            .small_domain
            .ifft_inplace(&mut q_c.coefficients[..]);

        let mut q_1_fft = q_l.clone();
        q_1_fft.resize(key_locked.circuit_size * 4, Fr::zero());
        let mut q_2_fft = q_r.clone();
        q_2_fft.resize(key_locked.circuit_size * 4, Fr::zero());
        let mut q_3_fft = q_o.clone();
        q_3_fft.resize(key_locked.circuit_size * 4, Fr::zero());
        let mut q_m_fft = q_m.clone();
        q_m_fft.resize(key_locked.circuit_size * 4, Fr::zero());
        let mut q_c_fft = q_c.clone();
        q_c_fft.resize(key_locked.circuit_size * 4, Fr::zero());

        key_locked
            .large_domain
            .coset_fft_inplace(&mut q_1_fft.coefficients[..]);
        key_locked
            .large_domain
            .coset_fft_inplace(&mut q_2_fft.coefficients[..]);
        key_locked
            .large_domain
            .coset_fft_inplace(&mut q_3_fft.coefficients[..]);
        key_locked
            .large_domain
            .coset_fft_inplace(&mut q_m_fft.coefficients[..]);
        key_locked
            .large_domain
            .coset_fft_inplace(&mut q_c_fft.coefficients[..]);

        key_locked.polynomial_store.insert(&"q_1".to_string(), q_l);
        key_locked.polynomial_store.insert(&"q_2".to_string(), q_r);
        key_locked.polynomial_store.insert(&"q_3".to_string(), q_o);
        key_locked.polynomial_store.insert(&"q_m".to_string(), q_m);
        key_locked.polynomial_store.insert(&"q_c".to_string(), q_c);

        key_locked
            .polynomial_store
            .insert(&"q_1_fft".to_string(), q_1_fft);
        key_locked
            .polynomial_store
            .insert(&"q_2_fft".to_string(), q_2_fft);
        key_locked
            .polynomial_store
            .insert(&"q_3_fft".to_string(), q_3_fft);
        key_locked
            .polynomial_store
            .insert(&"q_m_fft".to_string(), q_m_fft);
        key_locked
            .polynomial_store
            .insert(&"q_c_fft".to_string(), q_c_fft);
    }
    let permutation_widget: Box<ProverPermutationWidget<H, 3, false, 4>> =
        Box::new(ProverPermutationWidget::<H, 3, false, 4>::new(key.clone()));

    let widget: Box<ProverArithmeticWidget<H>> =
        Box::new(ProverArithmeticWidget::<H>::new(key.clone()));

    let kate_commitment_scheme = KateCommitmentScheme::<H, Fq, Fr, G1Affine>::new();

    let mut state: Prover<H, StandardSettings<H>> =
        Prover::new(Some(key), Some(create_manifest(0)), None);
    state.random_widgets.push(permutation_widget);
    state.transition_widgets.push(widget);
    state.commitment_scheme = kate_commitment_scheme;
    state
}

#[test]
fn compute_quotient_polynomial() -> anyhow::Result<()> {
    let n = 1 << 10;
    let mut state = generate_test_data::<Keccak256>(n);

    let mut rng = rand::thread_rng();

    state.execute_preamble_round(&mut rng)?;
    state.queue.process_queue()?;
    state.execute_first_round()?;
    state.queue.process_queue()?;
    state.execute_second_round(&mut rng)?;
    state.queue.process_queue()?;
    state.execute_third_round()?;
    state.queue.process_queue()?;

    // check that the max degree of our quotient polynomial is 3n
    for i in 0..n {
        assert_eq!(
            state.key.read().unwrap().quotient_polynomial_parts[3]
                .read()
                .unwrap()
                .coefficients[i],
            Fr::zero()
        );
    }
    Ok(())
}
