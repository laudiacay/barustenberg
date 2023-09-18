use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use ark_bn254::{Fq, Fr, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field, One, Zero};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::polynomials::{polynomial_arithmetic, Polynomial};
use crate::proof_system::work_queue::{Work, WorkItem, WorkQueue};
use crate::transcript::{BarretenHasher, Transcript};

use super::proving_key::ProvingKey;
use super::types::polynomial_manifest::PolynomialSource;
use super::types::proof::CommitmentOpenProof;
use super::types::prover_settings::Settings;
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
    S: Settings<Hasher = H, Field = Fr, Group = G>,
    H: BarretenHasher,
    Fq: Field + FftField,
    Fr: Field + FftField,
    G: AffineRepr,
> {
    _kate_open_proof: CommitmentOpenProof,
    settings: S,
    phantom: PhantomData<(H, Fr, G, Fq)>,
}

impl<
        S: Settings<Hasher = H, Field = Fr, Group = G>,
        H: BarretenHasher,
        Fq: Field + FftField,
        Fr: Field + FftField,
        G: AffineRepr,
    > KateCommitmentScheme<S, H, Fq, Fr, G>
{
    pub(crate) fn new(settings: S) -> Self {
        Self {
            _kate_open_proof: CommitmentOpenProof::default(),
            settings,
            phantom: PhantomData,
        }
    }
}

impl<S: Settings<Hasher = H, Field = Fr, Group = G1Affine>, H: BarretenHasher> CommitmentScheme
    for KateCommitmentScheme<S, H, Fq, Fr, G1Affine>
{
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
        transcript: &mut Transcript<H>,
        input_key: Option<Arc<RwLock<ProvingKey<Self::Fr>>>>,
        in_lagrange_form: bool,
    ) {
        // In this function, we compute the evaluations of all polynomials in the polynomial manifest at the
        // evaluation challenge "zeta", as well as the needed evaluations at shifts.
        //
        // We also allow this evaluation computation for lagrange (evaluation) forms of polynomials instead of
        // the usual coefficient forms.

        //
        let input_key = input_key.unwrap();
        let input_key = input_key.read().unwrap();

        let zeta: Self::Fr = transcript.get_challenge_field_element("z", None);
        let shifted_z = zeta * input_key.small_domain.root;
        let n = input_key.small_domain.size;

        for i in 0..input_key.polynomial_manifest.len() {
            let info = input_key.polynomial_manifest[i.into()].clone();
            let poly = input_key
                .polynomial_store
                .get(&info.polynomial_label)
                .unwrap();
            let poly = &poly.read().unwrap().coefficients;
            let poly_evaluation = if in_lagrange_form {
                input_key
                    .small_domain
                    .compute_barycentric_evaluation(poly, n, &zeta)
            } else {
                polynomial_arithmetic::evaluate(poly, &zeta, n)
            };

            transcript.add_field_element("poly_eval", &poly_evaluation);

            if info.requires_shifted_evaluation {
                let poly_evaluation = if in_lagrange_form {
                    // TODO is this a bug? Shouldn't we be using shifted_z instead of zeta?
                    input_key
                        .small_domain
                        .compute_barycentric_evaluation(poly, n, &zeta)
                } else {
                    polynomial_arithmetic::evaluate(poly, &shifted_z, n)
                };
                transcript.add_field_element(
                    format!("{}_omega", info.polynomial_label).as_str(),
                    &poly_evaluation,
                );
            }
        }
    }

    fn compute_opening_polynomial(&self, src: &[Fr], dest: &mut [Fr], z_point: &Fr, n: usize) {
        // open({cm_i}, {cm'_i}, {z, z'}, {s_i, s'_i})

        // if `coeffs` represents F(X), we want to compute W(X)
        // where W(X) = F(X) - F(z) / (X - z)
        // i.e. divide by the degree-1 polynomial [-z, 1]

        // We assume that the commitment is well-formed and that there is no remainder term.
        // Under these conditions we can perform this polynomial division in linear time with good constants.
        // Note that the opening polynomial always has (n+1) coefficients for Standard/Turbo/Ultra due to
        // the blinding of the quotient polynomial parts.
        let f = polynomial_arithmetic::evaluate(src, z_point, n + 1);

        // compute (1 / -z)
        let divisor = -z_point.inverse().unwrap();

        // we're about to shove these coefficients into a pippenger multi-exponentiation routine, where we need
        // to convert out of montgomery form. So, we can use lazy reduction techniques here without triggering overflows
        dest[0] = src[0] - f;
        dest[0] *= divisor;
        for i in 1..n {
            dest[i] = src[i] - dest[i - 1];
            dest[i] *= divisor;
        }
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
        transcript: &Transcript<H>,
        queue: &mut WorkQueue<H>,
        input_key: Option<Arc<RwLock<ProvingKey<Self::Fr>>>>,
    ) {
        let input_key = input_key.unwrap();
        let mut input_key = input_key.write().unwrap();
        /*
        Compute batch opening polynomials according to the Kate commitment scheme.

        Step 1: Compute the polynomial F(X) s.t. W_{\zeta}(X) = (F(X) - F(\zeta))/(X - \zeta) defined in round 5 of the
        PLONK paper. Step 2: Compute the polynomial z(X) s.t. W_{\zeta \omega}(X) = (z(X) - z(\zeta \omega))/(X -
        \zeta.\omega). Step 3: Compute coefficient form of W_{\zeta}(X) and W_{\zeta \omega}(X). Step 4: Commit to
        W_{\zeta}(X) and W_{\zeta \omega}(X).
        */
        let mut opened_polynomials_at_zeta: Vec<(Arc<RwLock<Polynomial<Fr>>>, Fr)> = Vec::new();
        let mut opened_polynomials_at_zeta_omega: Vec<(Arc<RwLock<Polynomial<Fr>>>, Fr)> =
            Vec::new();

        // Add the following tuples to the above data structures:
        //
        // [a(X), nu_1], [b(X), nu_2], [c(X), nu_3],
        // [S_{\sigma_1}(X), nu_4], [S_{\sigma_2}(X), nu_5],
        // [z(X), nu_6]
        //
        // Note that the challenges nu_1, ..., nu_6 depend on the label of the respective polynomial.

        // Add challenge-poly tuples for all polynomials in the manifest
        for i in 0..input_key.polynomial_manifest.len() {
            let info = &input_key.polynomial_manifest[i.into()];
            let poly_label = &info.polynomial_label;

            let poly = input_key.polynomial_store.get(poly_label).unwrap();

            let nu_challenge = transcript.get_challenge_field_element_from_map("nu", poly_label);
            opened_polynomials_at_zeta.push((poly.clone(), nu_challenge));

            if info.requires_shifted_evaluation {
                let nu_challenge = transcript
                    .get_challenge_field_element_from_map("nu", &format!("{}_omega", poly_label));
                opened_polynomials_at_zeta_omega.push((poly, nu_challenge));
            }
        }

        let zeta: Fr = transcript.get_challenge_field_element("z", None);

        // Note: the opening poly W_\frak{z} is always size (n + 1) due to blinding
        // of the quotient polynomial
        let opening_poly: Arc<RwLock<Polynomial<Fr>>> =
            Arc::new(RwLock::new(Polynomial::new(input_key.circuit_size + 1)));
        let shifted_opening_poly: Arc<RwLock<Polynomial<Fr>>> =
            Arc::new(RwLock::new(Polynomial::new(input_key.circuit_size)));

        // Add the tuples [t_{mid}(X), \zeta^{n}], [t_{high}(X), \zeta^{2n}]
        // Note: We don't need to include the t_{low}(X) since it is multiplied by 1 for combining with other witness
        // polynomials.
        //
        for i in 1..self.settings.program_width() {
            let offset = i * input_key.small_domain.size;
            let scalar = zeta.pow([offset as u64]);
            opened_polynomials_at_zeta
                .push((input_key.quotient_polynomial_parts[i].clone(), scalar));
        }

        // Add up things to get coefficients of opening polynomials.
        // TODO THESE LOCKS ARE FUCKED. UP. THEY WILL BE UNDER HEAVY CONTENTION. FIX THEM
        (0..input_key.small_domain.size)
            .into_par_iter()
            .for_each(|i| {
                let mut opening_poly = opening_poly.write().unwrap();
                opening_poly[i] = input_key.quotient_polynomial_parts[0].read().unwrap()[i];
                for &(ref poly, challenge) in &opened_polynomials_at_zeta {
                    opening_poly[i] += (*poly).read().unwrap()[i] * challenge;
                }
                let mut shifted_opening_poly = shifted_opening_poly.write().unwrap();
                shifted_opening_poly[i] = Fr::zero();
                for &(ref poly, challenge) in &opened_polynomials_at_zeta_omega {
                    shifted_opening_poly[i] += (*poly).read().unwrap()[i] * challenge;
                }
            });

        let opening_poly = Arc::try_unwrap(opening_poly).unwrap();
        let opening_poly = opening_poly.into_inner();
        let mut opening_poly = opening_poly.unwrap();
        let shifted_opening_poly = Arc::try_unwrap(shifted_opening_poly).unwrap();
        let shifted_opening_poly = shifted_opening_poly.into_inner();
        let shifted_opening_poly = shifted_opening_poly.unwrap();

        // Adjust the (n + 1)th coefficient of t_{0,1,2}(X) or r(X) (Note: t_4 (Turbo/Ultra) has only n coefficients)
        opening_poly[input_key.circuit_size] = Fr::zero();
        let zeta_pow_n = zeta.pow([input_key.circuit_size as u64]);

        let num_deg_n_poly = if self.settings.program_width() == 3 {
            self.settings.program_width()
        } else {
            self.settings.program_width() - 1
        };
        let mut scalar_mult = Fr::one();
        for i in 0..num_deg_n_poly {
            opening_poly[input_key.circuit_size] +=
                input_key.quotient_polynomial_parts[i].read().unwrap()[input_key.circuit_size]
                    * scalar_mult;
            scalar_mult *= zeta_pow_n;
        }

        // Compute the shifted evaluation point \frak{z}*omega
        let zeta_omega = zeta * input_key.small_domain.root;

        // Compute the W_{\zeta}(X) and W_{\zeta \omega}(X) polynomials
        self.compute_opening_polynomial(
            &[opening_poly[0]],
            &mut [opening_poly[0]],
            &zeta,
            input_key.circuit_size,
        );
        self.compute_opening_polynomial(
            &[shifted_opening_poly[0]],
            &mut [shifted_opening_poly[0]],
            &zeta_omega,
            input_key.circuit_size,
        );

        input_key
            .polynomial_store
            .put("opening_poly".to_string(), opening_poly);
        input_key
            .polynomial_store
            .put("shifted_opening_poly".to_string(), shifted_opening_poly);

        // Commit to the opening and shifted opening polynomials
        self.commit(
            input_key
                .polynomial_store
                .get(&"opening_poly".to_string())
                .unwrap(),
            "PI_Z".to_owned(),
            Fr::from(input_key.circuit_size as u64),
            queue,
        );
        self.commit(
            input_key
                .polynomial_store
                .get(&"shifted_opening_poly".to_string())
                .unwrap(),
            "PI_Z_OMEGA".to_owned(),
            Fr::from(input_key.circuit_size as u64),
            queue,
        );
    }

    fn batch_verify(
        &self,
        transcript: &Transcript<H>,
        kate_g1_elements: &mut HashMap<String, G1Affine>,
        kate_fr_elements: &mut HashMap<String, Fr>,
        input_key: Option<&VerificationKey<Fr>>,
    ) {
        let mut batch_eval = Fr::zero();
        let polynomial_manifest = &input_key.as_ref().unwrap().polynomial_manifest;
        for i in 0..polynomial_manifest.len() {
            let item = &polynomial_manifest[i.into()];
            let label = item.commitment_label.clone();
            let poly_label = item.polynomial_label.clone();
            match item.source {
                PolynomialSource::Witness => {
                    let element: G1Affine = transcript.get_group_element(&label);
                    // removed || element.is_point_at_infinity()
                    if !element.is_on_curve() {
                        panic!("polynomial commitment to witness is not a valid point.");
                    }
                    kate_g1_elements.insert(label.clone(), element);
                }
                PolynomialSource::Selector | PolynomialSource::Permutation => {
                    let element = input_key.as_ref().unwrap().commitments.get(&label).unwrap();
                    if !element.is_on_curve() {
                        panic!("polynomial commitment to selector is not a valid point.");
                    }
                    kate_g1_elements.insert(label.clone(), *element);
                }
                PolynomialSource::Other => {}
            }

            let has_shifted_evaluation = item.requires_shifted_evaluation;
            let mut kate_fr_scalar = Fr::zero();
            if has_shifted_evaluation {
                let challenge: Fr = transcript
                    .get_challenge_field_element_from_map("nu", &format!("{}_omega", poly_label));
                let separator_challenge: Fr =
                    transcript.get_challenge_field_element("separator", Some(0));
                kate_fr_scalar += separator_challenge * challenge;
                let poly_at_zeta_omega: Fr =
                    transcript.get_field_element(&format!("{}_omega", poly_label));
                batch_eval += separator_challenge * challenge * poly_at_zeta_omega;
            }

            let challenge: Fr = transcript.get_challenge_field_element_from_map("nu", &poly_label);
            kate_fr_scalar += challenge;
            let poly_at_zeta: Fr = transcript.get_field_element(&poly_label);
            batch_eval += challenge * poly_at_zeta;
            kate_fr_elements.insert(label, kate_fr_scalar);
        }

        let zeta: Fr = transcript.get_challenge_field_element("z", None);
        let quotient_challenge: Fr = transcript.get_challenge_field_element_from_map("nu", "t");

        let z_pow_n = zeta.pow([input_key.as_ref().unwrap().circuit_size as u64]);
        let mut z_power = Fr::one();
        for i in 0..self.settings.program_width() {
            let quotient_label = format!("T_{}", i + 1);
            let element = transcript.get_group_element(&quotient_label);
            kate_g1_elements.insert(quotient_label.clone(), element);
            kate_fr_elements.insert(quotient_label, quotient_challenge * z_power);
            z_power *= z_pow_n;
        }

        let quotient_eval: Fr = transcript.get_field_element("t");
        batch_eval += quotient_eval * quotient_challenge;

        kate_g1_elements.insert("BATCH_EVALUATION".to_string(), G1Affine::identity());
        kate_fr_elements.insert("BATCH_EVALUATION".to_string(), -batch_eval);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_kate_open() {

        /*
        
        TEST(commitment_scheme, kate_open)
{
    // generate random polynomial F(X) = coeffs
    size_t n = 256;
    auto coeffs = polynomial(n + 1);
    for (size_t i = 0; i < n; ++i) {
        coeffs[i] = fr::random_element();
    }
    polynomial W(coeffs, n + 1);
    coeffs[n] = 0;

    // generate random evaluation point z
    fr z = fr::random_element();

    // compute opening polynomial W(X), and evaluation f = F(z)
    transcript::StandardTranscript inp_tx = transcript::StandardTranscript(transcript::Manifest());
    plonk::KateCommitmentScheme<turbo_settings> newKate;

    // std::shared_ptr<barretenberg::srs::factories::CrsFactory<curve::BN254>> crs_factory = (new
    // FileReferenceStringFactory("../srs_db/ignition"));
    auto file_crs = std::make_shared<barretenberg::srs::factories::FileCrsFactory<curve::BN254>>("../srs_db/ignition");
    auto crs = file_crs->get_prover_crs(n);
    auto circuit_proving_key = std::make_shared<proving_key>(n, 0, crs, CircuitType::STANDARD);
    work_queue queue(circuit_proving_key.get(), &inp_tx);

    newKate.commit(coeffs.data(), "F_COMM", n, queue);
    queue.process_queue();

    fr y = fr::random_element();
    fr f_y = polynomial_arithmetic::evaluate(&coeffs[0], y, n);
    fr f = polynomial_arithmetic::evaluate(&coeffs[0], z, n);

    newKate.compute_opening_polynomial(&coeffs[0], &W[0], z, n);
    newKate.commit(W.data(), "W_COMM", n, queue);
    queue.process_queue();

    // check if W(y)(y - z) = F(y) - F(z)
    fr w_y = polynomial_arithmetic::evaluate(&W[0], y, n - 1);
    fr y_minus_z = y - z;
    fr f_y_minus_f = f_y - f;

    EXPECT_EQ(w_y * y_minus_z, f_y_minus_f);
}
 */
        todo!()
    }

    #[test]
    fn kate_batch_open ( ) {
        /*
        // generate random evaluation points [z_1, z_2, ...]
    size_t t = 8;
    std::vector<fr> z_points(t);
    for (size_t k = 0; k < t; ++k) {
        z_points[k] = fr::random_element();
    }

    // generate random polynomials F(X) = coeffs
    //
    // z_1 -> [F_{1,1},  F_{1,2},  F_{1, 3},  ...,  F_{1, m}]
    // z_2 -> [F_{2,1},  F_{2,2},  F_{2, 3},  ...,  F_{2, m}]
    // ...
    // z_t -> [F_{t,1},  F_{t,2},  F_{t, 3},  ...,  F_{t, m}]
    //
    // Note that each polynomial F_{k, j} \in F^{n}
    //
    size_t n = 64;
    size_t m = 4;
    polynomial coeffs(n * m * t);
    for (size_t k = 0; k < t; ++k) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t i = 0; i < n; ++i) {
                coeffs[k * (m * n) + j * n + i] = fr::random_element();
            }
        }
    }

    // setting up the Kate commitment scheme class
    transcript::StandardTranscript inp_tx = transcript::StandardTranscript(transcript::Manifest());
    plonk::KateCommitmentScheme<turbo_settings> newKate;

    auto file_crs = std::make_shared<barretenberg::srs::factories::FileCrsFactory<curve::BN254>>("../srs_db/ignition");
    auto crs = file_crs->get_prover_crs(n);
    auto circuit_proving_key = std::make_shared<proving_key>(n, 0, crs, CircuitType::STANDARD);
    work_queue queue(circuit_proving_key.get(), &inp_tx);

    // commit to individual polynomials
    for (size_t k = 0; k < t; ++k) {
        for (size_t j = 0; j < m; ++j) {
            newKate.commit(coeffs.data(), "F_{" + std::to_string(k + 1) + ", " + std::to_string(j + 1) + "}", n, queue);
        }
    }
    queue.process_queue();

    // create random challenges, tags and item_constants
    std::vector<fr> challenges(t);
    std::vector<std::string> tags(t);
    std::vector<fr> item_constants(t);
    for (size_t k = 0; k < t; ++k) {
        challenges[k] = fr::random_element();
        tags[k] = "W_" + std::to_string(k + 1);
        item_constants[k] = n;
    }

    // compute opening polynomials W_1, W_2, ..., W_t
    std::vector<fr> W(n * t);
    newKate.generic_batch_open(
        &coeffs[0], &W[0], m, &z_points[0], t, &challenges[0], n, &tags[0], &item_constants[0], queue);
    queue.process_queue();

    // check if W_{k}(y) * (y - z_k) = \sum_{j} challenge[k]^{j - 1} * [F_{k, j}(y) - F_{k, j}(z_k)]
    fr y = fr::random_element();
    for (size_t k = 0; k < t; ++k) {

        // compute lhs
        fr W_k_at_y = polynomial_arithmetic::evaluate(&W[k * n], y, n);
        fr y_minus_z_k = y - z_points[k];
        fr lhs = W_k_at_y * y_minus_z_k;

        fr challenge_pow = fr(1);
        fr rhs = fr(0);
        for (size_t j = 0; j < m; ++j) {

            // compute evaluations of source polynomials at y and z_points
            fr f_kj_at_y = polynomial_arithmetic::evaluate(&coeffs[k * m * n + j * n], y, n);
            fr f_kj_at_z = polynomial_arithmetic::evaluate(&coeffs[k * m * n + j * n], z_points[k], n);

            // compute rhs
            fr f_term = f_kj_at_y - f_kj_at_z;
            rhs += challenge_pow * f_term;
            challenge_pow *= challenges[k];
        }

        EXPECT_EQ(lhs, rhs);
    }
         */
        todo!()
    }
}
