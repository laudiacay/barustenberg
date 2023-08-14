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
        transcript: &mut Transcript<H>,
        input_key:Option<Arc<RwLock<ProvingKey<Self::Fr>>>>,
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

        let zeta : Self::Fr = transcript.get_challenge_field_element("z", None);
        let shifted_z = zeta * input_key.small_domain.root;
        let n = input_key.small_domain.size;

        for i in 0..input_key.polynomial_manifest.len() {
            let info = input_key.polynomial_manifest[i.into()].clone();
            let poly = input_key.polynomial_store.get(&info.polynomial_label).unwrap();
            let poly = &poly.read().unwrap().coefficients;
            let poly_evaluation
        
            = if in_lagrange_form {
                input_key.small_domain.compute_barycentric_evaluation(poly,n,&zeta)
            } else {
                polynomial_arithmetic::evaluate(poly, &zeta, n)
            };

            transcript.add_field_element("poly_eval", &poly_evaluation);

            if info.requires_shifted_evaluation {
                let poly_evaluation = if in_lagrange_form {
                    // TODO is this a bug? Shouldn't we be using shifted_z instead of zeta?
                    input_key.small_domain.compute_barycentric_evaluation(poly,n,&zeta)
                }
                else {
                    polynomial_arithmetic::evaluate(poly, &shifted_z, n)
                };
                transcript.add_field_element( format!("{}_omega",info.polynomial_label).as_str(), &poly_evaluation);
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
        let input_key = input_key.read().unwrap();
         /*
    Compute batch opening polynomials according to the Kate commitment scheme.

    Step 1: Compute the polynomial F(X) s.t. W_{\zeta}(X) = (F(X) - F(\zeta))/(X - \zeta) defined in round 5 of the
    PLONK paper. Step 2: Compute the polynomial z(X) s.t. W_{\zeta \omega}(X) = (z(X) - z(\zeta \omega))/(X -
    \zeta.\omega). Step 3: Compute coefficient form of W_{\zeta}(X) and W_{\zeta \omega}(X). Step 4: Commit to
    W_{\zeta}(X) and W_{\zeta \omega}(X).
    */
    let opened_polynomials_at_zeta : Vec<(&Fr, Fr)> = Vec::new();
    let opened_polynomials_at_zeta_omega : Vec<(&Fr, Fr)> = Vec::new();

    // Add the following tuples to the above data structures:
    //
    // [a(X), nu_1], [b(X), nu_2], [c(X), nu_3],
    // [S_{\sigma_1}(X), nu_4], [S_{\sigma_2}(X), nu_5],
    // [z(X), nu_6]
    //
    // Note that the challenges nu_1, ..., nu_6 depend on the label of the respective polynomial.

    // Add challenge-poly tuples for all polynomials in the manifest
    for i in 0..input_key.polynomial_manifest.len() {
        let info = input_key.polynomial_manifest[i];
        let poly_label = info.polynomial_label;

        let poly = input_key.polynomial_store.get(&poly_label);

        let nu_challenge = transcript.get_challenge_field_element_from_map("nu", poly_label);
        opened_polynomials_at_zeta.push_back({ poly, nu_challenge });

        if (info.requires_shifted_evaluation) {
            let nu_challenge = transcript.get_challenge_field_element_from_map("nu", poly_label + "_omega");
            opened_polynomials_at_zeta_omega.push_back({ poly, nu_challenge });
        }
    }

    let zeta = transcript.get_challenge_field_element("z", None);

    // Note: the opening poly W_\frak{z} is always size (n + 1) due to blinding
    // of the quotient polynomial
    polynomial opening_poly(input_key.circuit_size + 1);
    polynomial shifted_opening_poly(input_key.circuit_size);

    // Add the tuples [t_{mid}(X), \zeta^{n}], [t_{high}(X), \zeta^{2n}]
    // Note: We don't need to include the t_{low}(X) since it is multiplied by 1 for combining with other witness
    // polynomials.
    //
    for i in 1..S::program_width() {
        let offset = i * input_key.small_domain.size;
        let scalar = zeta.pow(static_cast<uint64_t>(offset));
        opened_polynomials_at_zeta.push_back({ &input_key->quotient_polynomial_parts[i][0], scalar });
    }

    // Add up things to get coefficients of opening polynomials.
    ITERATE_OVER_DOMAIN_START(input_key->small_domain);
    opening_poly[i] = input_key->quotient_polynomial_parts[0][i];
    for (const auto& [poly, challenge] : opened_polynomials_at_zeta) {
        opening_poly[i] += poly[i] * challenge;
    }
    shifted_opening_poly[i] = 0;
    for (const auto& [poly, challenge] : opened_polynomials_at_zeta_omega) {
        shifted_opening_poly[i] += poly[i] * challenge;
    }
    ITERATE_OVER_DOMAIN_END;

    // Adjust the (n + 1)th coefficient of t_{0,1,2}(X) or r(X) (Note: t_4 (Turbo/Ultra) has only n coefficients)
    opening_poly[input_key->circuit_size] = 0;
    const fr zeta_pow_n = zeta.pow(static_cast<uint64_t>(input_key->circuit_size));

    let num_deg_n_poly = if S::program_width() == 3{
        S::program_width()
    } else {
        S::program_width() - 1
    };
    let scalar_mult = 1;
    for i in 0..num_deg_n_poly {
        opening_poly[input_key.circuit_size] +=
            input_key.quotient_polynomial_parts[i][input_key.circuit_size] * scalar_mult;
        scalar_mult *= zeta_pow_n;
    }

    // compute the shifted evaluation point \frak{z}*omega
    let zeta_omega = zeta * input_key.small_domain.root;

    // Compute the W_{\zeta}(X) and W_{\zeta \omega}(X) polynomials
    self.compute_opening_polynomial(&opening_poly[0], &opening_poly[0], zeta, input_key->circuit_size);
    self.compute_opening_polynomial(
        &shifted_opening_poly[0], &shifted_opening_poly[0], zeta_omega, input_key->circuit_size);

    input_key->polynomial_store.put("opening_poly", std::move(opening_poly));
    input_key->polynomial_store.put("shifted_opening_poly", std::move(shifted_opening_poly));

    // Commit to the opening and shifted opening polynomials
    self.commit(
        input_key->polynomial_store.get("opening_poly").get_coefficients(), "PI_Z", input_key->circuit_size, queue);
    self.commit(input_key->polynomial_store.get("shifted_opening_poly").get_coefficients(),
                                 "PI_Z_OMEGA",
                                 input_key->circuit_size,
                                 queue);
    }

    fn batch_verify(
        &self,
        _transcript: &Transcript<H>,
        _kate_g1_elements: &mut HashMap<String, G1Affine>,
        _kate_fr_elements: &mut HashMap<String, Fr>,
        _input_key: Option<&VerificationKey<Fr>>,
    ) {
            // Compute batch evaluation commitment [F]_1
    // In this method, we accumulate scalars and corresponding group elements for the multi-scalar
    // multiplication required in the steps 10 and 11 of the verifier in the PLONK paper.
    //
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
    //

    fr batch_eval(0);
    const auto& polynomial_manifest = input_key->polynomial_manifest;
    for (size_t i = 0; i < input_key->polynomial_manifest.size(); ++i) {
        const auto& item = polynomial_manifest[i];
        const std::string label(item.commitment_label);
        const std::string poly_label(item.polynomial_label);
        switch (item.source) {
        case PolynomialSource::WITNESS: {
            // add [a]_1, [b]_1, [c]_1 to the group elements' vector
            const auto element = transcript.get_group_element(label);
            // rule out bad points and points at infinity (just to be on the safe side. all-zero witnesses won't be
            // zero-knowledge!)
            if (!element.on_curve() || element.is_point_at_infinity()) {
                throw_or_abort("polynomial commitment to witness is not a valid point.");
            }
            kate_g1_elements.insert({ label, element });
            break;
        }
        case PolynomialSource::SELECTOR:
        case PolynomialSource::PERMUTATION: {
            // add [qL]_1, [qR]_1, [qM]_1, [qC]_1, [qO]_1, [\sigma_1]_1, [\sigma_2]_1, [\sigma_3]_1 to the commitments
            // map.
            const auto element = input_key->commitments.at(label);
            // selectors can be all zeros so infinity point is valid
            if (!element.on_curve()) {
                throw_or_abort("polynomial commitment to selector is not a valid point.");
            }
            kate_g1_elements.insert({ label, element });
            break;
        }
        case PolynomialSource::OTHER: {
            break;
        }
        }

        // We iterate over the polynomials in polynomial_manifest to add their commitments,
        // their scalar multiplicands and their evaluations in the respective vector maps.

        bool has_shifted_evaluation = item.requires_shifted_evaluation;

        fr kate_fr_scalar(0);
        if (has_shifted_evaluation) {

            // compute scalar additively for the batch opening commitment [F]_1
            const auto challenge = transcript.get_challenge_field_element_from_map("nu", poly_label + "_omega");
            const auto separator_challenge = transcript.get_challenge_field_element("separator", 0);
            kate_fr_scalar += (separator_challenge * challenge);

            // compute the batch evaluation scalar additively for the batch evaluation commitment [E]_1
            const auto poly_at_zeta_omega = transcript.get_field_element(poly_label + "_omega");
            batch_eval += separator_challenge * challenge * poly_at_zeta_omega;
        }

        // compute scalar additively for the batch opening commitment [F]_1
        const auto challenge = transcript.get_challenge_field_element_from_map("nu", poly_label);
        kate_fr_scalar += challenge;

        // compute the batch evaluation scalar additively for the batch evaluation commitment [E]_1
        const auto poly_at_zeta = transcript.get_field_element(poly_label);
        batch_eval += challenge * poly_at_zeta;

        kate_fr_elements.insert({ label, kate_fr_scalar });
    }

    const auto zeta = transcript.get_challenge_field_element("z");
    barretenberg::fr quotient_challenge = transcript.get_challenge_field_element_from_map("nu", "t");

    // append the commitments to the parts of quotient polynomial and their scalar multiplicands
    fr z_pow_n = zeta.pow(input_key->circuit_size);
    fr z_power = 1;
    for (size_t i = 0; i < settings::program_width; ++i) {
        std::string quotient_label = "T_" + std::to_string(i + 1);
        const auto element = transcript.get_group_element(quotient_label);

        kate_g1_elements.insert({ quotient_label, element });
        kate_fr_elements.insert({ quotient_label, quotient_challenge * z_power });
        z_power *= z_pow_n;
    }

    // add the quotient eval t_eval term to batch evaluation
    const auto quotient_eval = transcript.get_field_element("t");
    batch_eval += (quotient_eval * quotient_challenge);

    // append batch evaluation in the scalar element vector map
    kate_g1_elements.insert({ "BATCH_EVALUATION", g1::affine_one });
    kate_fr_elements.insert({ "BATCH_EVALUATION", -batch_eval });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_commitment_scheme() {
        todo!("see commitment_scheme.test.cpp")
    }
}
