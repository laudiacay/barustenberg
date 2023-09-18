use std::sync::{Arc, RwLock};

use ark_bn254::{Fq, Fr, G1Affine};
use ark_ff::{Field, One, UniformRand, Zero};
use rand::RngCore;

use super::{
    commitment_scheme::{CommitmentScheme, KateCommitmentScheme},
    proving_key::ProvingKey,
    types::{
        prover_settings::{Settings, StandardSettings},
        Proof,
    },
    widgets::{
        random_widgets::random_widget::ProverRandomWidget,
        transition_widgets::transition_widget::TransitionWidgetBase,
    },
};

use typenum::Unsigned;

use crate::{
    polynomials::{polynomial_arithmetic, Polynomial},
    proof_system::work_queue::{self, Work, WorkItem},
    transcript::{BarretenHasher, Manifest, Transcript},
};

use anyhow::{ensure, Result};

use crate::proof_system::work_queue::WorkQueue;

#[cfg(test)]
mod test;

// todo https://doc.rust-lang.org/reference/const_eval.html
/// Plonk prover.
#[derive(Debug)]
pub struct Prover<H: BarretenHasher> {
    pub(crate) circuit_size: usize,
    pub(crate) transcript: Arc<RwLock<Transcript<H>>>,
    pub(crate) key: Arc<RwLock<ProvingKey<Fr>>>,
    pub(crate) queue: WorkQueue<H>,
    pub(crate) random_widgets: Vec<Box<dyn ProverRandomWidget<Fr = Fr, G1 = G1Affine, Hasher = H>>>,
    pub(crate) transition_widgets: Vec<Box<dyn TransitionWidgetBase<Hasher = H, Field = Fr>>>,
    pub(crate) commitment_scheme: KateCommitmentScheme<StandardSettings<H>, H, Fq, Fr, G1Affine>,
    pub(crate) settings: StandardSettings<H>,
}

impl<H: BarretenHasher + Default> Prover<H> {
    /// Create a new prover.
    /// Parameters:
    /// - `input_key` Proving key.
    /// - `input_manifest` Manifest.
    /// - `input_settings` Program settings.
    /// Returns:
    /// - `Self` Prover.
    pub fn new(
        input_key: Option<Arc<RwLock<ProvingKey<Fr>>>>,
        input_manifest: Option<Manifest>,
        input_settings: Option<StandardSettings<H>>,
    ) -> Self {
        let circuit_size = input_key
            .as_ref()
            .map_or(0, |key| key.read().unwrap().circuit_size);
        let transcript = Arc::new(RwLock::new(Transcript::new(
            input_manifest,
            H::PrngOutputSize::USIZE,
        )));
        let input_key = match input_key {
            Some(ik) => ik,
            None => Arc::new(RwLock::new(ProvingKey::default())),
        };
        let queue = WorkQueue::new(Some(input_key.clone()), Some(transcript.clone()));
        let settings = input_settings.unwrap_or_default();

        Self {
            circuit_size,
            transcript,
            key: input_key,
            queue,
            random_widgets: Vec::new(),
            transition_widgets: Vec::new(),
            commitment_scheme:
                KateCommitmentScheme::<StandardSettings<H>, H, Fq, Fr, G1Affine>::new(
                    settings.clone(),
                ),
            settings,
        }
    }
}

impl<H: BarretenHasher + Default> Prover<H> {
    fn _copy_placeholder(&self) {
        todo!("LOOK AT THE COMMENTS IN PROVERBASE");
    }

    /// Execute preamble round.
    /// - Execute init round
    /// - Add randomness to the wire witness polynomials for Honest-Verifier Zero Knowledge.
    ///  N.B. Maybe we need to refactor this, since before we execute this function wires are in lagrange basis
    /// and after they are in monomial form. This is an inconsistency that can mislead developers.
    /// Parameters:
    /// - `settings` Program settings.
    fn execute_preamble_round(&mut self, rng: &mut dyn RngCore) -> Result<()> {
        self.queue.flush_queue();

        (*self.transcript).write().unwrap().add_element(
            "circuit_size",
            vec![
                (self.circuit_size >> 24) as u8,
                (self.circuit_size >> 16) as u8,
                (self.circuit_size >> 8) as u8,
                (self.circuit_size) as u8,
            ],
        );

        (*self.transcript).write().unwrap().add_element(
            "public_input_size",
            vec![
                (self.key.read().unwrap().num_public_inputs >> 24) as u8,
                (self.key.read().unwrap().num_public_inputs >> 16) as u8,
                (self.key.read().unwrap().num_public_inputs >> 8) as u8,
                (self.key.read().unwrap().num_public_inputs) as u8,
            ],
        );

        (*self.transcript)
            .write()
            .unwrap()
            .apply_fiat_shamir("init");

        // If this is a plookup proof, do not queue up an ifft on W_4 - we can only finish computing
        // the lagrange-base values in W_4 once eta has been generated.
        // This is because of the RAM/ROM subprotocol, which adds witnesses into W_4 that depend on eta
        let end = if self.settings.is_plookup() {
            self.settings.program_width() - 1
        } else {
            self.settings.program_width()
        };
        for i in 0..end {
            let wire_tag = format!("w_{}", i + 1);
            let wire_lagrange = self
                .key
                .read()
                .unwrap()
                .polynomial_store
                .get(&format!("{}_lagrange", wire_tag))?
                .clone();
            let mut wire_lagrange = wire_lagrange.write().unwrap();

            /*
            Adding zero knowledge to the witness polynomials.
            */
            // To ensure that PLONK is honest-verifier zero-knowledge, we need to ensure that the witness polynomials
            // and the permutation polynomial look uniformly random to an adversary. To make the witness polynomials
            // a(X), b(X) and c(X) uniformly random, we need to add 2 random blinding factors into each of them.
            // i.e. a'(X) = a(X) + (r_1X + r_2)
            // where r_1 and r_2 are uniformly random scalar FieldExt elements. A natural question is:
            // Why do we need 2 random scalars in witness polynomials? The reason is: our witness polynomials are
            // evaluated at only 1 point (\scripted{z}), so adding a random degree-1 polynomial suffices.
            //
            // NOTE: In TurboPlonk and UltraPlonk, the witness polynomials are evaluated at 2 points and thus
            // we need to add 3 random scalars in them.
            //
            // We start adding random scalars in `wire` polynomials from index (n - k) upto (n - k + 2).
            // For simplicity, we add 3 random scalars even for standard plonk (recall, just 2 of them are required)
            // since an additional random scalar would not affect things.
            //
            // NOTE: If in future there is a need to cut off more zeros off the vanishing polynomial, this method
            // will not change. This must be changed only if the number of evaluations of witness polynomials
            // change.
            let w_randomness: usize = 3;
            ensure!(w_randomness < self.settings.num_roots_cut_out_of_vanishing_polynomial());
            for k in 0..w_randomness {
                wire_lagrange.set_coefficient(
                    self.circuit_size - self.settings.num_roots_cut_out_of_vanishing_polynomial()
                        + k,
                    Fr::rand(rng),
                )
            }
        }

        // perform an IFFT so that the "w_i" polynomial cache will contain the monomial form
        for i in 0..end {
            let wire_tag = format!("w_{}", i + 1);
            self.queue.add_to_queue(WorkItem {
                work: work_queue::Work::Ifft,
                tag: wire_tag,
            });
        }
        Ok(())
    }

    /// Execute the first round:
    /// - Compute wire commitments.
    /// - Add public input values to the transcript.
    ///
    /// N.B. Random widget precommitments aren't actually being computed, since we are using permutation widget
    /// which only does computation in compute_random_commitments function if the round is 3.
    fn execute_first_round(&mut self) -> Result<()> {
        // note that there were a lot of debug timing things here and i removed them because they were a mess

        self.queue.flush_queue();

        self.compute_wire_commitments()?;

        for widget in self.random_widgets.iter() {
            widget.compute_round_commitments(
                &mut (*self.transcript).write().unwrap(),
                1,
                &mut self.queue,
            )?;
        }
        Ok(())
    }

    /// Execute the second round:
    /// - Apply Fiat-Shamir transform to generate the "eta" challenge.
    /// - Compute the random_widgets' round commitments that need to be computed at round 2.
    /// - If using plookup, we compute some w_4 values here (for gates which access "memory"), and apply blinding factors,
    ///   before finally committing to w_4.
    fn execute_second_round(&mut self, rng: &mut dyn RngCore) -> Result<()> {
        self.queue.flush_queue();

        (*self.transcript).write().unwrap().apply_fiat_shamir("eta");

        for widget in self.random_widgets.iter() {
            widget.compute_round_commitments(
                &mut (*self.transcript).write().unwrap(),
                2,
                &mut self.queue,
            )?;
        }

        // RAM/ROM memory subprotocol requires eta is generated before w_4 is comitted
        if self.settings.is_plookup() {
            self.add_plookup_memory_records_to_w_4()?;
            let wire_tag = "w_4";

            let w_4_lagrange = self
                .key
                .read()
                .unwrap()
                .polynomial_store
                .get(&format!("{}_lagrange", wire_tag))?;
            let mut w_4_lagrange = w_4_lagrange.write().unwrap();

            // add randomness to w_4_lagrange
            let w_randomness = 3;
            ensure!(w_randomness < self.settings.num_roots_cut_out_of_vanishing_polynomial());
            for k in 0..w_randomness {
                // Blinding
                w_4_lagrange.set_coefficient(
                    self.circuit_size - self.settings.num_roots_cut_out_of_vanishing_polynomial()
                        + k,
                    Fr::rand(rng),
                );
            }

            // compute poly w_4 from w_4_lagrange and add it to the cache
            let mut w_4 = w_4_lagrange.clone();
            self.key
                .read()
                .unwrap()
                .small_domain
                .ifft_inplace(&mut w_4.coefficients);
            self.key
                .write()
                .unwrap()
                .polynomial_store
                .put(wire_tag.to_string(), w_4);

            // commit to w_4 using the monomial srs.
            self.queue.add_to_queue(WorkItem {
                work: work_queue::Work::ScalarMultiplication {
                    mul_scalars: self
                        .key
                        .read()
                        .unwrap()
                        .polynomial_store
                        .get(&wire_tag.to_string())?,
                    constant: Fr::from((self.key.read().unwrap().circuit_size + 1) as u64),
                },
                tag: "W_4".to_owned(),
            });
        }
        Ok(())
    }

    /// Execute the third round:
    /// - Apply Fiat-Shamir transform on the "beta" challenge.
    /// - Apply 3rd round random widgets.
    /// - FFT the wires.
    ///
    /// *For example, standard composer executes permutation widget for z polynomial construction at this round.
    fn execute_third_round(&mut self) -> Result<()> {
        self.queue.flush_queue();

        (*self.transcript)
            .write()
            .unwrap()
            .apply_fiat_shamir("beta");

        for widget in &mut self.random_widgets {
            widget.compute_round_commitments(
                &mut (*self.transcript).write().unwrap(),
                3,
                &mut self.queue,
            )?;
        }

        for i in 0..self.settings.program_width() {
            let wire_tag = format!("w_{}", i + 1);
            self.queue.add_to_queue(WorkItem {
                work: Work::Fft { index: 0 },
                tag: wire_tag,
            });
        }
        Ok(())
    }

    /// Computes the quotient polynomial, then commits to its degree-n split parts.
    fn execute_fourth_round(&mut self, rng: &mut dyn RngCore) -> Result<()> {
        self.queue.flush_queue();
        (*self.transcript)
            .write()
            .unwrap()
            .apply_fiat_shamir("alpha");

        let mut alpha_base = (*self.transcript)
            .write()
            .unwrap()
            .get_challenge_field_element("alpha", None);

        // Compute FFT of lagrange polynomial L_1 (needed in random widgets only)
        self.compute_lagrange_1_fft()?;

        for widget in &mut self.random_widgets {
            alpha_base = widget
                .compute_quotient_contribution(alpha_base, &self.transcript.read().unwrap())?;
        }

        for widget in &mut self.transition_widgets {
            alpha_base = widget.compute_quotient_contribution(
                alpha_base,
                &self.transcript.read().unwrap(),
                rng,
            );
        }

        // The parts of the quotient polynomial t(X) are stored as 4 separate polynomials in
        // the code. However, operations such as dividing by the pseudo vanishing polynomial
        // as well as iFFT (coset) are to be performed on the polynomial t(X) as a whole.
        // We avoid redundant copy of the parts t_1, t_2, t_3, t_4 and instead just tweak the
        // relevant functions to work on quotient polynomial parts.
        // TODO this does not work so good in rust. for now, we copy... is it still okay to do this?
        let mut quotient_poly_parts: Vec<&mut [Fr]> = Vec::new();
        {
            let key = self.key.read().unwrap();
            let poly0 = (*key.quotient_polynomial_parts[0]).write().unwrap();
            let mut poly_sliced0 = [poly0[0]];
            quotient_poly_parts.push(&mut poly_sliced0);
            let poly1 = (*key.quotient_polynomial_parts[1]).write().unwrap();
            let mut poly_sliced1 = [poly1[0]];
            quotient_poly_parts.push(&mut poly_sliced1);
            let poly2 = (*key.quotient_polynomial_parts[2]).write().unwrap();
            let mut poly_sliced2 = [poly2[0]];
            quotient_poly_parts.push(&mut poly_sliced2);
            let poly3 = (*key.quotient_polynomial_parts[3]).write().unwrap();
            let mut poly_sliced3 = [poly3[0]];
            quotient_poly_parts.push(&mut poly_sliced3);

            self.key
                .read()
                .unwrap()
                .small_domain
                .divide_by_pseudo_vanishing_polynomial(
                    quotient_poly_parts.as_mut_slice(),
                    &self.key.read().unwrap().large_domain,
                    0,
                )?;

            self.key
                .read()
                .unwrap()
                .large_domain
                .coset_ifft_vec(quotient_poly_parts.as_mut_slice());
        }
        // Manually copy the (n + 1)th coefficient of t_3 for StandardPlonk from t_4.
        // This is because the degree of t_3 for StandardPlonk is n.
        if self.settings.program_width() == 3 {
            self.key.read().unwrap().quotient_polynomial_parts[2]
                .write()
                .unwrap()[self.circuit_size] = self.key.read().unwrap().quotient_polynomial_parts
                [3]
            .write()
            .unwrap()[0];
            self.key.read().unwrap().quotient_polynomial_parts[3]
                .write()
                .unwrap()[0] = Fr::zero();
        }

        self.add_blinding_to_quotient_polynomial_parts(rng);

        self.compute_quotient_commitments();
        Ok(())
    }
    fn execute_fifth_round(&mut self) -> Result<()> {
        self.queue.flush_queue();
        (*self.transcript).write().unwrap().apply_fiat_shamir("z"); // end of 4th round
        self.compute_quotient_evaluation()
    }

    fn execute_sixth_round(&mut self) {
        self.queue.flush_queue();
        (*self.transcript).write().unwrap().apply_fiat_shamir("nu");
        self.commitment_scheme.batch_open(
            &(*self.transcript).read().unwrap(),
            &mut self.queue,
            Some(self.key.clone()),
        );
    }

    /// note that this is never defined in barettenberg
    fn _add_polynomial_evaluations_to_transcript(&self) {
        todo!("yeehaw")
    }
    /// note that this is never defined in barettenberg
    fn _compute_batch_opening_polynomials(&self) {
        todo!("yeehaw")
    }
    /// - Compute wire commitments and add them to the transcript.
    /// - Add public_inputs from w_2_fft to transcript.
    fn compute_wire_commitments(&mut self) -> Result<()> {
        // Compute wire commitments
        let end: usize = if self.settings.is_plookup() {
            self.settings.program_width() - 1
        } else {
            self.settings.program_width()
        };
        let key = self.key.read().unwrap();
        for i in 0..end {
            let wire_tag = format!("w_{}", i + 1);
            let commit_tag = format!("W_{}", i + 1);
            let coefficients = key.polynomial_store.get(&wire_tag)?;

            // This automatically saves the computed point to the transcript
            let domain_size_flag = if i > 2 {
                key.circuit_size
            } else {
                key.circuit_size + 1
            };
            self.commitment_scheme.commit(
                coefficients,
                commit_tag,
                Fr::from(domain_size_flag as u64),
                &mut self.queue,
            );
        }

        // add public inputs
        let public_wires_source = key.polynomial_store.get(&"w_2_lagrange".to_string())?;
        let mut public_wires = vec![];
        for i in 0..key.num_public_inputs {
            public_wires.push(public_wires_source.read().unwrap()[i]);
        }
        (*self.transcript)
            .write()
            .unwrap()
            .put_field_element_vector("public_inputs", &public_wires);
        Ok(())
    }

    /// In this method, we compute the commitments to polynomials t_{low}(X), t_{mid}(X) and t_{high}(X).
    /// Recall, the quotient polynomial t(X) = t_{low}(X) + t_{mid}(X).X^n + t_{high}(X).X^{2n}
    ///
    /// The reason we split t(X) into three degree-n polynomials is because:
    ///  (i) We want the opening proof polynomials bounded by degree n as the opening algorithm of the
    ///      polynomial commitment scheme results in O(n) prover computation.
    /// (ii) The size of the srs restricts us to compute commitments to polynomials of degree n
    ///      (and disallows for degree 2n and 3n for large n).
    ///
    /// The degree of t(X) is determined by the term:
    /// ((a(X) + βX + γ) (b(X) + βk_1X + γ) (c(X) + βk_2X + γ)z(X)) / Z*_H(X).
    ///
    /// Let k = num_roots_cut_out_of_vanishing_polynomial, we have
    /// deg(t) = (n - 1) * (program_width + 1) - (n - k)
    ///        = n * program_width - program_width - 1 + k
    ///
    /// Since we must cut at least 4 roots from the vanishing polynomial
    /// (refer to ./src/barretenberg/plonk/proof_system/widgets/random_widgets/permutation_widget_impl.hpp/L247),
    /// k = 4 => deg(t) = n * program_width - program_width + 3
    ///
    /// For standard plonk, program_width = 3 and thus, deg(t) = 3n. This implies that there would be
    /// (3n + 1) coefficients of t(X). Now, splitting them into t_{low}(X), t_{mid}(X) and t_{high}(X),
    /// t_{high} will have (n+1) coefficients while t_{low} and t_{mid} will have n coefficients.
    /// This means that to commit t_{high}, we need a multi-scalar multiplication of size (n+1).
    /// Thus, we first compute the commitments to t_{low}(X), t_{mid}(X) using n multi-scalar multiplications
    /// each and separately compute commitment to t_{high} which is of size (n + 1).
    /// Note that this must be done only when program_width = 3.
    ///
    ///
    /// NOTE: If in future there is a need to cut off more zeros off the vanishing polynomial, the degree of
    /// the quotient polynomial t(X) will increase, so the degrees of t_{high}, t_{mid}, t_{low} could also
    /// increase according to the type of the composer type we are using. Currently, for TurboPLONK and Ultra-
    /// PLONK, the degree of t(X) is (4n - 1) and hence each t_{low}, t_{mid}, t_{high}, t_{higher} each is of
    /// degree (n - 1) (and thus contains n coefficients). Therefore, we are on the brink!
    /// If we need to cut out more zeros off the vanishing polynomial, sizes of coefficients of individual
    /// t_{i} would change and so we will have to ensure the correct size of multi-scalar multiplication in
    /// computing the commitments to these polynomials.
    ///
    fn compute_quotient_commitments(&mut self) {
        let key = self.key.read().unwrap();
        for i in 0..self.settings.program_width() {
            let coefficients = key.quotient_polynomial_parts[i].clone();
            let quotient_tag = format!("T_{}", i + 1);
            // Set flag that determines domain size (currently n or n+1) in pippenger (see process_queue()).
            // Note: After blinding, all t_i have size n+1 representation (degree n) except t_4 in Turbo/Ultra.
            let domain_size_flag = if i > 2 {
                key.circuit_size
            } else {
                key.circuit_size + 1
            };
            // TODO concerned about the fr::from on domain_size_flag. this may not be correct. bberg just uses equivalent of std::mem::transmute
            self.commitment_scheme.commit(
                coefficients,
                quotient_tag,
                Fr::from(domain_size_flag as u64),
                &mut self.queue,
            );
        }
    }
    fn _init_quotient_polynomials(&self) {
        todo!("yeehaw")
    }
    fn _compute_opening_elements(&self) {
        todo!("yeehaw")
    }
    fn add_plookup_memory_records_to_w_4(&mut self) -> Result<()> {
        // We can only compute memory record values once W_1, W_2, W_3 have been comitted to,
        // due to the dependence on the `eta` challenge.

        let eta: Fr = (*self.transcript).write().unwrap().get_field_element("eta");
        let key = self.key.read().unwrap();

        // We need the lagrange-base forms of the first 3 wires to compute the plookup memory record
        // value. w4 = w3 * eta^3 + w2 * eta^2 + w1 * eta + read_write_flag;
        // a RAM write. See plookup_auxiliary_widget.hpp for details)
        let w_1 = key.polynomial_store.get(&"w_1_lagrange".to_string())?;
        let w_1 = w_1.read().unwrap();
        let w_2 = key.polynomial_store.get(&"w_2_lagrange".to_string())?;
        let w_2 = w_2.read().unwrap();
        let w_3 = key.polynomial_store.get(&"w_3_lagrange".to_string())?;
        let w_3 = w_3.read().unwrap();
        let w_4 = key.polynomial_store.get(&"w_4_lagrange".to_string())?;
        let mut w_4 = w_4.write().unwrap();
        for gate_idx in key.memory_read_records.iter() {
            w_4[*gate_idx] += w_3[*gate_idx];
            w_4[*gate_idx] *= eta;
            w_4[*gate_idx] += w_2[*gate_idx];
            w_4[*gate_idx] *= eta;
            w_4[*gate_idx] += w_1[*gate_idx];
            w_4[*gate_idx] *= eta;
        }
        for gate_idx in key.memory_write_records.iter() {
            w_4[*gate_idx] += w_3[*gate_idx];
            w_4[*gate_idx] *= eta;
            w_4[*gate_idx] += w_2[*gate_idx];
            w_4[*gate_idx] *= eta;
            w_4[*gate_idx] += w_1[*gate_idx];
            w_4[*gate_idx] *= eta;
            w_4[*gate_idx] += Fr::one();
        }
        Ok(())
    }

    fn compute_quotient_evaluation(&self) -> Result<()> {
        let key = self.key.read().unwrap();

        let zeta = (*self.transcript)
            .write()
            .unwrap()
            .get_field_element("zeta");

        self.commitment_scheme
            .add_opening_evaluations_to_transcript(
                &mut (*self.transcript).write().unwrap(),
                Some(self.key.clone()),
                false,
            );

        let mut t_eval = polynomial_arithmetic::evaluate(
            &[
                key.quotient_polynomial_parts[0].read().unwrap()[0],
                key.quotient_polynomial_parts[1].read().unwrap()[0],
                key.quotient_polynomial_parts[2].read().unwrap()[0],
                key.quotient_polynomial_parts[3].read().unwrap()[0],
            ],
            &zeta,
            4 * self.circuit_size,
        );

        // TODO are these limbs wrong?
        let zeta_pow_n = zeta.pow([key.circuit_size as u64]);
        let mut scalar = zeta_pow_n;
        // Adjust the evaluation to consider the (n + 1)th coefficient when needed (note that width 3 is just an avatar for
        // StandardComposer here)
        let num_deg_n_poly = if self.settings.program_width() == 3 {
            self.settings.program_width()
        } else {
            self.settings.program_width() - 1
        };
        for j in 0..num_deg_n_poly {
            t_eval += key.quotient_polynomial_parts[j].read().unwrap()[key.circuit_size] * scalar;
            scalar *= zeta_pow_n;
        }

        (*self.transcript)
            .write()
            .unwrap()
            .add_field_element("t", &t_eval);
        Ok(())
    }

    /// Add blinding to the components in such a way that the full quotient would be unchanged if reconstructed
    fn add_blinding_to_quotient_polynomial_parts(&mut self, rng: &mut dyn RngCore) {
        // Construct blinded quotient polynomial parts t_i by adding randomness to the unblinded parts t_i' in
        // such a way that the full quotient polynomial t is unchanged upon reconstruction, i.e.
        //
        //        t = t_1' + X^n*t_2' + X^2n*t_3' + X^3n*t_4' = t_1 + X^n*t_2 + X^2n*t_3 + X^3n*t_4
        //
        // Blinding is done as follows, where b_i are random FieldExt elements:
        //
        //              t_1 = t_1' +       b_0*X^n
        //              t_2 = t_2' - b_0 + b_1*X^n
        //              t_3 = t_3' - b_1 + b_2*X^n
        //              t_4 = t_4' - b_2
        //
        // For details, please head to: https://hackmd.io/JiyexiqRQJW55TMRrBqp1g.
        let key = self.key.read().unwrap();
        for i in 0..self.settings.program_width() - 1 {
            // Note that only program_width-1 random elements are required for full blinding
            let quotient_randomness = Fr::rand(rng);

            key.quotient_polynomial_parts[i].write().unwrap()[key.circuit_size] +=
                quotient_randomness; // update coefficient of X^n'th term
            key.quotient_polynomial_parts[i + 1].write().unwrap()[0] -= quotient_randomness;
            // update constant coefficient
        }
    }

    /// Compute FFT of lagrange polynomial L_1 needed in random widgets only
    fn compute_lagrange_1_fft(&self) -> Result<()> {
        let mut lagrange_1_fft: Polynomial<Fr> = Polynomial::new(4 * self.circuit_size + 8);

        {
            let key = self.key.read().unwrap();
            key.small_domain
                .compute_lagrange_polynomial_fft(&mut lagrange_1_fft, &key.large_domain)?;
            for i in 0..8 {
                lagrange_1_fft[4 * self.circuit_size + i] = lagrange_1_fft[i];
            }
        }
        self.key
            .write()
            .unwrap()
            .polynomial_store
            .put("lagrange_1_fft".to_string(), lagrange_1_fft);

        Ok(())
    }

    /// export the proof from the prover's transcript
    pub fn export_proof(&self) -> Proof {
        Proof {
            proof_data: (*self.transcript).write().unwrap().export_transcript(),
        }
    }

    /// construct the proof from a fully initialized proof state
    pub fn construct_proof(&mut self) -> Result<Proof> {
        let mut rng = rand::thread_rng();

        // Execute init round. Randomize witness polynomials.
        self.execute_preamble_round(&mut rng)?;
        self.queue.process_queue()?;

        // Compute wire precommitments and sometimes random widget round commitments
        self.execute_first_round()?;
        self.queue.process_queue()?;

        // Fiat-Shamir eta + execute random widgets.
        self.execute_second_round(&mut rng)?;
        self.queue.process_queue()?;

        // Fiat-Shamir beta & gamma, execute random widgets (Permutation widget is executed here)
        // and fft the witnesses
        self.execute_third_round()?;
        self.queue.process_queue()?;

        // Fiat-Shamir alpha, compute & commit to quotient polynomial.
        self.execute_fourth_round(&mut rng)?;
        self.queue.process_queue()?;

        self.execute_fifth_round()?;

        self.execute_sixth_round();
        self.queue.process_queue()?;

        self.queue.flush_queue();

        Ok(self.export_proof())
    }

    fn get_circuit_size(&self) -> usize {
        todo!("implement me")
    }

    /// Reset the transcript to the initial state
    fn reset(&mut self) {
        let manifest = (*self.transcript).write().unwrap().get_manifest();
        *(*self.transcript).write().unwrap() = Transcript::<H>::new(
            Some(manifest),
            (*self.transcript).read().unwrap().num_challenge_bytes,
        );
    }
}