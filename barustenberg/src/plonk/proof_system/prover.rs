use std::{
    borrow::BorrowMut,
    fmt::format,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use super::{
    commitment_scheme::{CommitmentScheme, KateCommitmentScheme},
    proving_key::ProvingKey,
    types::{prover_settings::Settings, Proof},
    widgets::{
        random_widgets::random_widget::ProverRandomWidget,
        transition_widgets::transition_widget::TransitionWidget,
    },
};

use typenum::Unsigned;

use crate::{
    polynomials::{polynomial_arithmetic, Polynomial},
    proof_system::work_queue::{self, QueuedFftInputs, WorkItem},
    transcript::{BarretenHasher, Manifest, Transcript},
};

use anyhow::{ensure, Result};

use crate::proof_system::work_queue::WorkQueue;

// todo https://doc.rust-lang.org/reference/const_eval.html

pub(crate) struct Prover<
    'a,
    Fq: Field,
    Fr: Field + FftField,
    G1Affine: AffineRepr,
    H: BarretenHasher,
    S: Settings<H>,
    CS: CommitmentScheme<Fq, Fr, G1Affine, H>,
> {
    pub(crate) circuit_size: usize,
    pub(crate) transcript: Arc<Transcript<H, Fr, G1Affine>>,
    pub(crate) key: Arc<ProvingKey<'a, Fr, G1Affine>>,
    pub(crate) queue: WorkQueue<'a, H, Fr, G1Affine>,
    pub(crate) random_widgets: Vec<ProverRandomWidget<'a, H, Fr, G1Affine>>,
    pub(crate) transition_widgets: Vec<dyn TransitionWidget<'a, H, Fr, G1Affine, S, U1, IDK>>,
    pub(crate) commitment_scheme: CS,
    pub(crate) settings: S,
    pub(crate) rng: Box<dyn rand::RngCore>,
    phantom: PhantomData<Fq>,
}

impl<
        'a,
        Fq: Field,
        Fr: Field + FftField,
        G1Affine: AffineRepr,
        H: BarretenHasher + Default,
        S: Settings<H> + Default,
    > Prover<'a, Fq, Fr, G1Affine, H, S, KateCommitmentScheme<H, S>>
{
    pub(crate) fn new(
        input_key: Option<Arc<ProvingKey<'a, Fr, G1Affine>>>,
        input_manifest: Option<Manifest>,
        input_settings: Option<S>,
    ) -> Self {
        let circuit_size = input_key.as_ref().map_or(0, |key| key.circuit_size);
        let transcript = Arc::new(Transcript::new(input_manifest, H::PrngOutputSize::USIZE));
        let queue = WorkQueue::new(
            input_key.as_ref().map(|a| a.clone()),
            Some(transcript.clone()),
        );
        let settings = input_settings.unwrap_or_default();

        Self {
            circuit_size,
            transcript,
            key: input_key.unwrap_or_else(|| Arc::new(ProvingKey::default())),
            queue,
            random_widgets: Vec::new(),
            transition_widgets: Vec::new(),
            commitment_scheme: KateCommitmentScheme::<H, S>::default(),
            settings,
            phantom: PhantomData,
            rng: Box::new(rand::thread_rng()),
        }
    }
}

impl<
        'a,
        Fq: Field,
        Fr: Field + FftField,
        G1Affine: AffineRepr,
        H: BarretenHasher + Default,
        S: Settings<H> + Default,
        CS: CommitmentScheme<Fq, Fr, G1Affine, H>,
    > Prover<'a, Fq, Fr, G1Affine, H, S, CS>
{
    fn copy_placeholder(&self) {
        todo!("LOOK AT THE COMMENTS IN PROVERBASE");
    }

    /// Execute preamble round.
    /// - Execute init round
    /// - Add randomness to the wire witness polynomials for Honest-Verifier Zero Knowledge.
    ///  N.B. Maybe we need to refactor this, since before we execute this function wires are in lagrange basis
    /// and after they are in monomial form. This is an inconsistency that can mislead developers.
    /// Parameters:
    /// - `settings` Program settings.
    fn execute_preamble_round(&self) -> Result<()> {
        self.queue.flush_queue();

        self.transcript.add_element(
            "circuit_size",
            vec![
                (self.circuit_size >> 24) as u8,
                (self.circuit_size >> 16) as u8,
                (self.circuit_size >> 8) as u8,
                (self.circuit_size) as u8,
            ],
        );

        self.transcript.add_element(
            "public_input_size",
            vec![
                (self.key.num_public_inputs >> 24) as u8,
                (self.key.num_public_inputs >> 16) as u8,
                (self.key.num_public_inputs >> 8) as u8,
                (self.key.num_public_inputs) as u8,
            ],
        );

        self.transcript.apply_fiat_shamir("init");

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
                .polynomial_store
                .get(format!("{}_lagrange", wire_tag))?;

            /*
            Adding zero knowledge to the witness polynomials.
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
                    Fr::rand(&mut self.rng),
                )
            }

            self.key
                .polynomial_store
                .put(wire_tag + "_lagrange", wire_lagrange);
        }

        // perform an IFFT so that the "w_i" polynomial cache will contain the monomial form
        for i in 0..end {
            let wire_tag = format!("w_{}", i + 1);
            self.queue.add_to_queue(WorkItem {
                work_type: work_queue::WorkType::Ifft,
                mul_scalars: nullptr,
                tag: wire_tag,
                constant: 0,
                index: 0,
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
    fn execute_first_round(&self) {
        /*
                    queue.flush_queue();
        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        #endif
        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "init quotient polys: " << diff.count() << "ms" << std::endl;
        #endif
        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif
        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute wire coefficients: " << diff.count() << "ms" << std::endl;
        #endif
        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif
            compute_wire_commitments();

            for (auto& widget : random_widgets) {
                widget->compute_round_commitments(transcript, 1, queue);
            }
        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute wire commitments: " << diff.count() << "ms" << std::endl;
        #endif
                 */
        todo!("execute_first_round implementation");
    }

    /// Execute the second round:
    /// - Apply Fiat-Shamir transform to generate the "eta" challenge.
    /// - Compute the random_widgets' round commitments that need to be computed at round 2.
    /// - If using plookup, we compute some w_4 values here (for gates which access "memory"), and apply blinding factors,
    ///   before finally committing to w_4.
    fn execute_second_round(&mut self) -> Result<()> {
        self.queue.flush_queue();

        self.transcript.apply_fiat_shamir("eta");

        for widget in self.random_widgets {
            widget.compute_round_commitments(&mut self.transcript, 2, &mut self.queue);
        }

        // RAM/ROM memory subprotocol requires eta is generated before w_4 is comitted
        if self.settings.is_plookup() {
            self.add_plookup_memory_records_to_w_4();
            let wire_tag = "w_4";
            let w_4_lagrange = self
                .key
                .polynomial_store
                .get(format!("{}_lagrange", wire_tag))?;

            // add randomness to w_4_lagrange
            let w_randomness = 3;
            ensure!(w_randomness < self.settings.num_roots_cut_out_of_vanishing_polynomial());
            for k in 0..w_randomness {
                // Blinding
                w_4_lagrange.set_coefficient(
                    self.circuit_size - self.settings.num_roots_cut_out_of_vanishing_polynomial()
                        + k,
                    Fr::rand(&mut self.rng),
                );
            }

            // compute poly w_4 from w_4_lagrange and add it to the cache
            let mut w_4 = w_4_lagrange.clone();
            w_4.ifft(self.key.small_domain);
            self.key.polynomial_store.put(wire_tag.to_string(), w_4);

            // commit to w_4 using the monomial srs.
            self.queue.add_to_queue(WorkItem {
                work_type: work_queue::WorkType::ScalarMultiplication,
                mul_scalars: Arc::new(
                    self.key
                        .polynomial_store
                        .get(wire_tag.to_string())?
                        .get_coefficients(),
                ),
                tag: "W_4".to_owned(),
                constant: Fr::from((self.key.circuit_size + 1) as u64),
                index: 0,
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
    fn execute_third_round(&self) {
        // TODO
        /*
                queue.flush_queue();

            transcript.apply_fiat_shamir("beta");

        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        #endif
        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute z coefficients: " << diff.count() << "ms" << std::endl;
        #endif
        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif
            for (auto& widget : random_widgets) {
                widget->compute_round_commitments(transcript, 3, queue);
            }

            for (size_t i = 0; i < settings::program_width; ++i) {
                std::string wire_tag = "w_" + std::to_string(i + 1);
                queue.add_to_queue({
                    .work_type = work_queue::WorkType::FFT,
                    .mul_scalars = nullptr,
                    .tag = wire_tag,
                    .constant = barretenberg::fr(0),
                    .index = 0,
                });
            }
        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute z commitment: " << diff.count() << "ms" << std::endl;
        #endif
                */
        todo!("execute_third_round implementation")
    }

    /// Computes the quotient polynomial, then commits to its degree-n split parts.
    fn execute_fourth_round(&self) {
        // TODO
        /*
                queue.flush_queue();
            transcript.apply_fiat_shamir("alpha");
        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        #endif
        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute wire ffts: " << diff.count() << "ms" << std::endl;
        #endif

        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif
        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "copy z: " << diff.count() << "ms" << std::endl;
        #endif
        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif
        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute permutation grand product coeffs: " << diff.count() << "ms" << std::endl;
        #endif
            fr alpha_base = fr::serialize_from_buffer(transcript.get_challenge("alpha").begin());

            // Compute FFT of lagrange polynomial L_1 (needed in random widgets only)
            compute_lagrange_1_fft();

            for (auto& widget : random_widgets) {
        #ifdef DEBUG_TIMING
                std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        #endif
                alpha_base = widget->compute_quotient_contribution(alpha_base, transcript);
        #ifdef DEBUG_TIMING
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cerr << "widget " << i << " quotient compute time: " << diff.count() << "ms" << std::endl;
        #endif
            }

            for (auto& widget : transition_widgets) {
                alpha_base = widget->compute_quotient_contribution(alpha_base, transcript);
            }
        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif

            // The parts of the quotient polynomial t(X) are stored as 4 separate polynomials in
            // the code. However, operations such as dividing by the pseudo vanishing polynomial
            // as well as iFFT (coset) are to be performed on the polynomial t(X) as a whole.
            // We avoid redundant copy of the parts t_1, t_2, t_3, t_4 and instead just tweak the
            // relevant functions to work on quotient polynomial parts.
            std::vector<fr*> quotient_poly_parts;
            quotient_poly_parts.push_back(&key->quotient_polynomial_parts[0][0]);
            quotient_poly_parts.push_back(&key->quotient_polynomial_parts[1][0]);
            quotient_poly_parts.push_back(&key->quotient_polynomial_parts[2][0]);
            quotient_poly_parts.push_back(&key->quotient_polynomial_parts[3][0]);
            barretenberg::polynomial_arithmetic::divide_by_pseudo_vanishing_polynomial(
                quotient_poly_parts, key->small_domain, key->large_domain);

        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "divide by vanishing polynomial: " << diff.count() << "ms" << std::endl;
        #endif
        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif
            polynomial_arithmetic::coset_ifft(quotient_poly_parts, key->large_domain);

            // Manually copy the (n + 1)th coefficient of t_3 for StandardPlonk from t_4.
            // This is because the degree of t_3 for StandardPlonk is n.
            if (settings::program_width == 3) {
                key->quotient_polynomial_parts[2][circuit_size] = key->quotient_polynomial_parts[3][0];
                key->quotient_polynomial_parts[3][0] = 0;
            }

        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "final inverse fourier transforms: " << diff.count() << "ms" << std::endl;
        #endif
        #ifdef DEBUG_TIMING
            start = std::chrono::steady_clock::now();
        #endif

            add_blinding_to_quotient_polynomial_parts();

            compute_quotient_commitments();
        #ifdef DEBUG_TIMING
            end = std::chrono::steady_clock::now();
            diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute quotient commitment: " << diff.count() << "ms" << std::endl;
        #endif
        } // namespace proof_system::plonk
                */
        todo!("implement me")
    }
    fn execute_fifth_round(&self) {
        // TODO
        /*

        template <typename settings> void ProverBase<settings>::execute_fifth_round()
        {
            queue.flush_queue();
            transcript.apply_fiat_shamir("z"); // end of 4th round
        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        #endif
            compute_quotient_evaluation();
        #ifdef DEBUG_TIMING
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::chrono::milliseconds diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cerr << "compute quotient evaluation: " << diff.count() << "ms" << std::endl;
        #endif
                 */
        todo!("implement me")
    }

    fn execute_sixth_round(&self) {
        // TODO
        /*
                queue.flush_queue();
        transcript.apply_fiat_shamir("nu");
        commitment_scheme->batch_open(transcript, queue, key);
        */
        todo!("implement me")
    }

    fn add_polynomial_evaluations_to_transcript(&self) {
        todo!("i don't know what this is")
    }
    fn compute_batch_opening_polynomials(&self) {
        todo!("i don't know what this is")
    }
    /// - Compute wire commitments and add them to the transcript.
    /// - Add public_inputs from w_2_fft to transcript.
    fn compute_wire_commitments(&self) -> Result<()> {
        // Compute wire commitments
        let end: usize = if self.settings.is_plookup() {
            self.settings.program_width() - 1
        } else {
            self.settings.program_width()
        };
        for i in 0..end {
            let wire_tag = format!("w_{}", i + 1);
            let commit_tag = format!("W_{}", i + 1);
            let mut coefficients = self.key.polynomial_store.get(wire_tag)?.get_coefficients();

            // This automatically saves the computed point to the transcript
            let domain_size_flag = if i > 2 {
                self.key.circuit_size
            } else {
                self.key.circuit_size + 1
            };
            self.commitment_scheme.commit(
                &mut coefficients,
                commit_tag,
                Fr::from(domain_size_flag as u64),
                &mut self.queue,
            );
        }

        // add public inputs
        let public_wires_source = self.key.polynomial_store.get("w_2_lagrange".to_string())?;
        let mut public_wires = vec![];
        for i in 0..self.key.num_public_inputs {
            public_wires.push(public_wires_source[i]);
        }
        self.transcript
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
    fn compute_quotient_commitments(&self) {
        for i in 0..self.settings.program_width() {
            let coefficients = self.key.quotient_polynomial_parts[i].get_coefficients();
            let quotient_tag = format!("T_{}", i + 1);
            // Set flag that determines domain size (currently n or n+1) in pippenger (see process_queue()).
            // Note: After blinding, all t_i have size n+1 representation (degree n) except t_4 in Turbo/Ultra.
            let domain_size_flag = if i > 2 {
                self.key.circuit_size
            } else {
                self.key.circuit_size + 1
            };
            // TODO concerned about the fr::from on domain_size_flag. this may not be correct. bberg just uses equivalent of std::mem::transmute
            self.commitment_scheme.commit(
                &mut coefficients,
                quotient_tag,
                Fr::from(domain_size_flag as u64),
                &mut self.queue,
            );
        }
    }
    fn init_quotient_polynomials(&self) {
        todo!("yeehaw")
    }
    fn compute_opening_elements(&self) {
        todo!("yeehaw")
    }
    fn add_plookup_memory_records_to_w_4(&mut self) -> Result<()> {
        // We can only compute memory record values once W_1, W_2, W_3 have been comitted to,
        // due to the dependence on the `eta` challenge.

        let eta = self.transcript.get_field_element("eta");

        // We need the lagrange-base forms of the first 3 wires to compute the plookup memory record
        // value. w4 = w3 * eta^3 + w2 * eta^2 + w1 * eta + read_write_flag;
        // a RAM write. See plookup_auxiliary_widget.hpp for details)
        let w_1 = self
            .key
            .polynomial_store
            .get("w_1_lagrange".to_string())
            .unwrap();
        let w_2 = self
            .key
            .polynomial_store
            .get("w_2_lagrange".to_string())
            .unwrap();
        let w_3 = self
            .key
            .polynomial_store
            .get("w_3_lagrange".to_string())
            .unwrap();
        let mut w_4 = self
            .key
            .polynomial_store
            .get("w_4_lagrange".to_string())
            .unwrap();
        for gate_idx in self.key.memory_read_records {
            w_4[gate_idx] += w_3[gate_idx];
            w_4[gate_idx] *= eta;
            w_4[gate_idx] += w_2[gate_idx];
            w_4[gate_idx] *= eta;
            w_4[gate_idx] += w_1[gate_idx];
            w_4[gate_idx] *= eta;
        }
        for gate_idx in self.key.memory_write_records {
            w_4[gate_idx] += w_3[gate_idx];
            w_4[gate_idx] *= eta;
            w_4[gate_idx] += w_2[gate_idx];
            w_4[gate_idx] *= eta;
            w_4[gate_idx] += w_1[gate_idx];
            w_4[gate_idx] *= eta;
            w_4[gate_idx] += Fr::one();
        }
        self.key
            .polynomial_store
            .put("w_4_lagrange".to_string(), w_4);
        Ok(())
    }

    fn compute_quotient_evaluation(&self) -> Result<()> {
        let zeta = self.transcript.get_field_element("zeta");

        self.commitment_scheme
            .add_opening_evaluations_to_transcript(
                self.transcript.borrow_mut(),
                Some(self.key),
                false,
            );

        let mut t_eval = polynomial_arithmetic::evaluate(
            &[
                self.key.quotient_polynomial_parts[0][0],
                self.key.quotient_polynomial_parts[1][0],
                self.key.quotient_polynomial_parts[2][0],
                self.key.quotient_polynomial_parts[3][0],
            ],
            &zeta,
            4 * self.circuit_size,
        );

        // TODO are these limbs wrong?
        let zeta_pow_n = zeta.pow([self.key.circuit_size as u64]);
        let mut scalar = zeta_pow_n;
        // Adjust the evaluation to consider the (n + 1)th coefficient when needed (note that width 3 is just an avatar for
        // StandardComposer here)
        let num_deg_n_poly = if self.settings.program_width() == 3 {
            self.settings.program_width()
        } else {
            self.settings.program_width() - 1
        };
        for j in 0..num_deg_n_poly {
            t_eval += self.key.quotient_polynomial_parts[j][self.key.circuit_size] * scalar;
            scalar *= zeta_pow_n;
        }

        self.transcript.add_field_element("t", &t_eval);
        Ok(())
    }

    /// Add blinding to the components in such a way that the full quotient would be unchanged if reconstructed
    fn add_blinding_to_quotient_polynomial_parts(&self) {
        // Construct blinded quotient polynomial parts t_i by adding randomness to the unblinded parts t_i' in
        // such a way that the full quotient polynomial t is unchanged upon reconstruction, i.e.
        //
        //        t = t_1' + X^n*t_2' + X^2n*t_3' + X^3n*t_4' = t_1 + X^n*t_2 + X^2n*t_3 + X^3n*t_4
        //
        // Blinding is done as follows, where b_i are random field elements:
        //
        //              t_1 = t_1' +       b_0*X^n
        //              t_2 = t_2' - b_0 + b_1*X^n
        //              t_3 = t_3' - b_1 + b_2*X^n
        //              t_4 = t_4' - b_2
        //
        // For details, please head to: https://hackmd.io/JiyexiqRQJW55TMRrBqp1g.
        for i in 0..self.settings.program_width() - 1 {
            // Note that only program_width-1 random elements are required for full blinding
            let quotient_randomness = Fr::rand(&mut self.rng);

            self.key.quotient_polynomial_parts[i][self.key.circuit_size] += quotient_randomness; // update coefficient of X^n'th term
            self.key.quotient_polynomial_parts[i + 1][0] -= quotient_randomness;
            // update constant coefficient
        }
    }

    /// Compute FFT of lagrange polynomial L_1 needed in random widgets only
    fn compute_lagrange_1_fft(&self) {
        let lagrange_1_fft = Polynomial::new(4 * self.circuit_size + 8);
        polynomial_arithmetic::compute_lagrange_polynomial_fft(
            lagrange_1_fft.get_coefficients(),
            self.key.small_domain,
            self.key.large_domain,
        );
        for i in 0..8 {
            lagrange_1_fft[4 * self.circuit_size + i] = lagrange_1_fft[i];
        }
        self.key
            .polynomial_store
            .put("lagrange_1_fft".to_string(), lagrange_1_fft);
    }

    fn export_proof(&self) -> Proof {
        Proof {
            proof_data: self.transcript.export_transcript(),
        }
    }

    pub(crate) fn construct_proof(&mut self) -> Proof {
        // Execute init round. Randomize witness polynomials.
        self.execute_preamble_round();
        self.queue.process_queue();

        // Compute wire precommitments and sometimes random widget round commitments
        self.execute_first_round();
        self.queue.process_queue();

        // Fiat-Shamir eta + execute random widgets.
        self.execute_second_round();
        self.queue.process_queue();

        // Fiat-Shamir beta & gamma, execute random widgets (Permutation widget is executed here)
        // and fft the witnesses
        self.execute_third_round();
        self.queue.process_queue();

        // Fiat-Shamir alpha, compute & commit to quotient polynomial.
        self.execute_fourth_round();
        self.queue.process_queue();

        self.execute_fifth_round();

        self.execute_sixth_round();
        self.queue.process_queue();

        self.queue.flush_queue();

        return self.export_proof();
    }

    fn get_circuit_size(&self) -> usize {
        todo!("implement me")
    }
    fn flush_queued_work_items(&mut self) {
        self.queue.flush_queue()
    }
    fn get_queued_work_item_info(&self) -> work_queue::WorkItemInfo {
        self.queue.get_queued_work_item_info()
    }
    fn get_scalar_multiplication_data(&self, work_item_number: usize) -> Option<Arc<Vec<Fr>>> {
        self.queue.get_scalar_multiplication_data(work_item_number)
    }
    fn get_scalar_multiplication_size(&self, work_item_number: usize) -> usize {
        self.queue.get_scalar_multiplication_size(work_item_number)
    }
    fn get_ifft_data(&self, work_item_number: usize) -> Option<Arc<Vec<Fr>>> {
        self.queue.get_ifft_data(work_item_number)
    }
    fn get_fft_data(&self, work_item_number: usize) -> Option<Arc<QueuedFftInputs<Fr>>> {
        self.queue.get_fft_data(work_item_number)
    }
    fn put_scalar_multiplication_data(&self, result: G1Affine, work_item_number: usize) {
        self.queue
            .put_scalar_multiplication_data(result, work_item_number);
    }
    fn put_fft_data(&self, result: Vec<Fr>, work_item_number: usize) {
        self.queue.put_fft_data(result, work_item_number);
    }
    fn put_ifft_data(&self, result: Vec<Fr>, work_item_number: usize) {
        self.queue.put_ifft_data(result, work_item_number);
    }
    fn reset(&mut self) {
        let manifest = self.transcript.get_manifest();
        self.transcript = Arc::new(Transcript::<H, Fr, G1Affine>::new(
            Some(manifest),
            self.transcript.num_challenge_bytes,
        ));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_prover() {
        todo!("get it from prover.test.cpp. there is like 300 lines in there.")
    }
}
