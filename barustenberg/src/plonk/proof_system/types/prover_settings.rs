use std::{collections::HashMap, sync::{Arc, RwLock}};

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use ark_bn254::{Fr, G1Affine};

use crate::{
    plonk::proof_system::{verification_key::VerificationKey, widgets::{random_widgets::permutation_widget::VerifierPermutationWidget, transition_widgets::{arithmetic_widget::VerifierArithmeticWidget, transition_widget::GenericVerifierWidgetBase}}},
    transcript::{BarretenHasher, Keccak256, PedersenBlake3s, PlookupPedersenBlake3s, Transcript},
};

// TODO bevy_reflect? or what
// or inline everything!
pub(crate) trait Settings: Sized {
    type Hasher: BarretenHasher;
    type Field: Field + FftField;
    type Group: AffineRepr;

    #[inline]
    fn requires_shifted_wire(wire_shift_settings: u64, wire_index: u64) -> bool {
        ((wire_shift_settings >> wire_index) & 1u64) == 1u64
    }
    fn num_challenge_bytes(&self) -> usize;
    fn program_width(&self) -> usize;
    fn num_shifted_wire_evaluations(&self) -> usize;
    fn wire_shift_settings(&self) -> u64;
    fn permutation_shift(&self) -> u32;
    fn permutation_mask(&self) -> u32;
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize;
    fn compute_quotient_evaluation_contribution(
        verification_key: Arc<RwLock<VerificationKey<Self::Field>>>,
        alpha_base: &Self::Field,
        transcript: &Transcript<Self::Hasher>,
        quotient_numerator_eval: &mut Self::Field,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field
    where
        Self: Sized;
    fn append_scalar_multiplication_inputs(
        verification_key: Arc<RwLock<VerificationKey<Self::Field>>>,
        alpha_base: &Self::Field,
        transcript: &Transcript<Self::Hasher>,
        scalars: &mut HashMap<String, Self::Field>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field
    where
        Self: Sized;
    fn is_plookup(&self) -> bool;
    fn hasher(&self) -> &Self::Hasher;
}

#[derive(Default, Debug, Clone, Copy)]
pub struct StandardSettings<H: BarretenHasher> {
    hasher: H,
}

impl<H: BarretenHasher> StandardSettings<H> {
    pub(crate) fn new(h: H) -> Self {
        Self { hasher: h }
    }
}

impl<H: BarretenHasher> Settings for StandardSettings<H> {
    type Hasher = H;
    type Field = Fr;
    type Group = G1Affine;
    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        3
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        1
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b0100
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &H {
        &self.hasher
    }

    #[inline]
    fn requires_shifted_wire(wire_shift_settings: u64, wire_index: u64) -> bool {
        ((wire_shift_settings >> wire_index) & 1u64) == 1u64
    }

    #[inline]
    fn compute_quotient_evaluation_contribution(
        verification_key: Arc<RwLock<VerificationKey<Fr>>>,
        alpha_base: &Fr,
        transcript: &Transcript<H>,
        quotient_numerator_eval: &mut Fr,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Fr {
        let updated_alpha_base = VerifierPermutationWidget::<H, Fr, G1Affine, 4>::compute_quotient_evaluation_contribution(
            &verification_key.read().unwrap(),
            *alpha_base,
            transcript,
            quotient_numerator_eval, 
        None);

        return VerifierArithmeticWidget::<H, Fr, G1Affine>::compute_quotient_evaluation_contribution(
            verification_key, updated_alpha_base, transcript, quotient_numerator_eval, rng);
    }

    #[inline]
    fn append_scalar_multiplication_inputs(
        verification_key: Arc<RwLock<VerificationKey<Fr>>>,
        alpha_base: &Fr,
        transcript: &Transcript<H>,
        scalars: &mut HashMap<String, Fr>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Fr {

        let updated_alpha = VerifierPermutationWidget::<H, Fr, G1Affine, 4>::append_scalar_multiplication_inputs(*alpha_base, transcript);

        return VerifierArithmeticWidget::<H, Fr, G1Affine>::append_scalar_multiplication_inputs(verification_key, updated_alpha, transcript, scalars, rng);
    }
}

pub(crate) struct TurboSettings {}

impl TurboSettings {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Settings for TurboSettings {
    type Hasher = PedersenBlake3s;
    type Field = Fr;
    type Group = G1Affine;

    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &PedersenBlake3s {
        &PedersenBlake3s {}
    }
    #[inline]
    fn compute_quotient_evaluation_contribution(
        _verification_key: Arc<RwLock<VerificationKey<Fr>>>,
        _alpha_base: &Fr,
        _transcript: &Transcript<PedersenBlake3s>,
        _quotient_numerator_eval: &mut Fr,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Fr {
        unimplemented!();
        /*
                auto updated_alpha_base = PermutationWidget::compute_quotient_evaluation_contribution(
            key, alpha_base, transcript, quotient_numerator_eval, idpolys);

        updated_alpha_base = TurboArithmeticWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);
        updated_alpha_base = TurboFixedBaseWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);
        updated_alpha_base = TurboRangeWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);
        updated_alpha_base = TurboLogicWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);

        return updated_alpha_base;
         */
    }
    fn append_scalar_multiplication_inputs(
        verification_key: Arc<RwLock<VerificationKey<Self::Field>>>,
        alpha_base: &Self::Field,
        transcript: &Transcript<Self::Hasher>,
        scalars: &mut HashMap<String, Self::Field>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field {
        unimplemented!("todo");
    }
}

pub(crate) struct UltraSettings {}

impl Settings for UltraSettings {
    type Hasher = PlookupPedersenBlake3s;
    type Field = Fr;
    type Group = G1Affine;

    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &PlookupPedersenBlake3s {
        &PlookupPedersenBlake3s {}
    }

    #[inline]
    fn compute_quotient_evaluation_contribution(
        _verification_key: Arc<RwLock<VerificationKey<Fr>>>,
        _alpha_base: &Fr,
        _transcript: &Transcript<Self::Hasher>,
        _quotient_numerator_eval: &mut Fr,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Fr {
        /*
                auto updated_alpha_base = PermutationWidget::compute_quotient_evaluation_contribution(
            key, alpha_base, transcript, quotient_numerator_eval, idpolys);
        updated_alpha_base = PlookupWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);
        updated_alpha_base = PlookupArithmeticWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);
        updated_alpha_base = GenPermSortWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);
        updated_alpha_base = EllipticWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);
        updated_alpha_base = PlookupAuxiliaryWidget::compute_quotient_evaluation_contribution(
            key, updated_alpha_base, transcript, quotient_numerator_eval);

        return updated_alpha_base;
         */
        unimplemented!()
    }

    fn append_scalar_multiplication_inputs(
        verification_key: Arc<RwLock<VerificationKey<Self::Field>>>,
        alpha_base: &Self::Field,
        transcript: &Transcript<Self::Hasher>,
        scalars: &mut HashMap<String, Self::Field>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field {
        unimplemented!("todo");
    }
}

pub(crate) struct UltraToStandardSettings {}

impl Settings for UltraToStandardSettings {
    type Hasher = PedersenBlake3s;
    type Field = Fr;
    type Group = G1Affine;

    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &PedersenBlake3s {
        &PedersenBlake3s {}
    }
    #[inline]
    fn compute_quotient_evaluation_contribution(
        _verification_key: Arc<RwLock<VerificationKey<Fr>>>,
        _alpha_base: &Fr,
        _transcript: &Transcript<PedersenBlake3s>,
        _quotient_numerator_eval: &mut Fr,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Fr {
        // UltraSettings::compute_quotient_evaluation_contribution(verification_key, alpha_base, transcript, quotient_numerator_eval)
        todo!()
    }

    fn append_scalar_multiplication_inputs(
        verification_key: Arc<RwLock<VerificationKey<Self::Field>>>,
        alpha_base: &Self::Field,
        transcript: &Transcript<Self::Hasher>,
        scalars: &mut HashMap<String, Self::Field>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field{
        unimplemented!("todo");
    }
}

pub(crate) struct UltraWithKeccakSettings {}

impl Settings for UltraWithKeccakSettings {
    type Hasher = Keccak256;
    type Field = Fr;
    type Group = G1Affine;

    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        32
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &Keccak256 {
        &Keccak256 {}
    }
    #[inline]
    fn compute_quotient_evaluation_contribution(
        _verification_key: Arc<RwLock<VerificationKey<Fr>>>,
        _alpha_base: &Fr,
        _transcript: &Transcript<Self::Hasher>,
        _quotient_numerator_eval: &mut Fr,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Fr {
        //UltraSettings::compute_quotient_evaluation_contribution(verification_key, alpha_base, transcript, quotient_numerator_eval)
        todo!()
    }
    fn append_scalar_multiplication_inputs(
        verification_key: Arc<RwLock<VerificationKey<Self::Field>>>,
        alpha_base: &Self::Field,
        transcript: &Transcript<Self::Hasher>,
        scalars: &mut HashMap<String, Self::Field>,
        rng: &mut Box<dyn rand::RngCore>,
    ) -> Self::Field {
        unimplemented!("todo");
    }
}
