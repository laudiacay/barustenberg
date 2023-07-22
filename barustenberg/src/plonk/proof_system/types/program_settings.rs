// use ark_ff::FieldExt;

// use crate::{ecc::Group, transcript::{HasherType, Transcript, self}, plonk::proof_system::verification_key::VerificationKey};
// use std::collections::HashMap;

// use super::prover_settings::SettingsBase;

// pub trait VerifierSettings<Fr: FieldExt, G1: Group, H: HasherType>: SettingsBase<H> {
//     type ArithmeticWidget; // Define ArithmeticWidget trait
//     type PermutationWidget; // Define PermutationWidget trait
//     type Transcript;

//     const NUM_CHALLENGE_BYTES: usize;
//     const HASH_TYPE: HasherType;
//     const ID_POLYS: bool;

//     fn append_scalar_multiplication_inputs(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         scalars: &mut HashMap<String, Self::Fr>,
//     ) -> Self::Fr;

//     fn compute_quotient_evaluation_contribution(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         quotient_numerator_eval: &mut Self::Fr,
//     ) -> Self::Fr;
// }

// pub struct StandardVerifierSettings;

// impl<Fr: ark_bn254::fr::Fr, G1: ark_bn254::g1::G, H: transcript::HasherType::PedersenBlake3s> VerifierSettings for StandardVerifierSettings {
//     type Transcript = transcript::Transcript;
//     type ArithmeticWidget; // Define ArithmeticWidget trait
//     type PermutationWidget; // Define PermutationWidget trait

//     const HASH_TYPE: transcript::HashType = transcript::HasherType::PedersenBlake3s;
//     const NUM_CHALLENGE_BYTES: usize = 16;
//     const ID_POLYS: bool = false;

//     fn append_scalar_multiplication_inputs(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         scalars: &mut HashMap<String, Self::Fr>,
//     ) -> Self::Fr;

//     fn compute_quotient_evaluation_contribution(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         quotient_numerator_eval: &mut Self::Fr,
//     ) -> Self::Fr;
// }
// pub trait StandardVerifierSettings: StandardSettings {
//     type Fr = ark_bn254::fr::Fr;
//     type G1 = ark_bn254::g1::G1;
//     type Transcript = transcript::StandardTranscript;
//     type ArithmeticWidget; // Define ArithmeticWidget trait
//     type PermutationWidget; // Define PermutationWidget trait

//     const HASH_TYPE: transcript::HashType = transcript::HashType::PedersenBlake3s;
//     const NUM_CHALLENGE_BYTES: usize = 16;
//     const ID_POLYS: bool = false;

//     fn append_scalar_multiplication_inputs(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         scalars: &mut HashMap<String, Self::Fr>,
//     ) -> Self::Fr;

//     fn compute_quotient_evaluation_contribution(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         quotient_numerator_eval: &mut Self::Fr,
//     ) -> Self::Fr;
// }

// pub trait TurboVerifierSettings: TurboSettings {
//     type Fr = fr;
//     type G1 = g1;
//     type Transcript = transcript::StandardTranscript;
//     // Define other widget types here

//     const NUM_CHALLENGE_BYTES: usize = 16;
//     const HASH_TYPE: transcript::HashType = transcript::HashType::PedersenBlake3s;
//     const ID_POLYS: bool = false;

//     fn append_scalar_multiplication_inputs(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         scalars: &mut HashMap<String, Self::Fr>,
//     ) -> Self::Fr;

//     fn compute_quotient_evaluation_contribution(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         quotient_numerator_eval: &mut Self::Fr,
//     ) -> Self::Fr;
// }

// pub trait UltraVerifierSettings: UltraSettings {
//     type Fr = fr;
//     type G1 = g1;
//     type Transcript = transcript::StandardTranscript;
//     // Define other widget types here

//     const NUM_CHALLENGE_BYTES: usize = 16;
//     const HASH_TYPE: transcript::HashType = transcript::HashType::PlookupPedersenBlake3s;
//     const ID_POLYS: bool = true;

//     fn append_scalar_multiplication_inputs(
//         key: &mut VerificationKey,
//         alpha_base: Self::Fr,
//         transcript: &Self::Transcript,
//         scalars: &mut HashMap<String, Self::Fr>,
//     ) -> Self::Fr;

//     fn compute_quotient_evaluation_con
