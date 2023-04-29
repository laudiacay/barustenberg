use barretenberg::common::mem;
use barretenberg::ecc::curves::bn254::{g1, g2};
use std::sync::Arc;

pub mod barretenberg {
    pub mod pairing {
        pub struct MillerLines;
    }
}

pub mod proof_system {
    use super::barretenberg::pairing::MillerLines;
    use super::{g1, g2};
    use std::sync::Arc;

    pub trait VerifierReferenceString {
        fn get_g2x(&self) -> g2::AffineElement;
        fn get_precomputed_g2_lines(&self) -> &MillerLines;
    }

    pub trait ProverReferenceString {
        fn get_monomial_points(&mut self) -> &mut g1::AffineElement;
        fn get_monomial_size(&self) -> usize;
    }

    pub struct ReferenceStringFactory {
        // You can add fields here if needed
    }

    impl ReferenceStringFactory {
        pub fn new() -> Self {
            ReferenceStringFactory {}
        }

        pub fn get_prover_crs(&self, _size: usize) -> Option<Arc<dyn ProverReferenceString>> {
            None
        }

        pub fn get_verifier_crs(&self) -> Option<Arc<dyn VerifierReferenceString>> {
            None
        }
    }
}
