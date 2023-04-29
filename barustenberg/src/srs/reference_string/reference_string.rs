use std::sync::Arc;

// TODO do this later properly
pub struct MillerLines {}
pub mod g2 {
    pub struct AffineElement {}
}
pub mod g1 {
    pub struct AffineElement {}
}

pub trait VerifierReferenceString {
    fn get_g2x(&self) -> g2::AffineElement;
    fn get_precomputed_g2_lines(&self) -> &MillerLines;
}

pub trait ProverReferenceString {
    fn get_monomial_points(&mut self) -> &mut g1::AffineElement;
    fn get_monomial_size(&self) -> usize;
}
pub trait ReferenceStringFactory {
    fn get_prover_crs(&self, _size: usize) -> Option<Arc<dyn ProverReferenceString>> {
        None
    }

    fn get_verifier_crs(&self) -> Option<Arc<dyn VerifierReferenceString>> {
        None
    }
}
