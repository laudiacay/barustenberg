use std::sync::Arc;

// TODO do this later properly
pub struct MillerLines {}

pub trait VerifierReferenceString {
    fn get_g2x(&self) -> G2Affine;
    fn get_precomputed_g2_lines(&self) -> &MillerLines;
}

pub trait ProverReferenceString {
    fn get_monomial_points(&mut self) -> &mut G1Affine;
    fn get_monomial_size(&self) -> usize;
}
pub trait ReferenceStringFactory {
    fn get_prover_crs(&self, _size: usize) -> Option<Arc<dyn ProverReferenceString>> {
        todo!()
    }

    fn get_verifier_crs(&self) -> Option<Arc<dyn VerifierReferenceString>> {
        todo!()
    }
}
