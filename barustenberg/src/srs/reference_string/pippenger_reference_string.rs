use crate::barretenberg::pairing::MillerLines;
use std::sync::Arc;

pub struct PippengerReferenceString<'a> {
    pippenger: &'a mut scalar_multiplication::Pippenger,
}

impl<'a> PippengerReferenceString<'a> {
    pub fn new(pippenger: &'a mut scalar_multiplication::Pippenger) -> Self {
        PippengerReferenceString { pippenger }
    }
}

impl<'a> ProverReferenceString for PippengerReferenceString<'a> {
    fn get_monomial_size(&self) -> usize {
        self.pippenger.get_num_points()
    }

    fn get_monomial_points(&mut self) -> &mut g1::AffineElement {
        self.pippenger.get_point_table()
    }
}

pub struct PippengerReferenceStringFactory<'a> {
    pippenger: &'a mut scalar_multiplication::Pippenger,
    g2x: &'a [u8],
}

impl<'a> PippengerReferenceStringFactory<'a> {
    pub fn new(pippenger: &'a mut scalar_multiplication::Pippenger, g2x: &'a [u8]) -> Self {
        PippengerReferenceStringFactory { pippenger, g2x }
    }
}

impl<'a> ReferenceStringFactory for PippengerReferenceStringFactory<'a> {
    fn get_prover_crs(&self, degree: usize) -> Option<Arc<dyn ProverReferenceString>> {
        assert!(degree <= self.pippenger.get_num_points());
        Some(Arc::new(PippengerReferenceString::new(self.pippenger)))
    }

    fn get_verifier_crs(&self) -> Option<Arc<dyn VerifierReferenceString>> {
        Some(Arc::new(VerifierMemReferenceString::new(self.g2x)))
    }
}
