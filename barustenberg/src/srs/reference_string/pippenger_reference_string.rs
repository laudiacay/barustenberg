use std::sync::Arc;

use crate::{
    ecc::curves::bn254::scalar_multiplication::pippenger::Pippenger,
    srs::reference_string::{
        ProverReferenceString, ReferenceStringFactory, VerifierReferenceString,
    },
};

use super::mem_reference_string::VerifierMemReferenceString;

pub struct PippengerReferenceString<'a> {
    pippenger: &'a mut Pippenger,
}

impl<'a> PippengerReferenceString<'a> {
    pub fn new(pippenger: &'a mut Pippenger) -> Self {
        PippengerReferenceString { pippenger }
    }
}

impl<'a> ProverReferenceString for PippengerReferenceString<'a> {
    // TODO
    fn get_monomial_size(&self) -> usize {
        todo!()
    }

    fn get_monomial_points(&mut self) -> &mut Affine<G1> {
        todo!()
    }
}

pub struct PippengerReferenceStringFactory<'a> {
    pippenger: &'a mut Pippenger,
    g2x: &'a [u8],
}

impl<'a> PippengerReferenceStringFactory<'a> {
    pub fn new(pippenger: &'a mut Pippenger, g2x: &'a [u8]) -> Self {
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
