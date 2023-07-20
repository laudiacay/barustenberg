use std::sync::{Arc, RwLock};

use ark_bn254::G1Affine;

use anyhow::Result;

use crate::{
    ecc::curves::bn254_scalar_multiplication::Pippenger,
    srs::reference_string::{ProverReferenceString, ReferenceStringFactory},
};

use super::mem_reference_string::VerifierMemReferenceString;

#[derive(Debug)]
pub(crate) struct PippengerReferenceString {
    pippenger: Arc<Pippenger>,
}

impl PippengerReferenceString {
    pub(crate) fn new(pippenger: Arc<Pippenger>) -> Self {
        PippengerReferenceString { pippenger }
    }
}

impl ProverReferenceString for PippengerReferenceString {
    // TODO
    fn get_monomial_size(&self) -> usize {
        todo!()
    }

    fn get_monomial_points(&self) -> Arc<Vec<G1Affine>> {
        // will we mutate self here?
        todo!()
    }
}

#[derive(Debug, Default)]
pub(crate) struct PippengerReferenceStringFactory<'a> {
    pippenger: Arc<Pippenger>,
    g2x: &'a [u8],
}

impl<'a> PippengerReferenceStringFactory<'a> {
    pub(crate) fn new(pippenger: Arc<Pippenger>, g2x: &'a [u8]) -> Self {
        PippengerReferenceStringFactory { pippenger, g2x }
    }
}

impl<'a> ReferenceStringFactory for PippengerReferenceStringFactory<'a> {
    type Pro = PippengerReferenceString;
    type Ver = VerifierMemReferenceString;

    fn get_prover_crs(&self, degree: usize) -> Result<Option<Arc<RwLock<Self::Pro>>>> {
        assert!(degree <= self.pippenger.get_num_points());
        Ok(Some(Arc::new(RwLock::new(PippengerReferenceString::new(
            self.pippenger.clone(),
        )))))
    }
    fn get_verifier_crs(&self) -> Result<Option<Arc<RwLock<Self::Ver>>>> {
        Ok(Some(Arc::new(RwLock::new(
            VerifierMemReferenceString::new(self.g2x),
        ))))
    }
}
