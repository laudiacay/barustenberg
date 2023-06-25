pub(crate) mod file_reference_string;
pub(crate) mod mem_reference_string;
pub(crate) mod pippenger_reference_string;

use std::rc::Rc;

use ark_bn254::{G1Affine, G2Affine};

use crate::ecc::MillerLines;

pub(crate) trait VerifierReferenceString {
    fn get_g2x(&self) -> G2Affine;
    fn get_precomputed_g2_lines(&self) -> Rc<Vec<MillerLines>>;
}

pub(crate) trait ProverReferenceString {
    fn get_monomial_points(&mut self) -> Rc<Vec<G1Affine>>;
    fn get_monomial_size(&self) -> usize;
}
pub(crate) trait ReferenceStringFactory {
    fn get_prover_crs(&self, _size: usize) -> Option<Rc<dyn ProverReferenceString>> {
        todo!()
    }

    fn get_verifier_crs(&self) -> Option<Rc<dyn VerifierReferenceString>> {
        todo!()
    }
}

#[derive(Clone, Default)]
pub(crate) struct BaseReferenceStringFactory {}

impl ReferenceStringFactory for BaseReferenceStringFactory {}
