pub(crate) mod file_reference_string;
pub(crate) mod mem_reference_string;
pub(crate) mod pippenger_reference_string;

use std::fmt::Debug;
use std::rc::Rc;

use ark_bn254::{G1Affine, G2Affine};

use crate::ecc::MillerLines;

pub(crate) trait VerifierReferenceString: Debug {
    fn get_g2x(&self) -> G2Affine;
    fn get_precomputed_g2_lines(&self) -> Rc<Vec<MillerLines>>;
}

pub(crate) trait ProverReferenceString: Debug {
    fn get_monomial_points(&mut self) -> Rc<Vec<G1Affine>>;
    fn get_monomial_size(&self) -> usize;
}
pub(crate) trait ReferenceStringFactory: Default {
    type Pro: ProverReferenceString;
    type Ver: VerifierReferenceString;
    fn get_prover_crs(&self, _size: usize) -> Option<Rc<Self::Pro>> {
        todo!()
    }

    fn get_verifier_crs(&self) -> Option<Rc<Self::Ver>> {
        todo!()
    }
}
