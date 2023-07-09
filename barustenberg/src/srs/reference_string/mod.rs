pub(crate) mod file_reference_string;
pub(crate) mod mem_reference_string;
pub(crate) mod pippenger_reference_string;

use std::rc::Rc;
use std::{cell::RefCell, fmt::Debug};

use ark_bn254::{G1Affine, G2Affine};

use crate::ecc::MillerLines;

pub(crate) trait VerifierReferenceString: Debug {
    fn get_g2x(&self) -> G2Affine;
    fn get_precomputed_g2_lines(&self) -> Rc<Vec<MillerLines>>;
}

pub(crate) trait ProverReferenceString: Debug {
    // cpp definition for this is non-const but all implementations are const,
    // unclear to me that we need a mut ref to self
    fn get_monomial_points(&self) -> Rc<Vec<G1Affine>>;
    fn get_monomial_size(&self) -> usize;
}
pub(crate) trait ReferenceStringFactory: Default {
    type Pro: ProverReferenceString + 'static;
    type Ver: VerifierReferenceString + 'static;
    fn get_prover_crs(&self, _size: usize) -> Option<Rc<RefCell<Self::Pro>>> {
        todo!()
    }

    fn get_verifier_crs(&self) -> Option<Rc<RefCell<Self::Ver>>> {
        todo!()
    }
}
