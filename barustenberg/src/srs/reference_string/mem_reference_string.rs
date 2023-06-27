use std::rc::Rc;

use ark_bn254::G2Affine;

use crate::ecc::MillerLines;

use super::VerifierReferenceString;

#[derive(Debug, Default)]
pub(crate) struct VerifierMemReferenceString {
    g2_x: G2Affine,
    precomputed_g2_lines: Rc<Vec<MillerLines>>,
}

impl VerifierMemReferenceString {
    pub(crate) fn new(_g2x: &[u8]) -> Self {
        // Add the necessary code to convert g2x bytes into g2::AffineElement
        // and initialize precomputed_g2_lines
        unimplemented!()
    }
}

impl VerifierReferenceString for VerifierMemReferenceString {
    fn get_g2x(&self) -> G2Affine {
        self.g2_x
    }

    fn get_precomputed_g2_lines(&self) -> Rc<Vec<MillerLines>> {
        self.precomputed_g2_lines.clone()
    }
}
