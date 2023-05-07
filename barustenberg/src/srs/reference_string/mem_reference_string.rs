use super::{MillerLines, VerifierReferenceString};

use ark_bn254::g2::G2Affine;

pub struct VerifierMemReferenceString {
    g2_x: G2Affine,
    precomputed_g2_lines: Box<MillerLines>,
}

impl VerifierMemReferenceString {
    pub fn new(g2x: &[u8]) -> Self {
        // Add the necessary code to convert g2x bytes into g2::AffineElement
        // and initialize precomputed_g2_lines
        unimplemented!()
    }
}

impl Drop for VerifierMemReferenceString {
    fn drop(&mut self) {
        // Implement the destructor logic here if necessary
    }
}

impl VerifierReferenceString for VerifierMemReferenceString {
    fn get_g2x(&self) -> G2Affine {
        self.g2_x
    }

    fn get_precomputed_g2_lines(&self) -> &MillerLines {
        &self.precomputed_g2_lines
    }
}
