use ark_ec::AffineRepr;

use super::{MillerLines, VerifierReferenceString};

pub struct VerifierMemReferenceString<G2Affine: AffineRepr> {
    g2_x: G2Affine,
    precomputed_g2_lines: Box<MillerLines>,
}

impl<G2Affine: AffineRepr> VerifierMemReferenceString<G2Affine> {
    pub fn new(g2x: &[u8]) -> Self {
        // Add the necessary code to convert g2x bytes into g2::AffineElement
        // and initialize precomputed_g2_lines
        unimplemented!()
    }
}

impl<G2Affine: AffineRepr> Drop for VerifierMemReferenceString<G2Affine> {
    fn drop(&mut self) {
        // Implement the destructor logic here if necessary
    }
}

impl<G2Affine: AffineRepr> VerifierReferenceString<G2Affine>
    for VerifierMemReferenceString<G2Affine>
{
    fn get_g2x(&self) -> G2Affine {
        self.g2_x
    }

    fn get_precomputed_g2_lines(&self) -> &MillerLines {
        &self.precomputed_g2_lines
    }
}
