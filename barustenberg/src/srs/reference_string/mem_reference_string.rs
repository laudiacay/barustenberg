use ark_bn254::G2Affine;
use ark_serialize::CanonicalDeserialize;

use super::VerifierReferenceString;

#[derive(Debug)]
pub(crate) struct VerifierMemReferenceString {
    g2_x: G2Affine,
}

impl VerifierMemReferenceString {
    pub(crate) fn new(g2x: &[u8]) -> Self {
        let g2_x = match G2Affine::deserialize_uncompressed(g2x) {
            Ok(g2_x) => g2_x,
            Err(_) => panic!("Failed to deserialize g2_x"),
        };

        VerifierMemReferenceString { g2_x }
    }

    pub(crate) fn from_affline(_g2x: G2Affine) -> Self {
        VerifierMemReferenceString { g2_x: _g2x }
    }

    pub(crate) fn default() -> Self {
        VerifierMemReferenceString::from_affline(G2Affine::default())
    }
}

impl VerifierReferenceString for VerifierMemReferenceString {
    fn get_g2x(&self) -> G2Affine {
        self.g2_x
    }
}
