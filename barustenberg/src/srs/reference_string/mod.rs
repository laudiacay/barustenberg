pub(crate) mod file_reference_string;
pub(crate) mod mem_reference_string;
pub(crate) mod pippenger_reference_string;

use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use ark_bn254::{G1Affine, G2Affine};

use anyhow::Result;

pub(crate) trait VerifierReferenceString: Debug + Send + Sync {
    fn get_g2x(&self) -> G2Affine;
}

pub(crate) trait ProverReferenceString: Debug + Send + Sync {
    // cpp definition for this is non-const but all implementations are const,
    // unclear to me that we need a mut ref to self
    fn get_monomial_points(&self) -> Arc<Vec<G1Affine>>;
    fn get_monomial_size(&self) -> usize;
}
pub(crate) trait ReferenceStringFactory: Default {
    type Pro: ProverReferenceString + 'static;
    type Ver: VerifierReferenceString + 'static;
    fn get_prover_crs(&self, _size: usize) -> Result<Option<Arc<RwLock<Self::Pro>>>>;

    fn get_verifier_crs(&self) -> Result<Option<Arc<RwLock<Self::Ver>>>>;
}
