pub(crate) mod file_reference_string;
pub(crate) mod mem_reference_string;
pub(crate) mod pippenger_reference_string;

use std::{marker::PhantomData, rc::Rc};

use ark_ec::Group;

use crate::ecc::MillerLines;

pub(crate) trait VerifierReferenceString<G2Affine: Group> {
    fn get_g2x(&self) -> G2Affine;
    fn get_precomputed_g2_lines(&self) -> Rc<Vec<MillerLines>>;
}

pub(crate) trait ProverReferenceString<G: Group> {
    fn get_monomial_points(&mut self) -> Rc<Vec<G>>;
    fn get_monomial_size(&self) -> usize;
}
pub(crate) trait ReferenceStringFactory<G: Group, G2Affine: Group> {
    fn get_prover_crs(&self, _size: usize) -> Option<Rc<dyn ProverReferenceString<G>>> {
        todo!()
    }

    fn get_verifier_crs(&self) -> Option<Rc<dyn VerifierReferenceString<G2Affine>>> {
        todo!()
    }
}

#[derive(Clone, Default)]
pub(crate) struct BaseReferenceStringFactory<G: Group, G2Affine: Group> {
    phantom: PhantomData<(G, G2Affine)>,
}

impl<G: Group, G2Affine: Group> ReferenceStringFactory<G, G2Affine>
    for BaseReferenceStringFactory<G, G2Affine>
{
}
