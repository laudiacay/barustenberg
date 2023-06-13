use std::{marker::PhantomData, rc::Rc};

use ark_ec::AffineRepr;

// TODO do this later properly
#[derive(Clone, Default)]
pub(crate) struct MillerLines {}

pub(crate) trait VerifierReferenceString<G2Affine: AffineRepr> {
    fn get_g2x(&self) -> G2Affine;
    fn get_precomputed_g2_lines(&self) -> Rc<Vec<MillerLines>>;
}

pub(crate) trait ProverReferenceString<G1Affine: AffineRepr> {
    fn get_monomial_points(&mut self) -> Rc<Vec<G1Affine>>;
    fn get_monomial_size(&self) -> usize;
}
pub(crate) trait ReferenceStringFactory<G1Affine: AffineRepr, G2Affine: AffineRepr> {
    fn get_prover_crs(&self, _size: usize) -> Option<Rc<dyn ProverReferenceString<G1Affine>>> {
        todo!()
    }

    fn get_verifier_crs(&self) -> Option<Rc<dyn VerifierReferenceString<G2Affine>>> {
        todo!()
    }
}

#[derive(Clone, Default)]
pub(crate) struct BaseReferenceStringFactory<G1Affine: AffineRepr, G2Affine: AffineRepr> {
    phantom: PhantomData<(G1Affine, G2Affine)>,
}

impl<G1Affine: AffineRepr, G2Affine: AffineRepr> ReferenceStringFactory<G1Affine, G2Affine>
    for BaseReferenceStringFactory<G1Affine, G2Affine>
{
}
