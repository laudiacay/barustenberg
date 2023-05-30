use std::{marker::PhantomData, sync::Arc};

use ark_ec::AffineRepr;

// TODO do this later properly
#[derive(Clone, Default)]
pub struct MillerLines {}

pub trait VerifierReferenceString<G2Affine: AffineRepr> {
    fn get_g2x(&self) -> G2Affine;
    fn get_precomputed_g2_lines(&self) -> Vec<MillerLines>;
}

pub trait ProverReferenceString<G1Affine: AffineRepr> {
    fn get_monomial_points(&mut self) -> Vec<G1Affine>;
    fn get_monomial_size(&self) -> usize;
}
pub trait ReferenceStringFactory<G1Affine: AffineRepr, G2Affine: AffineRepr> {
    fn get_prover_crs(&self, _size: usize) -> Option<Arc<dyn ProverReferenceString<G1Affine>>> {
        todo!()
    }

    fn get_verifier_crs(&self) -> Option<Arc<dyn VerifierReferenceString<G2Affine>>> {
        todo!()
    }
}

#[derive(Clone, Default)]
pub struct BaseReferenceStringFactory<G1Affine: AffineRepr, G2Affine: AffineRepr> {
    phantom: PhantomData<(G1Affine, G2Affine)>,
}

impl<G1Affine: AffineRepr, G2Affine: AffineRepr> ReferenceStringFactory<G1Affine, G2Affine>
    for BaseReferenceStringFactory<G1Affine, G2Affine>
{
}
