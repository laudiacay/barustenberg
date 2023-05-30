use std::{marker::PhantomData, sync::Arc};

use ark_ec::AffineRepr;

use crate::srs::reference_string::{
    ProverReferenceString, ReferenceStringFactory, VerifierReferenceString,
};

use super::mem_reference_string::VerifierMemReferenceString;

#[derive(Clone, Default)]
pub(crate) struct Pippenger {}

impl Pippenger {
    pub(crate) fn get_num_points(&self) -> usize {
        todo!()
    }
}

pub(crate) struct PippengerReferenceString<G1Affine: AffineRepr> {
    pippenger: Arc<Pippenger>,
    phantom: PhantomData<G1Affine>,
}

impl<G1Affine: AffineRepr> PippengerReferenceString<G1Affine> {
    pub(crate) fn new(pippenger: Arc<Pippenger>) -> Self {
        PippengerReferenceString {
            pippenger,
            phantom: PhantomData,
        }
    }
}

impl<G1Affine: AffineRepr> ProverReferenceString<G1Affine> for PippengerReferenceString<G1Affine> {
    // TODO
    fn get_monomial_size(&self) -> usize {
        todo!()
    }

    fn get_monomial_points(&mut self) -> &Vec<G1Affine> {
        todo!()
    }
}

pub(crate) struct PippengerReferenceStringFactory<'a, G1Affine: AffineRepr, G2Affine: AffineRepr> {
    pippenger: Arc<Pippenger>,
    g2x: &'a [u8],
    phantom: PhantomData<(G1Affine, G2Affine)>,
}

impl<'a, G1Affine: AffineRepr, G2Affine: AffineRepr>
    PippengerReferenceStringFactory<'a, G1Affine, G2Affine>
{
    pub(crate) fn new(pippenger: Arc<Pippenger>, g2x: &'a [u8]) -> Self {
        PippengerReferenceStringFactory {
            pippenger: pippenger.clone(),
            g2x,
            phantom: PhantomData,
        }
    }
}

impl<'a, G1Affine: AffineRepr, G2Affine: AffineRepr> ReferenceStringFactory<G1Affine, G2Affine>
    for PippengerReferenceStringFactory<'a, G1Affine, G2Affine>
{
    fn get_prover_crs(&self, degree: usize) -> Option<Arc<dyn ProverReferenceString<G1Affine>>> {
        assert!(degree <= self.pippenger.get_num_points());
        Some(Arc::new(PippengerReferenceString::new(
            self.pippenger.clone(),
        )))
    }
    fn get_verifier_crs(&self) -> Option<Arc<dyn VerifierReferenceString<G2Affine>>> {
        Some(Arc::new(VerifierMemReferenceString::new(self.g2x)))
    }
}
