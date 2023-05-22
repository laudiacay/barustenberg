use std::{marker::PhantomData, sync::Arc};

use ark_ec::AffineRepr;

use crate::srs::reference_string::{
    ProverReferenceString, ReferenceStringFactory, VerifierReferenceString,
};

use super::mem_reference_string::VerifierMemReferenceString;

struct Pippenger {}

impl Pippenger {
    pub fn get_num_points(&self) -> usize {
        todo!()
    }
}

pub struct PippengerReferenceString<'a, G1Affine: AffineRepr> {
    pippenger: &'a mut Pippenger,
    phantom: PhantomData<G1Affine>,
}

impl<'a, G1Affine: AffineRepr> PippengerReferenceString<'a, G1Affine> {
    pub fn new(pippenger: &'a mut Pippenger) -> Self {
        PippengerReferenceString {
            pippenger,
            phantom: PhantomData,
        }
    }
}

impl<'a, G1Affine: AffineRepr> ProverReferenceString<G1Affine>
    for PippengerReferenceString<'a, G1Affine>
{
    // TODO
    fn get_monomial_size(&self) -> usize {
        todo!()
    }

    fn get_monomial_points(&mut self) -> &mut G1Affine {
        todo!()
    }
}

pub struct PippengerReferenceStringFactory<'a, G1Affine: AffineRepr, G2Affine: AffineRepr> {
    pippenger: &'a mut Pippenger,
    g2x: &'a [u8],
    phantom: PhantomData<(G1Affine, G2Affine)>,
}

impl<'a, G1Affine: AffineRepr, G2Affine: AffineRepr>
    PippengerReferenceStringFactory<'a, G1Affine, G2Affine>
{
    pub fn new(pippenger: &'a mut Pippenger, g2x: &'a [u8]) -> Self {
        PippengerReferenceStringFactory {
            pippenger,
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
        Some(Arc::new(PippengerReferenceString::new(self.pippenger)))
    }
    fn get_verifier_crs(&self) -> Option<Arc<dyn VerifierReferenceString<G2Affine>>> {
        Some(Arc::new(VerifierMemReferenceString::new(self.g2x)))
    }
}
