use std::{marker::PhantomData, rc::Rc, sync::Arc};

use ark_ec::Group;

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

pub(crate) struct PippengerReferenceString<G: Group> {
    pippenger: Arc<Pippenger>,
    phantom: PhantomData<G>,
}

impl<G: Group> PippengerReferenceString<G> {
    pub(crate) fn new(pippenger: Arc<Pippenger>) -> Self {
        PippengerReferenceString {
            pippenger,
            phantom: PhantomData,
        }
    }
}

impl<G: Group> ProverReferenceString<G> for PippengerReferenceString<G> {
    // TODO
    fn get_monomial_size(&self) -> usize {
        todo!()
    }

    fn get_monomial_points(&mut self) -> Rc<Vec<G>> {
        todo!()
    }
}

pub(crate) struct PippengerReferenceStringFactory<'a, G: Group, G2Affine: Group> {
    pippenger: Arc<Pippenger>,
    g2x: &'a [u8],
    phantom: PhantomData<(G, G2Affine)>,
}

impl<'a, G: Group, G2Affine: Group> PippengerReferenceStringFactory<'a, G, G2Affine> {
    pub(crate) fn new(pippenger: Arc<Pippenger>, g2x: &'a [u8]) -> Self {
        PippengerReferenceStringFactory {
            pippenger,
            g2x,
            phantom: PhantomData,
        }
    }
}

impl<'a, G: Group, G2Affine: Group> ReferenceStringFactory<G, G2Affine>
    for PippengerReferenceStringFactory<'a, G, G2Affine>
{
    fn get_prover_crs(&self, degree: usize) -> Option<Rc<dyn ProverReferenceString<G>>> {
        assert!(degree <= self.pippenger.get_num_points());
        Some(Rc::new(PippengerReferenceString::new(
            self.pippenger.clone(),
        )))
    }
    fn get_verifier_crs(&self) -> Option<Rc<dyn VerifierReferenceString<G2Affine>>> {
        Some(Rc::new(VerifierMemReferenceString::new(self.g2x)))
    }
}
