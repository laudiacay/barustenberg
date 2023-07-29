use std::{cell::RefCell, rc::Rc};

use ark_bn254::{G1Affine, G2Affine};

use crate::{ecc::curves::bn254_scalar_multiplication::Pippenger, srs::io::read_transcript_g2};

use super::{ProverReferenceString, ReferenceStringFactory, VerifierReferenceString};

use anyhow::Result;

#[derive(Debug, Default)]
pub(crate) struct VerifierFileReferenceString {
    g2_x: G2Affine,
}

impl VerifierFileReferenceString {
    pub(crate) fn new(path: &str) -> Result<Self> {
        let mut g2_x = G2Affine::default();
        read_transcript_g2(&mut g2_x, path)?;

        Ok(Self { g2_x })
    }
}

impl VerifierReferenceString for VerifierFileReferenceString {
    fn get_g2x(&self) -> G2Affine {
        self.g2_x.clone()
    }
}

#[derive(Debug, Default)]
pub(crate) struct FileReferenceString {
    num_points: usize,
    pippenger: Pippenger,
}

impl FileReferenceString {
    pub(crate) fn new(num_points: usize, path: &str) -> Self {
        // Implementation depends on your project.
        let pippenger = Pippenger::from_path(path, num_points);
        Self {
            num_points,
            pippenger,
        }
    }

    pub(crate) fn read_from_path(_path: &str) -> Result<Self, std::io::Error> {
        // Implementation depends on your project.
        todo!("FileReferenceString::read_from_path")
    }
}

impl ProverReferenceString for FileReferenceString {
    fn get_monomial_points(&self) -> Rc<Vec<G1Affine>> {
        // Implementation depends on your project.
        todo!()
    }

    fn get_monomial_size(&self) -> usize {
        self.num_points
    }
}

#[derive(Debug, Default)]
pub(crate) struct FileReferenceStringFactory {
    path: String,
}

impl FileReferenceStringFactory {
    pub(crate) fn new(path: String) -> Self {
        Self { path }
    }
}
impl ReferenceStringFactory for FileReferenceStringFactory {
    type Pro = FileReferenceString;
    type Ver = VerifierFileReferenceString;
    fn get_prover_crs(&self, degree: usize) -> Option<Rc<RefCell<Self::Pro>>> {
        Some(Rc::new(RefCell::new(FileReferenceString::new(
            degree, &self.path,
        ))))
    }

    fn get_verifier_crs(&self) -> Result<Option<Rc<RefCell<Self::Ver>>>> {
        Ok(Some(Rc::new(RefCell::new(
            VerifierFileReferenceString::new(&self.path)?,
        ))))
    }
}

#[derive(Debug, Default)]
pub(crate) struct DynamicFileReferenceStringFactory {
    path: String,
    degree: RefCell<usize>,
    prover_crs: Rc<RefCell<FileReferenceString>>,
    verifier_crs: Rc<RefCell<VerifierFileReferenceString>>,
}

impl DynamicFileReferenceStringFactory {
    pub(crate) fn new(path: String, initial_degree: usize) -> Result<Self> {
        let verifier_crs = Rc::new(RefCell::new(VerifierFileReferenceString::new(&path)?));
        let prover_crs = Rc::new(RefCell::new(FileReferenceString::new(
            initial_degree,
            &path,
        )));
        Ok(Self {
            path,
            degree: RefCell::new(initial_degree),
            prover_crs,
            verifier_crs,
        })
    }
}

impl ReferenceStringFactory for DynamicFileReferenceStringFactory {
    type Pro = FileReferenceString;
    type Ver = VerifierFileReferenceString;
    fn get_prover_crs(&self, degree: usize) -> Option<Rc<RefCell<Self::Pro>>> {
        if degree != *self.degree.borrow() {
            *self.prover_crs.borrow_mut() = FileReferenceString::new(degree, &self.path);
            *self.degree.borrow_mut() = degree;
        }
        Some(self.prover_crs.clone())
    }

    fn get_verifier_crs(&self) -> Result<Option<Rc<RefCell<Self::Ver>>>> {
        Ok(Some(self.verifier_crs.clone()))
    }
}
