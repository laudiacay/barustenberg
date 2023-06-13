use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use ark_ec::AffineRepr;

use super::{
    pippenger_reference_string::Pippenger, MillerLines, ProverReferenceString,
    ReferenceStringFactory, VerifierReferenceString,
};

pub(crate) struct VerifierFileReferenceString<G2Affine: AffineRepr> {
    g2_x: G2Affine,
    precomputed_g2_lines: Rc<Vec<MillerLines>>,
}

impl<G2Affine: AffineRepr> VerifierFileReferenceString<G2Affine> {
    pub(crate) fn new(_path: &str) -> Self {
        // Please replace the actual types and functions with ones that you have in your Rust codebase.
        // let g2_x: G2Affine = read_transcript_g2(path);
        // let precomputed_g2_lines: Vec<MillerLines> = vec![MillerLines::default(); 2];

        // precompute_miller_lines(g2_x, &mut precomputed_g2_lines[1]);

        // Self {
        //     g2_x,
        //     precomputed_g2_lines,
        // }
        unimplemented!()
    }
}

impl<G2Affine: AffineRepr> VerifierReferenceString<G2Affine>
    for VerifierFileReferenceString<G2Affine>
{
    fn get_g2x(&self) -> G2Affine {
        self.g2_x
    }

    fn get_precomputed_g2_lines(&self) -> Rc<Vec<MillerLines>> {
        self.precomputed_g2_lines.clone()
    }
}

pub(crate) struct FileReferenceString<G1Affine: AffineRepr> {
    num_points: usize,
    pippenger: Pippenger,
    phantom: PhantomData<G1Affine>,
}

impl<G1Affine: AffineRepr> FileReferenceString<G1Affine> {
    pub(crate) fn new(_num_points: usize, _path: &str) -> Self {
        // Implementation depends on your project.
        todo!("FileReferenceString::new")
    }

    pub(crate) fn read_from_path(_path: &str) -> Result<Self, std::io::Error> {
        // Implementation depends on your project.
        todo!("FileReferenceString::read_from_path")
    }
}

impl<G1Affine: AffineRepr> Default for FileReferenceString<G1Affine> {
    fn default() -> Self {
        Self {
            num_points: 0,
            pippenger: Pippenger::default(),
            phantom: PhantomData,
        }
    }
}

impl<G1Affine: AffineRepr> ProverReferenceString<G1Affine> for FileReferenceString<G1Affine> {
    fn get_monomial_points(&mut self) -> Rc<Vec<G1Affine>> {
        // Implementation depends on your project.
        todo!()
    }

    fn get_monomial_size(&self) -> usize {
        self.num_points
    }
}

pub(crate) struct FileReferenceStringFactory<G1Affine: AffineRepr, G2Affine: AffineRepr> {
    path: String,
    phantom: PhantomData<(G1Affine, G2Affine)>,
}

impl<G1Affine: AffineRepr, G2Affine: AffineRepr> FileReferenceStringFactory<G1Affine, G2Affine> {
    pub(crate) fn new(path: String) -> Self {
        Self {
            path,
            phantom: PhantomData,
        }
    }
}
impl<G1Affine: AffineRepr, G2Affine: AffineRepr> ReferenceStringFactory<G1Affine, G2Affine>
    for FileReferenceStringFactory<G1Affine, G2Affine>
{
    fn get_prover_crs(&self, degree: usize) -> Option<Rc<dyn ProverReferenceString<G1Affine>>> {
        Some(Rc::new(FileReferenceString::<G1Affine>::new(
            degree, &self.path,
        )))
    }

    fn get_verifier_crs(&self) -> Option<Rc<dyn VerifierReferenceString<G2Affine>>> {
        Some(Rc::new(VerifierFileReferenceString::new(&self.path)))
    }
}

pub(crate) struct DynamicFileReferenceStringFactory<G1Affine: AffineRepr, G2Affine: AffineRepr> {
    path: String,
    degree: RefCell<usize>,
    prover_crs: RefCell<Rc<FileReferenceString<G1Affine>>>,
    verifier_crs: Rc<VerifierFileReferenceString<G2Affine>>,
    phantom: PhantomData<(G1Affine, G2Affine)>,
}

impl<G1Affine: AffineRepr, G2Affine: AffineRepr>
    DynamicFileReferenceStringFactory<G1Affine, G2Affine>
{
    pub(crate) fn new(path: String, initial_degree: usize) -> Self {
        let verifier_crs = Rc::new(VerifierFileReferenceString::new(&path));
        let prover_crs = RefCell::new(Rc::new(FileReferenceString::<G1Affine>::new(
            initial_degree,
            &path,
        )));
        Self {
            path,
            degree: RefCell::new(initial_degree),
            prover_crs,
            verifier_crs,
            phantom: PhantomData,
        }
    }
}

impl<G1Affine: AffineRepr, G2Affine: AffineRepr> ReferenceStringFactory<G1Affine, G2Affine>
    for DynamicFileReferenceStringFactory<G1Affine, G2Affine>
{
    fn get_prover_crs(&self, degree: usize) -> Option<Rc<dyn ProverReferenceString<G1Affine>>> {
        if degree != *self.degree.borrow() {
            *self.prover_crs.borrow_mut() =
                Rc::new(FileReferenceString::<G1Affine>::new(degree, &self.path));
            *self.degree.borrow_mut() = degree;
        }
        Some((self.prover_crs.borrow_mut()).clone())
    }

    fn get_verifier_crs(&self) -> Option<Rc<dyn VerifierReferenceString<G2Affine>>> {
        Some(self.verifier_crs.clone())
    }
}
