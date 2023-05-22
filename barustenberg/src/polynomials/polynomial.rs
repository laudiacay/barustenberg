use std::marker::PhantomData;

use ark_ff::Field;

#[derive(Debug, Clone)]
pub(crate) struct Polynomial<F: Field> {
    degree: usize,
    phantom: PhantomData<F>,
}

impl<F: Field> Polynomial<F> {
    pub(crate) const fn new(degree: usize) -> Self {
        Self {
            degree,
            phantom: PhantomData,
        }
    }
    pub(crate) const fn get_degree(&self) -> usize {
        todo!("unimplemented, see comment below");
    }
    pub(crate) const fn size(&self) -> usize {
        todo!("unimplemented, see comment below");
    }
}
