use std::marker::PhantomData;

use crate::ecc::fields::field::FieldParams;

#[derive(Debug, Clone)]
pub(crate) struct Polynomial<F: FieldParams> {
    phantom: PhantomData<F>,
}

impl<F: FieldParams> Polynomial<F> {
    pub(crate) const fn new() -> Self {
        Self {
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
