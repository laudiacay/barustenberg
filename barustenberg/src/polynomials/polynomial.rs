use std::marker::PhantomData;

use crate::ecc::fields::field::FieldParams;

#[derive(Debug, Clone)]
pub(crate) struct Polynomial<F: FieldParams> {
    phandom: PhantomData<F>,
}
