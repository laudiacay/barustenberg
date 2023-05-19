use std::marker::PhantomData;

use crate::ecc::fields::field::FieldParams;

pub(crate) struct Polynomial<F: FieldParams> {
    phandom: PhantomData<F>,
}
