pub(crate) trait FieldExt {}

impl<T: ark_ff::Field + ark_ff::FftField> FieldExt for T {}
