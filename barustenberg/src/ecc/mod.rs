use ark_ff::{FftField, Field};

// TODO todo - stubs to get the compiler to cooperate.
pub(crate) mod curves;
pub mod groups;

#[inline]
pub(crate) fn conditionally_subtract_from_double_modulus<Fr: Field + FftField>(
    _this: &Fr,
    _predicate: u64,
) -> Fr {
    todo!("see comment")
}

#[inline]
pub(crate) fn tag_coset_generator<Fr: Field + FftField>() -> Fr {
    todo!("see comment")
}
#[inline]
pub(crate) fn coset_generator<Fr: Field + FftField>(_n: u8) -> Fr {
    todo!("see comment")
}
