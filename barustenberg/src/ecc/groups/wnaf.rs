pub(crate) const SCALAR_BITS: usize = 127;
pub(crate) const fn wnaf_size(x: usize) -> usize {
    (SCALAR_BITS + x - 1) / x
}
