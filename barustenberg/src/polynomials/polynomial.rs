use std::{
    marker::PhantomData,
    ops::{AddAssign, Index, MulAssign, Range, SubAssign},
};

use anyhow::{anyhow, Result};
use ark_ff::Field;

#[derive(Debug, Clone, Default)]
pub(crate) struct Polynomial<'a, F: Field> {
    size: usize,
    coefficients: &'a [F],
    /// storage for the coefficients of the polynomial, if they happen to belong to this one :)
    underlying_coefficients: Option<Vec<F>>,
    phantom: PhantomData<F>,
}

impl<'a, F: Field> Polynomial<'a, F> {
    pub(crate) fn from_interpolations(_interpolation_points: &[F], _values: &[F]) -> Self {
        todo!("unimplemented, see comment below");
    }

    #[inline]
    pub(crate) fn new(size: usize) -> Self {
        let underlying = vec![F::zero(); size];
        Self {
            size,
            coefficients: &underlying,
            underlying_coefficients: Some(underlying),
            phantom: PhantomData,
        }
    }
    /// TODO Question to Claudia: Aren't these functionally the same?
    #[inline]
    pub(crate) fn get_degree(&self) -> usize {
        todo!("unimplemented, see comment below");
    }
    #[inline]
    pub(crate) fn size(&self) -> usize {
        todo!("unimplemented, see comment below");
    }
    #[inline]
    pub(crate) fn get_coefficients(&self) -> &[F] {
        self.coefficients
    }
    #[inline]
    pub(crate) fn set_coefficient(&mut self, idx: usize, v: F) {
        self.coefficients[idx] = v
    }
    #[inline]
    pub(crate) fn get_mut_coefficients(&mut self) -> &mut [F] {
        &mut self.coefficients
    }
    #[inline]
    pub(crate) fn resize(&mut self, new_len: usize, val: F) -> Result<()> {
        match self.underlying_coefficients {
            Some(ref mut underlying) => {
                underlying.resize(new_len, val);
                self.coefficients = underlying;
                return Ok(());
            }
            None => Err(anyhow!(
                "Cannot resize polynomial without owning the underlying coefficients"
            )),
        }
    }
}

impl<'a, F: Field> AddAssign for Polynomial<'a, F> {
    fn add_assign(&mut self, _rhs: Self) {
        todo!("unimplemented, see comment below");
    }
}

impl<'a, F: Field> SubAssign for Polynomial<'a, F> {
    fn sub_assign(&mut self, _rhs: Self) {
        todo!("unimplemented, see comment below");
    }
}

impl<'a, F: Field> MulAssign<F> for Polynomial<'a, F> {
    fn mul_assign(&mut self, _rhs: F) {
        todo!("unimplemented, see comment below");
    }
}

impl<'a, F: Field> IntoIterator for Polynomial<'a, F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        todo!("unimplemented, see comment below");
    }
}

impl<'a, F: Field> Index<usize> for Polynomial<'a, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}

impl<'a, F: Field> Index<Range<usize>> for Polynomial<'a, F> {
    type Output = &'a [F];

    fn index(&self, index: Range<usize>) -> &'a Self::Output {
        &&self.coefficients[index]
    }
}

impl<'a, F: Field> Index<std::ops::RangeFrom<usize>> for Polynomial<'a, F> {
    type Output = &'a [F];

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &'a Self::Output {
        &&self.coefficients[index]
    }
}
