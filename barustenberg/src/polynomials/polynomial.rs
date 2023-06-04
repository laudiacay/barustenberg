use std::{
    marker::PhantomData,
    ops::{AddAssign, Index, SubAssign},
};

use ark_ff::Field;

#[derive(Debug, Clone, Default)]
pub(crate) struct Polynomial<F: Field> {
    size: usize,
    coefficients: Vec<F>,
    phantom: PhantomData<F>,
}

impl<F: Field> Polynomial<F> {
    pub(crate) fn from_interpolations(_interpolation_points: &[F], _values: &[F]) -> Self {
        todo!("unimplemented, see comment below");
    }

    #[inline]
    pub(crate) fn new(size: usize) -> Self {
        Self {
            size,
            coefficients: vec![F::zero(); size],
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
}

impl<F: Field> AddAssign for Polynomial<F> {
    fn add_assign(&mut self, _rhs: Self) {
        todo!("unimplemented, see comment below");
    }
}

impl<F: Field> SubAssign for Polynomial<F> {
    fn sub_assign(&mut self, _rhs: Self) {
        todo!("unimplemented, see comment below");
    }
}

impl<F: Field> std::ops::MulAssign<F> for Polynomial<F> {
    fn mul_assign(&mut self, _rhs: F) {
        todo!("unimplemented, see comment below");
    }
}

impl<F: Field> IntoIterator for Polynomial<F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        todo!("unimplemented, see comment below");
    }
}

impl<F: Field> Index<usize> for Polynomial<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}
