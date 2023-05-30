use std::{marker::PhantomData, ops::{AddAssign, SubAssign}};

use ark_ff::Field;

#[derive(Debug, Clone, Default)]
pub(crate) struct Polynomial<F: Field> {
    size: usize,
    coefficients: Vec<F>,
    phantom: PhantomData<F>,
}

impl<F: Field> Polynomial<F> {
    pub fn from_interpolations(interpolation_points: &[F], values: &[F]) -> Self {
        todo!("unimplemented, see comment below");
    }

    pub(crate) const fn new(size: usize) -> Self {
        Self {
            size,
            coefficients: vec![F::zero(); size],
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

impl<F:Field> AddAssign for Polynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        todo!("unimplemented, see comment below");
    }
}

impl<F:Field> SubAssign for Polynomial<F> {
    fn sub_assign(&mut self, rhs: Self) {
        todo!("unimplemented, see comment below");
    }
}

impl<F: Field> std::ops::MulAssign<F> for Polynomial<F> {
    fn mul_assign(&mut self, rhs: F) {
        todo!("unimplemented, see comment below");
    }
}

impl<F:Field> IntoIterator for Polynomial<F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        todo!("unimplemented, see comment below");
    }
}
