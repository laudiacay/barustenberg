use std::{
    marker::PhantomData,
    ops::{AddAssign, Index, IndexMut, MulAssign, Range, SubAssign},
};

use ark_ff::Field;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Polynomial<F: Field> {
    size: usize,
    pub(crate) coefficients: Vec<F>,
    phantom: PhantomData<F>,
}

impl<F: Field> Polynomial<F> {
    pub(crate) fn from_interpolations(_interpolation_points: &[F], _values: &[F]) -> Self {
        todo!("unimplemented, see comment below");
    }

    #[inline]
    pub(crate) fn new(size: usize) -> Self {
        let underlying = vec![F::zero(); size];
        Self {
            size,
            coefficients: underlying,
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
    pub(crate) fn set_coefficient(&mut self, idx: usize, v: F) {
        self.coefficients[idx] = v
    }
    #[inline]
    pub(crate) fn resize(&mut self, new_len: usize, val: F) {
        self.coefficients.resize(new_len, val)
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

impl<F: Field> MulAssign<F> for Polynomial<F> {
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

impl<F: Field> IndexMut<usize> for Polynomial<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}

impl<F: Field> Index<Range<usize>> for Polynomial<F> {
    type Output = [F];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.coefficients[index]
    }
}

impl<F: Field> IndexMut<Range<usize>> for Polynomial<F> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}
impl<F: Field> Index<std::ops::RangeFrom<usize>> for Polynomial<F> {
    type Output = [F];

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.coefficients[index]
    }
}
