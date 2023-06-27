use std::ops::{AddAssign, Index, IndexMut, MulAssign, Range, SubAssign};

use anyhow::Result;
use ark_ff::{FftField, Field};

use crate::polynomials::polynomial_arithmetic;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Polynomial<F: Field + FftField> {
    size: usize,
    pub(crate) coefficients: Vec<F>,
}

impl<F: Field + FftField> Polynomial<F> {
    pub(crate) fn from_interpolations(
        interpolation_points: &[F],
        evaluations: &[F],
    ) -> Result<Self> {
        assert!(!interpolation_points.is_empty());
        let mut coefficients = vec![F::zero(); interpolation_points.len()];
        polynomial_arithmetic::compute_efficient_interpolation(
            evaluations,
            &mut coefficients,
            interpolation_points,
            interpolation_points.len(),
        )?;
        Ok(Self {
            size: interpolation_points.len(),
            coefficients,
        })
    }

    #[inline]
    pub(crate) fn new(size: usize) -> Self {
        let underlying = vec![F::zero(); size];
        Self {
            size,
            coefficients: underlying,
        }
    }
    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.size
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

impl<F: Field + FftField> AddAssign for Polynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        // pad the smaller polynomial with zeros
        if self.size < rhs.size {
            self.resize(rhs.size, F::zero());
        }
        for i in 0..rhs.size {
            self.coefficients[i] += rhs.coefficients[i];
        }
    }
}

impl<F: Field + FftField> SubAssign for Polynomial<F> {
    fn sub_assign(&mut self, rhs: Self) {
        // pad the smaller polynomial with zeros
        if self.size < rhs.size {
            self.resize(rhs.size, F::zero());
        }
        for i in 0..rhs.size {
            self.coefficients[i] -= rhs.coefficients[i];
        }
    }
}

impl<F: Field + FftField> MulAssign<F> for Polynomial<F> {
    fn mul_assign(&mut self, rhs: F) {
        for i in 0..self.size {
            self.coefficients[i] *= rhs;
        }
    }
}

impl<F: Field + FftField> IntoIterator for Polynomial<F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.coefficients.into_iter()
    }
}

impl<F: Field + FftField> Index<usize> for Polynomial<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}

impl<F: Field + FftField> IndexMut<usize> for Polynomial<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}

impl<F: Field + FftField> Index<Range<usize>> for Polynomial<F> {
    type Output = [F];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.coefficients[index]
    }
}

impl<F: Field + FftField> IndexMut<Range<usize>> for Polynomial<F> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}
impl<F: Field + FftField> Index<std::ops::RangeFrom<usize>> for Polynomial<F> {
    type Output = [F];

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.coefficients[index]
    }
}

impl<F: Field + FftField> Polynomial<F> {
    // Evaluate the polynomial at a given point
    pub(crate) fn evaluate(&self, z: &F) -> F {
        polynomial_arithmetic::evaluate(&self.coefficients, z, self.size())
    }

    /// Evaluates p(X) = ∑ᵢ aᵢ⋅Xⁱ considered as multi-linear extension p(X₀,…,Xₘ₋₁) = ∑ᵢ aᵢ⋅Lᵢ(X₀,…,Xₘ₋₁)
    /// at u = (u₀,…,uₘ₋₁)
    ///
    /// This function allocates a temporary buffer of size n/2
    ///
    /// # Arguments
    ///
    /// * `evaluation_points` - an MLE evaluation point u = (u₀,…,uₘ₋₁)
    /// * `shift` - evaluates p'(X₀,…,Xₘ₋₁) = 1⋅L₀(X₀,…,Xₘ₋₁) + ∑ᵢ˲₁ aᵢ₋₁⋅Lᵢ(X₀,…,Xₘ₋₁) if true
    ///
    /// # Returns
    ///
    /// - Fr p(u₀,…,uₘ₋₁)
    ///
    pub(crate) fn evaluate_mle(&self, evaluation_points: &[F], shift: bool) -> F {
        let m = evaluation_points.len();

        // To simplify handling of edge cases, we assume that size is always a power of 2
        assert_eq!(self.size, 1 << m);

        // we do m rounds l = 0,...,m-1.
        // in round l, n_l is the size of the buffer containing the polynomial partially evaluated
        // at u₀,..., u_l.
        // in round 0, this is half the size of n
        let mut n_l = 1 << (m - 1);

        // temporary buffer of half the size of the polynomial
        let mut tmp: Vec<F> = vec![F::zero(); n_l];

        let mut prev = self.coefficients.clone();
        if shift {
            assert_eq!(prev[0], F::zero());
            prev.remove(0);
        }

        let mut u_l = evaluation_points[0];
        for i in 0..n_l {
            // curr[i] = (Fr(1) - u_l) * prev[i << 1] + u_l * prev[(i << 1) + 1];
            tmp[i] =
                prev[i << 1] + u_l * (*prev.get((i << 1) + 1).unwrap_or(&F::zero()) - prev[i << 1]);
        }
        // partially evaluate the m-1 remaining points
        for l in 1..m {
            n_l = 1 << (m - l - 1);
            u_l = evaluation_points[l];
            for i in 0..n_l {
                tmp[i] = tmp[i << 1] + u_l * (tmp[(i << 1) + 1] - tmp[i << 1]);
            }
        }
        tmp[0]
    }

    // Factor roots out of the polynomial
    pub(crate) fn factor_root(&mut self, root: &F) {
        polynomial_arithmetic::factor_root(&mut self.coefficients, root)
    }

    // Factor roots out of the polynomial
    /// Divides p(X) by (X-r₁)⋯(X−rₘ) in-place.
    /// Assumes that p(rⱼ)=0 for all j
    ///
    /// We specialize the method when only a single root is given.
    /// if one of the roots is 0, then we first factor all other roots.
    /// dividing by X requires only a left shift of all coefficient.
    ///
    /// # Arguments
    ///
    /// * roots list of roots (r₁,…,rₘ)
    pub(crate) fn factor_roots(&mut self, roots: &[F]) {
        polynomial_arithmetic::factor_roots(&mut self.coefficients, roots)
    }
}
