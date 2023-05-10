use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{
    field::Field,
    field2::{Field2, Field2Params},
};

pub trait Field6Params<F1: Field, F: Field2<F1, dyn Field2Params<F1>>> {
    const frobenius_coeffs_c1_1: F;
    const frobenius_coeffs_c1_2: F;
    const frobenius_coeffs_c1_3: F;
    const frobenius_coeffs_c2_1: F;
    const frobenius_coeffs_c2_2: F;
    const frobenius_coeffs_c2_3: F;
}

pub trait Field6<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> {
    // non residue = 9 + i \in Fq2
    fn mul_by_non_residue(a: &F2) -> F2 {
        // non residue = 9 + i \in Fq2
        // r.c0 = 9a0 - a1
        // r.c1 = 9a1 + a0
        let mut t0: F1 = a.c0 + a.c0;
        t0 += t0;
        t0 += t0;
        t0 += a.c0;
        let mut t1: F1 = a.c1 + a.c1;
        t1 += t1;
        t1 += t1;
        t1 += a.c1;
        let t2: F1 = t1 - a.c1;

        F2::new_from_elems(t2, t1 + a.c0);
        // T0 = a.c0 + a.c0; ???
    }

    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;

    fn sqr(&self) -> Self;
    fn invert(&self) -> Self;
    fn mul_by_fq2(&self, other: &F2) -> Self;
    fn frobenius_map_one(&self) -> Self;
    fn frobenius_map_two(&self) -> Self;
    fn frobenius_map_three(&self) -> Self;
    fn random_element(engine: &mut impl rand::RngCore) -> Self;
    fn to_montgomery_form(&self) -> Self;
    fn from_montgomery_form(&self) -> Self;
}

pub struct Field6Impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>>
{
    c0: F2,
    c1: F2,
    c2: F2,
    phantom: PhantomData<Params>,
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>>
    Field6<F1, F2, Params> for Field6Impl<F1, F2, Params>
{
    fn zero() -> Self {
        Field6Impl {
            c0: F2::zero(),
            c1: F2::zero(),
            c2: F2::zero(),
            phantom: PhantomData,
        }
    }

    fn one() -> Self {
        Field6Impl {
            c0: F2::one(),
            c1: F2::zero(),
            c2: F2::zero(),
            phantom: PhantomData,
        }
    }
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> Add
    for dyn Field6<F1, F2, Params>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Field6Impl {
            c0: self.c0 + other.c0,
            c1: self.c1 + other.c1,
            c2: self.c2 + other.c2,
            phantom: PhantomData,
        }
    }
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> Sub
    for dyn Field6<F1, F2, Params>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Field6Impl {
            c0: self.c0 - other.c0,
            c1: self.c1 - other.c1,
            c2: self.c2 - other.c2,
            phantom: PhantomData,
        }
    }
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> Neg
    for dyn Field6<F1, F2, Params>
{
    fn neg(self) -> Self {
        Field6Impl {
            c0: -self.c0,
            c1: -self.c1,
            c2: -self.c2,
            phantom: PhantomData,
        }
    }
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> Mul
    for dyn Field6<F1, F2, Params>
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 4
        //  * (Karatsuba) */
        let t0 = self.c0 * other.c0;
        let t1 = self.c1 * other.c1;
        let t2 = self.c2 * other.c2;

        let t3 = (self.c0 + self.c2) * (other.c0 + other.c2);
        let t4 = (self.c0 + self.c1) * (other.c0 + other.c1);
        let t5 = (self.c1 + self.c2) * (other.c1 + other.c2);

        Field6Impl {
            c0: t0 + Self::mul_by_non_residue(t5 - (t1 + t2)),
            c1: t4 - (t0 + t1) + Self::mul_by_non_residue(t2),
            c2: t3 + t1 - (t0 + t2),
            phantom: PhantomData,
        }
    }
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> Div
    for dyn Field6<F1, F2, Params>
{
    fn div(self, other: Self) -> Self {
        self * other.invert()
    }
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> AddAssign
    for dyn Field6<F1, F2, Params>
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other
    }
}
impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> SubAssign
    for dyn Field6<F1, F2, Params>
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}
impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> MulAssign
    for dyn Field6<F1, F2, Params>
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}
impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> DivAssign
    for dyn Field6<F1, F2, Params>
{
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}

impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> PartialEq
    for dyn Field6<F1, F2, Params>
{
}
impl<F1: Field, F2: Field2<F1, dyn Field2Params<F1>>, Params: Field6Params<F1, F2>> Eq
    for dyn Field6<F1, F2, Params>
{
}
