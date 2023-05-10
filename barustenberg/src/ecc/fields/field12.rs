use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::ecc::EllCoeffs;

use super::{
    field::Field,
    field2::{Field2, Field2Params},
    field6::{Field6, Field6Impl, Field6Params},
};

pub trait Field12Params<
    F1: Field,
    F2: Field2<F1, dyn Field2Params<F1>>,
    F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
>
{
    const frobenius_coefficients_1: F2;
    const frobenius_coefficients_2: F2;
    const frobenius_coefficients_3: F2;
}

pub trait Field12<
    F1: Field,
    F2: Field2<F1, dyn Field2Params<F1>>,
    F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
    Params: Field12Params<F1, F2, F6>,
>
{
    fn new(c0: F6, c1: F6) -> Self;
    fn zero() -> Self {
        Self::new(F6::zero(), F6::zero())
    }

    fn one() -> Self {
        Self::new(F6::one(), F6::zero())
    }

    fn mul_by_non_residue(&self, a: &F6) -> F6 {
        Field6Impl {
            c0: F6::mul_by_non_residue(a.c2),
            c1: a.c0,
            c2: a.c1,
            phantom: PhantomData,
        }
    }

    fn self_sparse_mul(&mut self, ell: &EllCoeffs<F2>) {
        todo!("todo")
    }

    fn sqr(&self) -> Self {
        todo!()
    }
    fn invert(&self) -> Self {
        todo!()
    }
    fn frobenius_map_three(&self) -> Self {
        todo!()
    }
    fn frobenius_map_two(&self) -> Self {
        todo!()
    }
    fn frobenius_map_one(&self) -> Self {
        todo!()
    }
    fn cyclotomic_squared(&self) -> Self {
        todo!()
    }
    fn unitary_inverse(&self) -> Self {
        todo!()
    }
    fn random_element(engine: &mut impl rand::RngCore) -> Self {
        todo!()
    }
    fn from_montgomery_form(&self) -> Self {
        todo!()
    }
    fn is_zero(&self) -> bool {
        todo!()
    }
}

// eq
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > PartialEq for dyn Field12<F1, F2, F6, Params>
{
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}
//add
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > Add for dyn Field12<F1, F2, F6, Params>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.c0 + other.c0, self.c1 + other.c1)
    }
}
// sub
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > Sub for dyn Field12<F1, F2, F6, Params>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.c0 - other.c0, self.c1 - other.c1)
    }
}
// mul
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > Mul for dyn Field12<F1, F2, F6, Params>
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        todo!()
    }
}
// div
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > Div for dyn Field12<F1, F2, F6, Params>
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        todo!()
    }
}
// addassign
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > AddAssign for dyn Field12<F1, F2, F6, Params>
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs)
    }
}
// subassign
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > SubAssign for dyn Field12<F1, F2, F6, Params>
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs)
    }
}
// mulassign
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > MulAssign for dyn Field12<F1, F2, F6, Params>
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs)
    }
}
// divassign
impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > DivAssign for dyn Field12<F1, F2, F6, Params>
{
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div(rhs)
    }
}

pub struct Field12Impl<
    F1: Field,
    F2: Field2<F1, dyn Field2Params<F1>>,
    F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
    Params: Field12Params<F1, F2, F6>,
> {
    c0: F6,
    c1: F6,
    phantom: PhantomData<Params>,
}

impl<
        F1: Field,
        F2: Field2<F1, dyn Field2Params<F1>>,
        F6: Field6<F1, F2, dyn Field6Params<F1, F2>>,
        Params: Field12Params<F1, F2, F6>,
    > Field12<F1, F2, F6, Params> for Field12Impl<F1, F2, F6, Params>
{
    fn new(c0: F6, c1: F6) -> Self {
        Self {
            c0,
            c1,
            phantom: PhantomData,
        }
    }
}
