use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::ecc::EllCoeffs;

use super::{
    field::{FieldGeneral, FieldParams, FieldParamsGeneral},
    field2::{Field2, Field2Params},
    field6::{Field6, Field6Params},
};

pub trait Field12Params<F1P: FieldParams, F2P: Field2Params<F1P>>: FieldParamsGeneral {
    const frobenius_coefficients_1: Field2<F1P, F2P>;
    const frobenius_coefficients_2: Field2<F1P, F2P>;
    const frobenius_coefficients_3: Field2<F1P, F2P>;
}

pub struct Field12<
    F1P: FieldParams,
    F2P: Field2Params<F1P>,
    F6P: Field6Params<F1P, F2P>,
    Params: Field12Params<F1P, F2P>,
> {
    c0: Field6<F1P, F2P, F6P>,
    c1: Field6<F1P, F2P, F6P>,
    phantom: PhantomData<Params>,
}

impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > FieldGeneral<Params> for Field12<F1P, F2P, F6P, Params>
{
}

impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > Field12<F1P, F2P, F6P, Params>
{
    fn new(c0: Field6<F1P, F2P, F6P>, c1: Field6<F1P, F2P, F6P>) -> Self {
        Self {
            c0,
            c1,
            phantom: PhantomData,
        }
    }
    fn zero() -> Self {
        Self::new(
            Field6::<F1P, F2P, F6P>::zero(),
            Field6::<F1P, F2P, F6P>::zero(),
        )
    }

    fn one() -> Self {
        Self::new(Self::F6::one(), Self::F6::zero())
    }

    fn mul_by_non_residue(&self, a: &Field6<F1P, F2P, F6P>) -> Field6<F1P, F2P, F6P> {
        Self::F6 {
            c0: Field6::<F1P, F2P, F6P>::mul_by_non_residue(a.c2),
            c1: a.c0,
            c2: a.c1,
            phantom: PhantomData,
        }
    }

    fn self_sparse_mul(&mut self, ell: &EllCoeffs<F2P, Field2<F1P, F2P>>) {
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
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > PartialEq for Field12<F1P, F2P, F6P, Params>
{
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0 && self.c1 == other.c1
    }
}
//add
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > Add for Field12<F1P, F2P, F6P, Params>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.c0 + other.c0, self.c1 + other.c1)
    }
}
// sub
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > Sub for Field12<F1P, F2P, F6P, Params>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.c0 - other.c0, self.c1 - other.c1)
    }
}
// mul
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > Mul for Field12<F1P, F2P, F6P, Params>
{
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        todo!()
    }
}
// div
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > Div for Field12<F1P, F2P, F6P, Params>
{
    type Output = Self;

    fn div(self, other: Self) -> Self {
        todo!()
    }
}
// addassign
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > AddAssign for Field12<F1P, F2P, F6P, Params>
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs)
    }
}
// subassign
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > SubAssign for Field12<F1P, F2P, F6P, Params>
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs)
    }
}
// mulassign
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > MulAssign for Field12<F1P, F2P, F6P, Params>
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul(rhs)
    }
}
// divassign
impl<
        F1P: FieldParams,
        F2P: Field2Params<F1P>,
        F6P: Field6Params<F1P, F2P>,
        Params: Field12Params<F1P, F2P>,
    > DivAssign for Field12<F1P, F2P, F6P, Params>
{
    fn div_assign(&mut self, rhs: Self) {
        *self = self.div(rhs)
    }
}
