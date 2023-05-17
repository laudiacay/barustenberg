use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::{
    field::{FieldGeneral, FieldParams, FieldParamsGeneral},
    field2::{Field2, Field2Params},
};

pub trait Field6Params<F1P: FieldParams, F2P: Field2Params<F1P>>: FieldParamsGeneral {
    const frobenius_coeffs_c1_1: Field2<F1P, F2P>;
    const frobenius_coeffs_c1_2: Field2<F1P, F2P>;
    const frobenius_coeffs_c1_3: Field2<F1P, F2P>;
    const frobenius_coeffs_c2_1: Field2<F1P, F2P>;
    const frobenius_coeffs_c2_2: Field2<F1P, F2P>;
    const frobenius_coeffs_c2_3: Field2<F1P, F2P>;
}

pub struct Field6<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> {
    pub c0: Field2<F1P, F2P>,
    pub c1: Field2<F1P, F2P>,
    pub c2: Field2<F1P, F2P>,
    phantom: PhantomData<Params>,
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> FieldGeneral<Params>
    for Field6<F1P, F2P, Params>
{
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>>
    Field6<F1P, F2P, Params>
{
    pub fn zero() -> Self {
        Field6 {
            c0: Field2::<F1P, F2P>::zero(),
            c1: Field2::<F1P, F2P>::zero(),
            c2: Field2::<F1P, F2P>::zero(),
            phantom: PhantomData,
        }
    }

    pub fn one() -> Self {
        Field6 {
            c0: Field2::<F1P, F2P>::one(),
            c1: Field2::<F1P, F2P>::zero(),
            c2: Field2::<F1P, F2P>::zero(),
            phantom: PhantomData,
        }
    }
    fn is_zero(&self) -> bool {}

    fn sqr(&self) -> Self {}
    fn invert(&self) -> Self {}
    fn mul_by_fq2(&self, other: &Field2<F1P, F2P>) -> Self {}
    fn frobenius_map_one(&self) -> Self {}
    fn frobenius_map_two(&self) -> Self {}
    fn frobenius_map_three(&self) -> Self {}
    fn random_element(engine: &mut impl rand::RngCore) -> Self {}
    fn to_montgomery_form(&self) -> Self {}
    fn from_montgomery_form(&self) -> Self {}
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> Add
    for Field6<F1P, F2P, Params>
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Field6 {
            c0: self.c0 + other.c0,
            c1: self.c1 + other.c1,
            c2: self.c2 + other.c2,
            phantom: PhantomData,
        }
    }
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> Sub
    for Field6<F1P, F2P, Params>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Field6 {
            c0: self.c0 - other.c0,
            c1: self.c1 - other.c1,
            c2: self.c2 - other.c2,
            phantom: PhantomData,
        }
    }
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> Neg
    for Field6<F1P, F2P, Params>
{
    fn neg(self) -> Self {
        Field6 {
            c0: -self.c0,
            c1: -self.c1,
            c2: -self.c2,
            phantom: PhantomData,
        }
    }
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> Mul
    for Field6<F1P, F2P, Params>
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

        Field6 {
            c0: t0 + (t5 - (t1 + t2)).fq6_mul_by_non_residue(),
            c1: t4 - (t0 + t1) + t2.fq6_mul_by_non_residue(),
            c2: t3 + t1 - (t0 + t2),
            phantom: PhantomData,
        }
    }
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> Div
    for Field6<F1P, F2P, Params>
{
    fn div(self, other: Self) -> Self {
        self * other.invert()
    }
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> AddAssign
    for Field6<F1P, F2P, Params>
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other
    }
}
impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> SubAssign
    for Field6<F1P, F2P, Params>
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}
impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> MulAssign
    for Field6<F1P, F2P, Params>
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}
impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> DivAssign
    for Field6<F1P, F2P, Params>
{
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}

impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> PartialEq
    for Field6<F1P, F2P, Params>
{
}
impl<F1P: FieldParams, F2P: Field2Params<F1P>, Params: Field6Params<F1P, F2P>> Eq
    for Field6<F1P, F2P, Params>
{
}
