use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::ecc::fields::field::Field;
use primitive_types::U256;
use serde::{Deserialize, Serialize};

use super::field::{FieldGeneral, FieldParams, FieldParamsGeneral};

pub trait Field2Params<F1P: FieldParams>: FieldParamsGeneral {
    const twist_coeff_b_0: Field<F1P>;
    const twist_coeff_b_1: Field<F1P>;
    const twist_mul_by_q_x_0: Field<F1P>;
    const twist_mul_by_q_x_1: Field<F1P>;
    const twist_mul_by_q_y_0: Field<F1P>;
    const twist_mul_by_q_y_1: Field<F1P>;
    const twist_cube_root_0: Field<F1P>;
    const twist_cube_root_1: Field<F1P>;
}

pub struct Field2<F1P: FieldParams, Params: Field2Params<F1P>> {
    c0: Field<F1P>,
    c1: Field<F1P>,
    phantom: PhantomData<Params>,
}

impl<F1P: FieldParams, Params: Field2Params<F1P>> FieldGeneral<Params> for Field2<F1P, Params> {}

impl<F1P: FieldParams, Params: Field2Params<F1P>> PartialEq for Field2<F1P, Params> {
    // todo
}
impl<F1P: FieldParams, Params: Field2Params<F1P>> Eq for Field2<F1P, Params> {
    // todo
}

impl<F1P: FieldParams, Params: Field2Params<F1P>> Add for Field2<F1P, Params> {
    type Output = Self;
    //    return { c0 + other.c0, c1 + other.c1 };
}
impl<F1P: FieldParams, Params: Field2Params<F1P>> Sub for Field2<F1P, Params> {
    type Output = Self;
    //    return { c0 - other.c0, c1 - other.c1 };
}
impl<F1P: FieldParams, Params: Field2Params<F1P>> Mul for Field2<F1P, Params> {
    type Output = Self;
    /*
        // no funny primes please! we assume -1 is not a quadratic residue
    static_assert((base::modulus.data[0] & 0x3UL) == 0x3UL);
    base t1 = c0 * other.c0;
    base t2 = c1 * other.c1;
    base t3 = c0 + c1;
    base t4 = other.c0 + other.c1;

    return { t1 - t2, t3 * t4 - (t1 + t2) };
     */
}
impl<F1P: FieldParams, Params: Field2Params<F1P>> Neg for Field2<F1P, Params> {
    type Output = Self;
    //    return { -c0, -c1 };
}
impl<F1P: FieldParams, Params: Field2Params<F1P>> Div for Field2<F1P, Params> {
    type Output = Self;
    //    return operator*(other.invert());
}

impl<F1P: FieldParams, Params: Field2Params<F1P>> AddAssign for Field2<F1P, Params> {
    fn add_assign(&mut self, rhs: Self) {
        self = self + rhs
    }
}

impl<F1P: FieldParams, Params: Field2Params<F1P>> SubAssign for Field2<F1P, Params> {
    fn sub_assign(&mut self, rhs: Self) {
        self = self - rhs
    }
}

impl<F1P: FieldParams, Params: Field2Params<F1P>> MulAssign for Field2<F1P, Params> {
    fn mul_assign(&mut self, rhs: Self) {
        self = self * rhs
    }
}

impl<F1P: FieldParams, Params: Field2Params<F1P>> DivAssign for Field2<F1P, Params> {
    fn div_assign(&mut self, rhs: Self) {
        self = self / rhs
    }
}

impl<F1P: FieldParams, Params: Field2Params<F1P>> Serialize for Field2<F1P, Params> {}

impl<'de, F1P: FieldParams, Params: Field2Params<F1P>> Deserialize<'de> for Field2<F1P, Params> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}

// TODO a shitton of this stuff should be done with macros at compiletime for speed.
impl<F1P: FieldParams, Params: Field2Params<F1P>> Field2<F1P, Params> {
    // TODO sin that this is a function
    pub fn modulus() -> U256 {
        Field::<F1P>::modulus()
    }

    pub fn new() -> Self {
        Self::zero()
    }

    pub fn zero() -> Self {
        Field2 {
            c0: Self::BaseField::zero(),
            c1: Self::BaseField::zero(),
            phantom: PhantomData,
        }
    }
    pub fn one() -> Self {
        Field2 {
            c0: Self::BaseField::one(),
            c1: Self::BaseField::zero(),
            phantom: PhantomData,
        }
    }
    pub fn twist_coeff_b() -> Self {
        Field2 {
            c0: Params::twist_coeff_b_0,
            c1: Params::twist_coeff_b_1,
            phantom: PhantomData,
        }
    }
    fn twist_mul_by_q_x() -> Self {
        Field2 {
            c0: Params::twist_mul_by_q_x_0,
            c1: Params::twist_mul_by_q_x_1,
            phantom: PhantomData,
        }
    }
    fn twist_mul_by_q_y() -> Self {
        Field2 {
            c0: Params::twist_mul_by_q_y_0,
            c1: Params::twist_mul_by_q_y_1,
            phantom: PhantomData,
        }
    }
    fn cube_root_of_unity() -> Self {
        Field2 {
            c0: Params::twist_cube_root_0,
            c1: Params::twist_cube_root_1,
            phantom: PhantomData,
        }
    }

    // non residue = 9 + i \in Fq2
    pub fn fq6_mul_by_non_residue(&self) -> Self {
        // non residue = 9 + i \in Fq2
        // r.c0 = 9a0 - a1
        // r.c1 = 9a1 + a0
        let mut t0: Self::F1 = self.c0 + self.c0;
        t0 += t0;
        t0 += t0;
        t0 += self.c0;
        let mut t1: Self::F1 = self.c1 + self.c1;
        t1 += t1;
        t1 += t1;
        t1 += self.c1;
        let t2: Self::F1 = t1 - self.c1;

        Self::new_from_elems(t2, t1 + self.c0)
        // T0 = a.c0 + a.c0; ???
    }

    fn sqr(&self) -> Self {
        let t1 = self.c0 * self.c1;
        Field2 {
            c0: (self.c0 + self.c1) * (self.c0 - self.c1),
            c1: t1 + t1,
            phantom: PhantomData,
        }
    }

    fn self_sqr(&mut self) {
        *self = self.sqr();
    }

    fn to_montgomery_form(&self) -> Self {
        Field2 {
            c0: self.c0.to_montgomery_form(),
            c1: self.c1.to_montgomery_form(),
            phantom: PhantomData,
        }
    }

    fn from_montgomery_form(&self) -> Self {
        Field2 {
            c0: self.c0.from_montgomery_form(),
            c1: self.c1.from_montgomery_form(),
            phantom: PhantomData,
        }
    }
    fn mul_by_fq(&self, a: Field<F1P>) -> Self {
        Field2 {
            c0: a * &self.c0,
            c1: a * &self.c1,
            phantom: PhantomData,
        }
    }

    fn pow(&self, exponent: &U256) -> Self {
        /*
                    field2 accumulator = *this;
        field2 to_mul = *this;
        const uint64_t maximum_set_bit = exponent.get_msb();

        for (int i = static_cast<int>(maximum_set_bit) - 1; i >= 0; --i) {
            accumulator.self_sqr();
            if (exponent.get_bit(static_cast<uint64_t>(i))) {
                accumulator *= to_mul;
            }
        }

        if (*this == zero()) {
            accumulator = zero();
        } else if (exponent == uint256_t(0)) {
            accumulator = one();
        }
        return accumulator;
                 */
        todo!() // Implement the pow method with uint256 exponent
    }

    fn pow_u64(&self, exponent: u64) -> Self {
        /*
        return pow({ exponent, 0, 0, 0 })
         */
        todo!() // Implement the pow method with u64 exponent
    }

    fn invert(&self) -> Self {
        /*
                base t3 = (c0.sqr() + c1.sqr()).invert();
        return { c0 * t3, -(c1 * t3) };
                 */
        todo!() // Implement the invert method
    }

    fn self_neg(&mut self) {
        self.c0.self_neg();
        self.c1.self_neg();
    }

    fn self_to_montgomery_form(&mut self) {
        self.c0.self_to_montgomery_form();
        self.c1.self_to_montgomery_form();
    }

    fn self_from_montgomery_form(&mut self) {
        self.c0.self_from_montgomery_form();
        self.c1.self_from_montgomery_form();
    }

    fn self_conditional_negate(&mut self, predicate: u64) {
        /*
         *this = predicate ? -(*this) : *this;
         */
        todo!() // Implement the self_conditional_negate method
    }

    fn reduce_once(&self) -> Self {
        self
        // return { c0.reduce_once(), c1.reduce_once() };
    }

    fn self_reduce_once(&mut self) {
        // c0.self_reduce_once();
        // c1.self_reduce_once();
    }

    fn self_set_msb(&mut self) {
        //     c0.data[3] = 0ULL | (1ULL << 63ULL);

        todo!() // Implement the self_set_msb method
    }

    fn is_msb_set(&self) -> bool {
        //    return (c0.data[3] >> 63ULL) == 1ULL;

        todo!() // Implement the is_msb_set method
    }

    fn is_msb_set_word(&self) -> u64 {
        //    return (c0.data[3] >> 63ULL);

        todo!() // Implement the is_msb_set_word method
    }

    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    fn frobenius_map(&self) -> Self {
        Field2 {
            c0: self.c0,
            c1: -self.c1,
            phantom: PhantomData,
        }
    }

    fn self_frobenius_map(&mut self) {
        self.c1.self_neg()
    }

    fn random_element(engine: Option<&mut dyn rand::RngCore>) -> Self {
        //     return { base::random_element(engine), base::random_element(engine) };

        todo!() // Implement the random_element logic
    }
}
