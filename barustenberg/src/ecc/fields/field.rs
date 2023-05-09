use std::{
    cmp::Ordering,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use primitive_types::U256;
use serde::{Deserialize, Serialize};

#[derive(Copy, Serialize, Deserialize)]
pub struct Field<Params> {
    data: [u64; 4],
}

pub struct WideArray {
    pub data: [u64; 8],
}

pub struct WnafTable {
    windows: [u8; 64],
}

impl WnafTable {
    pub const fn new(target: &U256) -> WnafTable {
        WnafTable {
            windows: [
                (target.data[0] & 15) as u8,
                ((target.data[0] >> 4) & 15) as u8,
                ((target.data[0] >> 8) & 15) as u8,
                ((target.data[0] >> 12) & 15) as u8,
                ((target.data[0] >> 16) & 15) as u8,
                ((target.data[0] >> 20) & 15) as u8,
                ((target.data[0] >> 24) & 15) as u8,
                ((target.data[0] >> 28) & 15) as u8,
                ((target.data[0] >> 32) & 15) as u8,
                ((target.data[0] >> 36) & 15) as u8,
                ((target.data[0] >> 40) & 15) as u8,
                ((target.data[0] >> 44) & 15) as u8,
                ((target.data[0] >> 48) & 15) as u8,
                ((target.data[0] >> 52) & 15) as u8,
                ((target.data[0] >> 56) & 15) as u8,
                ((target.data[0] >> 60) & 15) as u8,
                (target.data[1] & 15) as u8,
                ((target.data[1] >> 4) & 15) as u8,
                ((target.data[1] >> 8) & 15) as u8,
                ((target.data[1] >> 12) & 15) as u8,
                ((target.data[1] >> 16) & 15) as u8,
                ((target.data[1] >> 20) & 15) as u8,
                ((target.data[1] >> 24) & 15) as u8,
                ((target.data[1] >> 28) & 15) as u8,
                ((target.data[1] >> 32) & 15) as u8,
                ((target.data[1] >> 36) & 15) as u8,
                ((target.data[1] >> 40) & 15) as u8,
                ((target.data[1] >> 44) & 15) as u8,
                ((target.data[1] >> 48) & 15) as u8,
                ((target.data[1] >> 52) & 15) as u8,
                ((target.data[1] >> 56) & 15) as u8,
                ((target.data[1] >> 60) & 15) as u8,
                (target.data[2] & 15) as u8,
                ((target.data[2] >> 4) & 15) as u8,
                ((target.data[2] >> 8) & 15) as u8,
                ((target.data[2] >> 12) & 15) as u8,
                ((target.data[2] >> 16) & 15) as u8,
                ((target.data[2] >> 20) & 15) as u8,
                ((target.data[2] >> 24) & 15) as u8,
                ((target.data[2] >> 28) & 15) as u8,
                ((target.data[2] >> 32) & 15) as u8,
                ((target.data[2] >> 36) & 15) as u8,
                ((target.data[2] >> 40) & 15) as u8,
                ((target.data[2] >> 44) & 15) as u8,
                ((target.data[2] >> 48) & 15) as u8,
                ((target.data[2] >> 52) & 15) as u8,
                ((target.data[2] >> 56) & 15) as u8,
                ((target.data[2] >> 60) & 15) as u8,
                (target.data[3] & 15) as u8,
                ((target.data[3] >> 4) & 15) as u8,
                ((target.data[3] >> 8) & 15) as u8,
                ((target.data[3] >> 12) & 15) as u8,
                ((target.data[3] >> 16) & 15) as u8,
                ((target.data[3] >> 20) & 15) as u8,
                ((target.data[3] >> 24) & 15) as u8,
                ((target.data[3] >> 28) & 15) as u8,
                ((target.data[3] >> 32) & 15) as u8,
                ((target.data[3] >> 36) & 15) as u8,
                ((target.data[3] >> 40) & 15) as u8,
                ((target.data[3] >> 44) & 15) as u8,
                ((target.data[3] >> 48) & 15) as u8,
                ((target.data[3] >> 52) & 15) as u8,
                ((target.data[3] >> 56) & 15) as u8,
                ((target.data[3] >> 60) & 15) as u8,
            ],
        }
    }
}

pub const fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    todo!()
}

pub const fn mac(a: u64, b: u64, c: u64, carry_in: u64, carry_out: &mut u64) -> u64 {
    todo!("implement mac")
}

pub const fn mac_discard_lo(a: u64, b: u64, c: u64) -> u64 {
    todo!()
}

pub const fn addc(a: u64, b: u64, carry_in: u64) -> u64 {
    todo!()
}

pub const fn sbb(a: u64, b: u64, borrow_in: u64, borrow_out: &mut u64) -> u64 {
    todo!("implement sbb")
}

pub fn square_accumulate(
    a: u64,
    b: u64,
    c: u64,
    carry_in_lo: u64,
    carry_in_hi: u64,
    carry_lo: &mut u64,
    carry_hi: &mut u64,
) -> u64 {
    todo!()
}

impl<Params> Field<Params> {
    #[cfg(all(features = "SIZEOFINT128", not(features = "WASM")))]
    const LO_MASK: u128 = 0xffffffffffffffff;

    pub fn new() -> Self {
        Field { data: [0, 0, 0, 0] }
    }

    pub fn from_u256(input: U256) -> Self {
        let mut result = Field {
            data: [
                input.bits(0..64) as u64,
                input.bits(64..128) as u64,
                input.bits(128..192) as u64,
                input.bits(192..256) as u64,
            ],
        };
        result.self_to_montgomery_form();
        result
    }

    pub fn from_u64(input: u64) -> Self {
        let mut result = Field {
            data: [input, 0, 0, 0],
        };
        result.self_to_montgomery_form();
        result
    }

    pub fn from_i64(input: i64) -> Self {
        let mut result = Field { data: [0, 0, 0, 0] };
        if input < 0 {
            result.data[0] = (-input) as u64;
            result.data[1] = 0;
            result.data[2] = 0;
            result.data[3] = 0;
            result.self_to_montgomery_form();
            result.self_neg();
            result.self_reduce_once();
        } else {
            result.data[0] = input as u64;
            result.data[1] = 0;
            result.data[2] = 0;
            result.data[3] = 0;
            result.self_to_montgomery_form();
        }
        result
    }

    pub fn from_parts(a: u64, b: u64, c: u64, d: u64) -> Self {
        Field { data: [a, b, c, d] }
    }

    pub fn as_u32(&self) -> u32 {
        let out = self.from_montgomery_form();
        out.data[0] as u32
    }

    pub fn as_u64(&self) -> u64 {
        let out = self.from_montgomery_form();
        out.data[0]
    }

    pub fn as_u128(&self) -> u128 {
        let out = self.from_montgomery_form();
        let lo = out.data[0] as u128;
        let hi = out.data[1] as u128;
        (hi << 64) | lo
    }

    pub fn as_u256(&self) -> U256 {
        let out = self.from_montgomery_form();
        U256::from_parts(
            out.data[0] as u64,
            out.data[1] as u64,
            out.data[2] as u64,
            out.data[3] as u64,
        )
    }

    pub const fn uint256_t_no_montgomery_conversion(&self) -> U256 {
        U256::from_parts(self.data[0], self.data[1], self.data[2], self.data[3])
    }

    pub const MODULUS: U256 = U256::from_parts(
        Params::MODULUS_0,
        Params::MODULUS_1,
        Params::MODULUS_2,
        Params::MODULUS_3,
    );

    pub const fn cube_root_of_unity() -> Self {
        if Params::CUBE_ROOT_0 != 0 {
            Self::from_parts(
                Params::CUBE_ROOT_0,
                Params::CUBE_ROOT_1,
                Params::CUBE_ROOT_2,
                Params::CUBE_ROOT_3,
            )
        } else {
            let two_inv = Field::from_i64(2).invert();
            let numerator = -Field::from_i64(3).sqrt() - Field::from_i64(1);
            let result = two_inv * numerator;
            result
        }
    }

    pub const fn zero() -> Self {
        Self::from_parts(0, 0, 0, 0)
    }

    pub const fn neg_one() -> Self {
        -Self::from_i64(1)
    }

    pub const fn one() -> Self {
        Self::from_i64(1)
    }

    pub fn external_coset_generator() -> Self {
        Self::from_parts(
            Params::COSET_GENERATORS_0[7],
            Params::COSET_GENERATORS_1[7],
            Params::COSET_GENERATORS_2[7],
            Params::COSET_GENERATORS_3[7],
        )
    }

    pub fn tag_coset_generator() -> Self {
        Self::from_parts(
            Params::COSET_GENERATORS_0[6],
            Params::COSET_GENERATORS_1[6],
            Params::COSET_GENERATORS_2[6],
            Params::COSET_GENERATORS_3[6],
        )
    }

    pub fn coset_generator(idx: usize) -> Self {
        assert!(idx < 7);
        Self::from_parts(
            Params::COSET_GENERATORS_0[idx],
            Params::COSET_GENERATORS_1[idx],
            Params::COSET_GENERATORS_2[idx],
            Params::COSET_GENERATORS_3[idx],
        )
    }

    pub fn to_montgomery_form(&self) -> Self {
        todo!("this might ought to be an into or a from, btw")
    }

    pub fn from_montgomery_form(&self) -> Self {
        todo!("this might ought to be an into or a from, btw")
    }

    pub fn sqr(&self) -> Self {
        todo!("implement sqr")
    }

    pub fn sqr_inplace(&mut self) {
        todo!("implement sqr_inplace")
    }

    pub fn pow(&self, exp: &U256) -> Self {
        todo!("implement pow")
    }

    pub fn pow_64(&self, exp: u64) -> Self {
        todo!("implement pow_64")
    }
    // TODO BUG is this little_endian rep correct?
    pub const modulus_minus_two: U256 = U256::from_little_endian(&[
        Params::modulus_0 - 2u64,
        Params::modulus_1,
        Params::modulus_2,
        Params::modulus_3,
    ]);

    pub const fn invert(&self) -> Self {
        // TODO: Implement the inversion logic
    }

    pub fn batch_invert(coeffs: &mut [Field<Params>]) {
        // TODO: Implement the batch inversion logic
    }

    pub fn batch_invert_slice(coeffs: &mut [Field<Params>], n: usize) {
        // TODO: Implement the batch inversion logic for a slice
    }

    pub const fn sqrt(&self) -> (bool, Self) {
        // TODO: Implement the square root logic
    }

    pub const fn self_neg(&mut self) {
        // TODO: Implement the self negation logic
    }

    pub const fn self_to_montgomery_form(&mut self) {
        // TODO: Implement the conversion to Montgomery form logic for self
    }

    pub const fn self_from_montgomery_form(&mut self) {
        // TODO: Implement the conversion from Montgomery form logic for self
    }

    pub const fn self_conditional_negate(&mut self, predicate: u64) {
        // TODO: Implement the self conditional negate logic
    }

    pub const fn reduce_once(&self) -> Self {
        // TODO: Implement the reduce once logic
    }

    pub const fn self_reduce_once(&mut self) {
        // TODO: Implement the self reduce once logic
    }

    pub const fn self_set_msb(&mut self) {
        // TODO: Implement the set msb logic for self
    }

    pub const fn is_msb_set(&self) -> bool {
        // TODO: Implement the is msb set logic
    }

    pub const fn is_msb_set_word(&self) -> u64 {
        // TODO: Implement the is msb set word logic
    }

    pub const fn is_zero(&self) -> bool {
        // TODO: Implement the is zero logic
    }

    pub const fn get_root_of_unity(degree: usize) -> Self {
        todo!() // Implement the get_root_of_unity logic
    }

    pub fn serialize_to_buffer(value: &Self, buffer: &mut [u8]) {
        todo!() // Implement the serialize_to_buffer logic
    }

    pub fn serialize_from_buffer(buffer: &[u8]) -> Self {
        todo!() // Implement the serialize_from_buffer logic
    }

    pub fn to_buffer(&self) -> Vec<u8> {
        todo!() // Implement the to_buffer logic
    }

    pub const fn mul_512(&self, other: &Self) -> WideArray {
        todo!() // Implement the mul_512 logic
    }

    pub const fn sqr_512(&self) -> WideArray {
        todo!() // Implement the sqr_512 logic
    }

    pub const fn conditionally_subtract_from_double_modulus(&self, predicate: u64) -> Self {
        if predicate != 0 {
            todo!() // Implement the subtraction when predicate is non-zero
        } else {
            *self
        }
    }

    /**
     * For short Weierstrass curves y^2 = x^3 + b mod r, if there exists a cube root of unity mod r,
     * we can take advantage of an enodmorphism to decompose a 254 bit scalar into 2 128 bit scalars.
     * \beta = cube root of 1, mod q (q = order of fq)
     * \lambda = cube root of 1, mod r (r = order of fr)
     *
     * For a point P1 = (X, Y), where Y^2 = X^3 + b, we know that
     * the point P2 = (X * \beta, Y) is also a point on the curve
     * We can represent P2 as a scalar multiplication of P1, where P2 = \lambda * P1
     *
     * For a generic multiplication of P1 by a 254 bit scalar k, we can decompose k
     * into 2 127 bit scalars (k1, k2), such that k = k1 - (k2 * \lambda)
     *
     * We can now represent (k * P1) as (k1 * P1) - (k2 * P2), where P2 = (X * \beta, Y).
     * As k1, k2 have half the bit length of k, we have reduced the number of loop iterations of our
     * scalar multiplication algorithm in half
     *
     * To find k1, k2, We use the extended euclidean algorithm to find 4 short scalars [a1, a2], [b1, b2] such that
     * modulus = (a1 * b2) - (b1 * a2)
     * We then compute scalars c1 = round(b2 * k / r), c2 = round(b1 * k / r), where
     * k1 = (c1 * a1) + (c2 * a2), k2 = -((c1 * b1) + (c2 * b2))
     * We pre-compute scalars g1 = (2^256 * b1) / n, g2 = (2^256 * b2) / n, to avoid having to perform long division
     * on 512-bit scalars
     **/
    fn split_into_endomorphism_scalars(k: &Self, k1: &Self, k2: &Self) {
        //  // if the modulus is a 256-bit integer, we need to use a basis where g1, g2 have been shifted by 2^384
        //  if constexpr (Params::modulus_3 >= 0x4000000000000000ULL) {
        //      split_into_endomorphism_scalars_384(k, k1, k2);
        //      return;
        //  }
        //  field input = k.reduce_once();
        //  // uint64_t lambda_reduction[4] = { 0 };
        //  // __to_montgomery_form(lambda, lambda_reduction);

        //  constexpr field endo_g1 = { Params::endo_g1_lo, Params::endo_g1_mid, Params::endo_g1_hi, 0 };

        //  constexpr field endo_g2 = { Params::endo_g2_lo, Params::endo_g2_mid, 0, 0 };

        //  constexpr field endo_minus_b1 = { Params::endo_minus_b1_lo, Params::endo_minus_b1_mid, 0, 0 };

        //  constexpr field endo_b2 = { Params::endo_b2_lo, Params::endo_b2_mid, 0, 0 };

        //  // compute c1 = (g2 * k) >> 256
        //  wide_array c1 = endo_g2.mul_512(input);
        //  // compute c2 = (g1 * k) >> 256
        //  wide_array c2 = endo_g1.mul_512(input);

        //  // (the bit shifts are implicit, as we only utilize the high limbs of c1, c2

        //  field c1_hi = {
        //      c1.data[4], c1.data[5], c1.data[6], c1.data[7]
        //  }; // *(field*)((uintptr_t)(&c1) + (4 * sizeof(uint64_t)));
        //  field c2_hi = {
        //      c2.data[4], c2.data[5], c2.data[6], c2.data[7]
        //  }; // *(field*)((uintptr_t)(&c2) + (4 * sizeof(uint64_t)));

        //  // compute q1 = c1 * -b1
        //  wide_array q1 = c1_hi.mul_512(endo_minus_b1);
        //  // compute q2 = c2 * b2
        //  wide_array q2 = c2_hi.mul_512(endo_b2);

        //  // FIX: Avoid using 512-bit multiplication as its not necessary.
        //  // c1_hi, c2_hi can be uint256_t's and the final result (without montgomery reduction)
        //  // could be casted to a field.
        //  field q1_lo{ q1.data[0], q1.data[1], q1.data[2], q1.data[3] };
        //  field q2_lo{ q2.data[0], q2.data[1], q2.data[2], q2.data[3] };

        //  field t1 = (q2_lo - q1_lo).reduce_once();
        //  field beta = cube_root_of_unity();
        //  field t2 = (t1 * beta + input).reduce_once();
        //  k2.data[0] = t1.data[0];
        //  k2.data[1] = t1.data[1];
        //  k1.data[0] = t2.data[0];
        //  k1.data[1] = t2.data[1];a
        todo!("split_into_endomorphism_scalars")
    }

    fn split_into_endomorphism_scalars_384(input: &Self, k1_out: &Self, k2_out: &Self) {
        // constexpr field minus_b1f{
        //     Params::endo_minus_b1_lo,
        //     Params::endo_minus_b1_mid,
        //     0,
        //     0,
        // };
        // constexpr field b2f{
        //     Params::endo_b2_lo,
        //     Params::endo_b2_mid,
        //     0,
        //     0,
        // };
        // constexpr uint256_t g1{
        //     Params::endo_g1_lo,
        //     Params::endo_g1_mid,
        //     Params::endo_g1_hi,
        //     Params::endo_g1_hihi,
        // };
        // constexpr uint256_t g2{
        //     Params::endo_g2_lo,
        //     Params::endo_g2_mid,
        //     Params::endo_g2_hi,
        //     Params::endo_g2_hihi,
        // };

        // field kf = input.reduce_once();
        // uint256_t k{ kf.data[0], kf.data[1], kf.data[2], kf.data[3] };

        // uint512_t c1 = (uint512_t(k) * uint512_t(g1)) >> 384;
        // uint512_t c2 = (uint512_t(k) * uint512_t(g2)) >> 384;

        // field c1f{ c1.lo.data[0], c1.lo.data[1], c1.lo.data[2], c1.lo.data[3] };
        // field c2f{ c2.lo.data[0], c2.lo.data[1], c2.lo.data[2], c2.lo.data[3] };

        // c1f.self_to_montgomery_form();
        // c2f.self_to_montgomery_form();
        // c1f = c1f * minus_b1f;
        // c2f = c2f * b2f;
        // field r2f = c1f - c2f;
        // field beta = cube_root_of_unity();
        // field r1f = input.reduce_once() - r2f * beta;
        // k1_out = r1f;
        // k2_out = -r2f;
        todo!("split_into_endomorphism_scalars_384")
    }

    pub fn random_element(engine: Option<&mut rand::Rng>) -> Self {
        todo!() // Implement the random_element logic
    }

    pub const fn multiplicative_generator() -> Self {
        todo!() // Implement the multiplicative_generator logic
    }

    const fn twice_modulus() -> U256 {
        Self::modulus() + Self::modulus()
    }

    const fn not_modulus() -> U256 {
        -Self::modulus()
    }

    const fn twice_not_modulus() -> U256 {
        -(Self::twice_modulus())
    }

    const fn reduce(&self) -> Self {
        todo!()
    }

    const fn add(&self, other: &Self) -> Self {
        todo!()
    }

    const fn subtract(&self, other: &Self) -> Self {
        todo!()
    }

    const fn subtract_coarse(&self, other: &Self) -> Self {
        todo!()
    }

    const fn montgomery_mul(&self, other: &Self) -> Self {
        todo!()
    }

    const fn montgomery_mul_big(&self, other: &Self) -> Self {
        todo!()
    }

    const fn montgomery_square(&self) -> Self {
        todo!()
    }

    const COSET_GENERATOR_SIZE: usize = 15;
    const fn tonelli_shanks_sqrt(&self) -> Self {
        todo!()
    }

    // ... Implement the required methods like self_to_montgomery_form, from_montgomery_form, self_neg, self_reduce_once
}

//TODO handle BBERG_INLINE as macros

#[cfg(not(features = "BBERG_NO_ASM"))]
impl<Params: FieldParameters> Field<Params> {
    const ZERO_REFERENCE: u64 = 0x00;
    fn asm_mul(a: &Self, b: &Self) -> Self {
        todo!()
    }
    fn asm_sqr(a: &Self) -> Self {
        todo!()
    }
    fn asm_add(a: &Self, b: &Self) -> Self {
        todo!()
    }
    fn asm_sub(a: &Self, b: &Self) -> Self {
        todo!()
    }
    fn asm_mul_with_coarse_reduction(a: &Self, b: &Self) -> Self {
        todo!()
    }
    fn asm_sqr_with_coarse_reduction(a: &Self) -> Self {
        todo!()
    }
    fn asm_add_with_coarse_reduction(a: &Self, b: &Self) -> Self {
        todo!()
    }
    fn asm_sub_with_coarse_reduction(a: &Self, b: &Self) -> Self {
        todo!()
    }
    fn asm_add_without_reduction(a: &Self, b: &Self) -> Self {
        todo!()
    }
    fn asm_self_sqr(a: &Self) {
        todo!()
    }
    fn asm_self_add(a: &Self, b: &Self) {
        todo!()
    }
    fn asm_self_sub(a: &Self, b: &Self) {
        todo!()
    }
    fn asm_self_mul_with_coarse_reduction(a: &Self, b: &Self) {
        todo!()
    }
    fn asm_self_sqr_with_coarse_reduction(a: &Self) {
        todo!()
    }
    fn asm_self_add_with_coarse_reduction(a: &Self, b: &Self) {
        todo!()
    }
    fn asm_self_sub_with_coarse_reduction(a: &Self, b: &Self) {
        todo!()
    }
    fn asm_self_add_without_reduction(a: &Self, b: &Self) {
        todo!()
    }
    fn asm_conditional_negate(a: &Self, predicate: u64) {
        todo!()
    }
    fn asm_reduce_once(a: &Self) -> Self {
        todo!()
    }
    fn asm_self_reduce_once(a: &Self) {
        todo!()
    }
}

impl<Params: FieldParameters> fmt::Display for Field<Params> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let out = self.from_montgomery_form();
        write!(
            f,
            "{:#018x}{:018x}{:018x}{:018x}",
            out.data[3], out.data[2], out.data[1], out.data[0]
        )
    }
}

impl<Params> Mul for Field<Params> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        todo!();
    }
}

impl<Params> Add for Field<Params> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        todo!();
    }
}

impl<Params> Sub for Field<Params> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        todo!();
    }
}

impl<Params> Neg for Field<Params> {
    type Output = Self;

    fn neg(self) -> Self {
        todo!();
    }
}

impl<Params> Div for Field<Params> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        todo!();
    }
}

impl<Params> AddAssign for Field<Params> {
    fn add_assign(&mut self, rhs: Self) {
        todo!();
    }
}

impl<Params> SubAssign for Field<Params> {
    fn sub_assign(&mut self, rhs: Self) {
        todo!();
    }
}

impl<Params> MulAssign for Field<Params> {
    fn mul_assign(&mut self, rhs: Self) {
        todo!();
    }
}

impl<Params> DivAssign for Field<Params> {
    fn div_assign(&mut self, rhs: Self) {
        todo!();
    }
}

impl<Params> PartialEq for Field<Params> {
    fn eq(&self, other: &Self) -> bool {
        todo!();
    }
}

impl<Params> Eq for Field<Params> {}

impl<Params> PartialOrd for Field<Params> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        todo!();
    }
}

impl<Params> Ord for Field<Params> {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}
