use std::ops::{Add, Mul};

use super::{affine_element::AffineElement, GroupParams};

pub struct ElementImpl<Fq, Fr, Params> {
    pub x: Fq,
    pub y: Fq,
    pub z: Fq,
}

pub trait Element<Fq, Fr, Params: GroupParams<Fq>> {
    fn one() -> Self;

    fn zero() -> Self;

    fn random_element(rng: &mut impl rand::RngCore) -> Self;

    fn dbl(&self) -> Self;

    fn self_dbl(&mut self);

    fn self_mixed_add_or_sub(&mut self, other: &dyn AffineElement<Fq, Fr, Params>, predicate: u64);
    fn normalize(&self) -> Self;

    fn infinity() -> Self;

    fn set_infinity(&self) -> Self;

    fn self_set_infinity(&mut self);

    fn is_point_at_infinity(&self) -> bool;

    fn on_curve(&self) -> bool;

    fn batch_normalize(elements: &mut [Self]);

    fn batch_mul_with_endomorphism(
        points: &[dyn AffineElement<Fq, Fr, Params>],
        exponent: &Fr,
    ) -> Vec<dyn AffineElement<Fq, Fr, Params>>;

    fn mul_without_endomorphism(&self, exponent: &Fr) -> dyn AffineElement<Fq, Fr, Params>;

    fn mul_with_endomorphism(&self, exponent: &Fr) -> dyn AffineElement<Fq, Fr, Params>;
}

impl<Fq, Fr, Params> Element<Fq, Fr, Params> for ElementImpl<Fq, Fr, Params> {
    fn one() -> Self {
        ElementImpl {
            x: Params::one_x,
            y: Params::one_y,
            z: Fq::one(),
        }
    }

    fn zero() -> Self {
        let mut zero = ElementImpl {
            x: Fq::zero(),
            y: Fq::zero(),
            z: Fq::zero(),
        };
        zero.self_set_infinity();
        zero
    }

    fn random_element(rng: &mut impl rand::RngCore) -> Self {
        // Implement random_element logic
    }

    fn dbl(&self) -> Self {
        // Implement dbl logic
    }

    fn self_dbl(&mut self) {
        // Implement self_dbl logic
    }

    fn self_mixed_add_or_sub(&mut self, other: &dyn AffineElement<Fq, Fr, Params>, predicate: u64) {
        // Implement self_mixed_add_or_sub logic
    }

    // Implement other methods

    fn normalize(&self) -> Self {
        // Implement normalize logic
    }

    fn infinity() -> Self {
        let mut infinity = ElementImpl {
            x: Fq::zero(),
            y: Fq::one(),
            z: Fq::zero(),
        };
        infinity.self_set_infinity();
        infinity
    }

    fn set_infinity(&self) -> Self {
        todo!("set_infinity")
        // Implement set_infinity logic
    }

    fn self_set_infinity(&mut self) {
        todo!("self_set_infinity")
        // Implement self_set_infinity logic
    }

    fn is_point_at_infinity(&self) -> bool {
        todo!("is_point_at_infinity")
        // Implement is_point_at_infinity logic
    }

    fn on_curve(&self) -> bool {
        // Implement on_curve logic
        todo!("on_curve")
    }

    fn batch_normalize(elements: &mut [Self]) {
        // Implement batch_normalize logic
        todo!("fix batch_normalize")
    }

    fn batch_mul_with_endomorphism(
        points: &[dyn AffineElement<Fq, Fr, Params>],
        exponent: &Fr,
    ) -> Vec<dyn AffineElement<Fq, Fr, Params>> {
        // Implement batch_mul_with_endomorphism logic
    }

    fn mul_without_endomorphism(&self, exponent: &Fr) -> dyn AffineElement<Fq, Fr, Params> {
        // Implement mul_without_endomorphism logic
    }

    fn mul_with_endomorphism(&self, exponent: &Fr) -> dyn AffineElement<Fq, Fr, Params> {
        // Implement mul_with_endomorphism logic
    }
}

impl<Fq, Fr, Params> Add for dyn Element<Fq, Fr, Params> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Implement add logic
        todo!()
    }
}

// Implement other operator traits for Element
impl<Fq, Fr, Params> Mul<Fr> for dyn Element<Fq, Fr, Params> {
    type Output = Self;

    fn mul(self, other: Fr) -> Self {
        // Implement mul logic
    }
}
