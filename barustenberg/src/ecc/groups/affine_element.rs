use std::cmp::{PartialEq, PartialOrd};
use std::default::Default;
use std::vec;

use primitive_types::U256;

use super::element::Element;
use super::GroupParams;

pub trait AffineElement<Fq, Fr, Params: GroupParams> {
    fn new(a: Fq, b: Fq) -> Self;

    fn one() -> Self;

    fn from_compressed(compressed: U256) -> Self
    where
        Fq: 'static;

    fn from_compressed_unsafe(compressed: U256) -> [Self; 2]
    where
        Fq: 'static;

    fn is_point_at_infinity(&self) -> bool;

    fn on_curve(&self) -> bool;

    fn compress(&self) -> U256;

    fn serialize_to_buffer(&self, buffer: &mut [u8]);

    fn serialize_from_buffer(buffer: &[u8]) -> Self;

    fn to_buffer(&self) -> Vec<u8>;
}

impl<Fq, Fr, Params> PartialEq for dyn AffineElement<Fq, Fr, Params>
where
    Fq: Default + Clone + PartialEq,
    Fr: Default + Clone,
    Params: Default + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

pub struct AffineElementImpl<Fq, Fr, Params>
where
    Fq: Default + Clone + PartialEq,
    Fr: Default + Clone,
    Params: Default + Clone,
{
    x: Fq,
    y: Fq,
}

impl<Fq, Fr, Params> Default for AffineElementImpl<Fq, Fr, Params>
where
    Fq: Default + Clone + PartialEq,
    Fr: Default + Clone,
    Params: Default + Clone,
{
    fn default() -> Self {
        Self {
            x: Fq::default(),
            y: Fq::default(),
        }
    }
}

impl<Fq, Fr, Params> AffineElement<Fq, Fr, Params> for AffineElementImpl<Fq, Fr, Params>
where
    Fq: Default + Clone + PartialEq,
    Fr: Default + Clone,
    Params: Default + Clone,
{
    fn new(a: Fq, b: Fq) -> Self {
        Self { x: a, y: b }
    }

    fn one() -> Self {
        Self {
            x: Params::one_x(),
            y: Params::one_y(),
        }
    }

    fn from_compressed(compressed: U256) -> Self
    where
        Fq: 'static,
    {
        // Implementation for (BaseField::modulus >> 255) == 0
        let _compile_time_enabled: std::marker::PhantomData<()>;
        // TODO: Implement logic for this case
        unimplemented!();
    }

    fn from_compressed_unsafe(compressed: U256) -> [Self; 2]
    where
        Fq: 'static,
    {
        // Implementation for (BaseField::modulus >> 255) == 1
        let _compile_time_enabled: std::marker::PhantomData<()>;
        // TODO: Implement logic for this case
        unimplemented!();
    }

    fn is_point_at_infinity(&self) -> bool {
        // Implement logic to check if point is at infinity
        // Return true if point is at infinity, false otherwise
        unimplemented!();
    }

    fn on_curve(&self) -> bool {
        // Implement logic to check if point is on the curve
        // Return true if point is on the curve, false otherwise
        unimplemented!();
    }

    fn compress(&self) -> U256 {
        // Implement logic to compress the point
        // Return the compressed point as UInt256
        unimplemented!();
    }

    fn serialize_to_buffer(&self, buffer: &mut [u8]) {
        unimplemented!("serialize_to_buffer");
    }

    fn serialize_from_buffer(buffer: &[u8]) -> Self {
        unimplemented!("serialize_from_buffer")
    }

    fn to_buffer(&self) -> Vec<u8> {
        let mut buffer = vec![0u8; 64];
        self.serialize_to_buffer(&mut buffer);
        buffer
    }
}

impl<Fq, Fr, Params: GroupParams> PartialEq for AffineElement<Fq, Fr, Params>
where
    Fq: Default + Clone + PartialEq,
    Fr: Default + Clone,
    Params: Default + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl<Fq, Fr, Params: GroupParams> PartialOrd for AffineElement<Fq, Fr, Params>
where
    Fq: Default + Clone + PartialEq + PartialOrd,
    Fr: Default + Clone,
    Params: Default + Clone,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else if self.on_curve() && !other.on_curve() {
            Some(std::cmp::Ordering::Less)
        } else if !self.on_curve() && other.on_curve() {
            Some(std::cmp::Ordering::Greater)
        } else if self.x != other.x {
            self.x.partial_cmp(&other.x)
        } else {
            self.y.partial_cmp(&other.y)
        }
    }
}

impl<Fq, Fr, Params> From<Element<Fq, Fr, Params>> for AffineElement<Fq, Fr, Params> {
    fn from(element: Element<Fq, Fr, Params>) -> Self {
        // Implement From trait
    }
}
