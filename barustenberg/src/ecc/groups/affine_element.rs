use std::cmp::{PartialEq, PartialOrd};
use std::default::Default;
use std::vec;

use primitive_types::U256;

use crate::ecc::fields::field::{Field, FieldGeneral, FieldParams, FieldParamsGeneral};

use super::GroupParams;

#[derive(Default, PartialEq, Eq)]
pub struct Affine<FqP, Fq: FieldGeneral<FqP>, FrP, Params>
where
    FqP: FieldParamsGeneral,
    FrP: FieldParams,
    Params: GroupParams<FqP, FrP>,
{
    x: Fq,
    y: Fq,
    phantom: std::marker::PhantomData<(FqP, FrP, Params)>,
}

impl<
        FqP: FieldParamsGeneral,
        Fq: FieldGeneral<FqP>,
        FrP: FieldParams,
        Params: GroupParams<FqP, FrP>,
    > Affine<FqP, Fq, FrP, Params>
{
    fn new(a: dyn FieldGeneral<FqP>, b: Field<FrP>) -> Self {
        Self {
            x: a,
            y: b,
            phantom: std::marker::PhantomData,
        }
    }

    fn one() -> Self {
        Self {
            x: Params::one_x(),
            y: Params::one_y(),
            phantom: std::marker::PhantomData,
        }
    }

    fn from_compressed(compressed: U256) -> Self {
        // Implementation for (BaseField::modulus >> 255) == 0
        let _compile_time_enabled: std::marker::PhantomData<()>;
        // TODO: Implement logic for this case
        unimplemented!();
    }

    fn from_compressed_unsafe(compressed: U256) -> [Self; 2]
// where
    //     Fq: 'static,
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

impl<FqP, Fq, FrP, Params: GroupParams<FqP, FrP>> PartialOrd for Affine<FqP, Fq, FrP, Params>
where
    FqP: FieldParams + PartialEq,
    Fq: FieldGeneral<FqP> + PartialOrd + PartialEq,
    FrP: FieldParams + PartialEq,
    Params: PartialEq,
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
