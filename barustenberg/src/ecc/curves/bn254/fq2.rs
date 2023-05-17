use crate::ecc::fields::{
    field::FieldParamsGeneral,
    field2::{Field2, Field2Params},
};

use super::fq::{Bn254FqParamsImpl, Fq};

#[derive(Default, PartialEq, Eq)]
pub struct Bn254Fq2ParamsImpl {}

impl FieldParamsGeneral for Bn254Fq2ParamsImpl {}

impl Field2Params<Bn254FqParamsImpl> for Bn254Fq2ParamsImpl {
    const twist_coeff_b_0: Fq = Fq::from_parts(
        0x3bf938e377b802a8,
        0x020b1b273633535d,
        0x26b7edf049755260,
        0x2514c6324384a86d,
    );

    const twist_coeff_b_1: Fq = Fq::from_parts(
        0x38e7ecccd1dcff67,
        0x65f0b37d93ce0d3e,
        0xd749d0dd22ac00aa,
        0x0141b9ce4a688d4d,
    );

    const twist_mul_by_q_x_0: Fq = Fq::from_parts(
        0xb5773b104563ab30,
        0x347f91c8a9aa6454,
        0x7a007127242e0991,
        0x1956bcd8118214ec,
    );

    const twist_mul_by_q_x_1: Fq = Fq::from_parts(
        0x6e849f1ea0aa4757,
        0xaa1c7b6d89f89141,
        0xb6e713cdfae0ca3a,
        0x26694fbb4e82ebc3,
    );

    const twist_mul_by_q_y_0: Fq = Fq::from_parts(
        0xe4bbdd0c2936b629,
        0xbb30f162e133bacb,
        0x31a9d1b6f9645366,
        0x253570bea500f8dd,
    );

    const twist_mul_by_q_y_1: Fq = Fq::from_parts(
        0xa1d77ce45ffe77c7,
        0x07affd117826d1db,
        0x6d16bd27bb7edc6b,
        0x2c87200285defecc,
    );

    const twist_cube_root_0: Fq = Fq::from_parts(
        0x505ecc6f0dff1ac2,
        0x2071416db35ec465,
        0xf2b53469fa43ea78,
        0x18545552044c99aa,
    );

    const twist_cube_root_1: Fq = Fq::from_parts(
        0xad607f911cfe17a8,
        0xb6bb78aa154154c4,
        0xb53dd351736b20db,
        0x1d8ed57c5cc33d41,
    );
}

pub type Fq2 = Field2<Bn254FqParamsImpl, Bn254Fq2ParamsImpl>;
