use crate::ecc::fields::field12::{Field12, Field12Params};

use super::{
    fq::{Bn254FqParamsImpl, Fq},
    fq2::{Bn254Fq2ParamsImpl, Fq2},
    fq6::Bn254Fq6ParamsImpl,
};

pub trait Bn254Fq12Params: Field12Params<Fq, Fq2> {
    const frobenius_coefficients_1: Fq2 = Fq2::new(
        0xaf9ba69633144907,
        0xca6b1d7387afb78a,
        0x11bded5ef08a2087,
        0x02f34d751a1f3a7c,
        0xa222ae234c492d72,
        0xd00f02a4565de15b,
        0xdc2ff3a253dfc926,
        0x10a75716b3899551,
    );

    const frobenius_coefficients_2: Fq2 = Fq2::new(
        0xca8d800500fa1bf2,
        0xf0c5d61468b39769,
        0x0e201271ad0d4418,
        0x04290f65bad856e6,
        0,
        0,
        0,
        0,
    );

    const frobenius_coefficients_3: Fq2 = Fq2::new(
        0x365316184e46d97d,
        0x0af7129ed4c96d9f,
        0x659da72fca1009b5,
        0x08116d8983a20d23,
        0xb1df4af7c39c1939,
        0x3d9f02878a73bf7f,
        0x9b2220928caf0ae0,
        0x26684515eff054a6,
    );
}

struct Bn254Fq12ParamsImpl {}

impl Bn254Fq12Params for Bn254Fq12ParamsImpl {}
pub type Fq12 =
    Field12<Bn254FqParamsImpl, Bn254Fq2ParamsImpl, Bn254Fq6ParamsImpl, Bn254Fq12ParamsImpl>;
