use crate::ecc::groups::{affine_element::Affine, Group, GroupParams};

use super::{
    fq2::{Bn254Fq2ParamsImpl, Fq2},
    fr::Bn254FrParamsImpl,
};
struct Bn254G2ParamsImpl {}

impl Bn254G2ParamsImpl {
    const can_hash_to_curve: bool = false;
    const small_elements: bool = false;
}

impl GroupParams<Bn254Fq2ParamsImpl, Bn254FrParamsImpl> for Bn254G2ParamsImpl {
    const USE_ENDOMORPHISM: bool = false;
    const has_a: bool = false;

    const one_x: Fq2 = Fq2::from_parts(
        0x8e83b5d102bc2026,
        0xdceb1935497b0172,
        0xfbb8264797811adf,
        0x19573841af96503b,
        0xafb4737da84c6140,
        0x6043dd5a5802d8c4,
        0x09e950fc52a02f86,
        0x14fef0833aea7b6b,
    );
    const one_y: Fq2 = Fq2::from_parts(
        0x619dfa9d886be9f6,
        0xfe7fd297f59e9b78,
        0xff9e1a62231b7dfe,
        0x28fd7eebae9e4206,
        0x64095b56c71856ee,
        0xdc57f922327d3cbb,
        0x55f935be33351076,
        0x0da4a0e693fd6482,
    );
    const a: Fq2 = Fq2::zero();
    const b: Fq2 = Fq2::twist_coeff_b();
}

pub type G2 = Group<Bn254Fq2ParamsImpl, Fq2, Bn254FrParamsImpl, Bn254G2ParamsImpl>;
pub type G2Affine = Affine<Bn254Fq2ParamsImpl, Fq2, Bn254FrParamsImpl, Bn254G2ParamsImpl>;
