use crate::ecc::groups::affine_element::Affine;
use crate::ecc::groups::{Group, GroupParams};

use super::fq::{Bn254FqParamsImpl, Fq};
use super::fr::Bn254FrParamsImpl;

struct Bn254G1ParamsImpl {}

impl Bn254FqParamsImpl {
    const can_hash_to_curve: bool = true;
    const small_elements: bool = true;
}

impl GroupParams<Bn254FqParamsImpl, Bn254FrParamsImpl> for Bn254G1ParamsImpl {
    const USE_ENDOMORPHISM: bool = true;

    const has_a: bool = true;
    const one_x: Fq = Fq::one();
    const one_y: Fq = Fq::from_parts(
        0xa6ba871b8b1e1b3a,
        0x14f1d651eb8e167b,
        0xccdd46def0f28c58,
        0x1c14ef83340fbe5e,
    );
    const a: Fq = Fq::from_parts(0, 0, 0, 0);
    const b: Fq = Fq::from_parts(
        0x7a17caa950ad28d7,
        0x1f6ac17ae15521b9,
        0x334bea4e696bd284,
        0x2a1f6744ce179d8e,
    );
}

pub type G1 = Group<Bn254FqParamsImpl, Fq, Bn254FrParamsImpl, Bn254G1ParamsImpl>;
pub type G1Affine = Affine<Bn254FqParamsImpl, Fq, Bn254FrParamsImpl, Bn254G1ParamsImpl>;
