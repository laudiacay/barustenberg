use ark_bn254::{Fq, Fq2};
use ark_ff::{FftField, Field};

//pub(crate) mod grumpkin;
pub(crate) mod bn254_scalar_multiplication;
pub(crate) mod pairings;

const B_POINT_0: [u8; 32] = [
    0x3b, 0xf9, 0x38, 0xe3, 0x77, 0xb8, 0x02, 0xa8, 0x02, 0x0b, 0x1b, 0x27, 0x36, 0x33, 0x53, 0x5d,
    0x26, 0xb7, 0xed, 0xf0, 0x49, 0x75, 0x52, 0x60, 0x25, 0x14, 0xc6, 0x32, 0x43, 0x84, 0xa8, 0x6d,
];

const B_POINT_1: [u8; 32] = [
    0x38, 0xe7, 0xec, 0xcc, 0xd1, 0xdc, 0xff, 0x67, 0x65, 0xf0, 0xb3, 0x7d, 0x93, 0xce, 0x0d, 0x3e,
    0xd7, 0x49, 0xd0, 0xdd, 0x22, 0xac, 0x00, 0xaa, 0x01, 0x41, 0xb9, 0xce, 0x4a, 0x68, 0x8d, 0x4d,
];

pub trait Fq2Coeffs {
    fn twist_coeff_b() -> Self;
}

impl Fq2Coeffs for Fq2 {
    fn twist_coeff_b() -> Self {
        Self::new(
            Fq::from_random_bytes(&B_POINT_0[..]).unwrap(),
            Fq::from_random_bytes(&B_POINT_1[..]).unwrap(),
        )
    }
}

pub(crate) fn coset_generator<F: Field + FftField>(_idx: usize) -> F {
    /*
        ASSERT(idx < 7);
    const FieldExt result{
        Params::coset_generators_0[idx],
        Params::coset_generators_1[idx],
        Params::coset_generators_2[idx],
        Params::coset_generators_3[idx],
    };
    return result; */
    unimplemented!("coset_generator")
}

pub(crate) fn external_coset_generator<F: Field + FftField>() -> F {
    unimplemented!("external_coset_generator")
}
