use crate::ecc::fields::field6::{Field6Impl, Field6Params};

use super::{fq::Fq, fq2::Fq2};

pub trait Bn254Fq6Params: Field6Params<Fq, Fq2> {
    const frobenius_coeffs_c1_1: Fq2 = Fq2::new(
        0xb5773b104563ab30,
        0x347f91c8a9aa6454,
        0x7a007127242e0991,
        0x1956bcd8118214ec,
        0x6e849f1ea0aa4757,
        0xaa1c7b6d89f89141,
        0xb6e713cdfae0ca3a,
        0x26694fbb4e82ebc3,
    );

    const frobenius_coeffs_c1_2: Fq2 = Fq2::new(
        0x3350c88e13e80b9c,
        0x7dce557cdb5e56b9,
        0x6001b4b8b615564a,
        0x2682e617020217e0,
        0,
        0,
        0,
        0,
    );

    const frobenius_coeffs_c1_3: Fq2 = Fq2::new(
        0xc9af22f716ad6bad,
        0xb311782a4aa662b2,
        0x19eeaf64e248c7f4,
        0x20273e77e3439f82,
        0xacc02860f7ce93ac,
        0x3933d5817ba76b4c,
        0x69e6188b446c8467,
        0x0a46036d4417cc55,
    );

    const frobenius_coeffs_c2_1: Fq2 = Fq2::new(
        0x7361d77f843abe92,
        0xa5bb2bd3273411fb,
        0x9c941f314b3e2399,
        0x15df9cddbb9fd3ec,
        0x5dddfd154bd8c949,
        0x62cb29a5a4445b60,
        0x37bc870a0c7dd2b9,
        0x24830a9d3171f0fd,
    );

    const frobenius_coeffs_c2_2: Fq2 = Fq2::new(
        0x71930c11d782e155,
        0xa6bb947cffbe3323,
        0xaa303344d4741444,
        0x2c3b3f0d26594943,
        0,
        0,
        0,
        0,
    );

    const frobenius_coeffs_c2_3: Fq2 = Fq2::new(
        0x448a93a57b6762df,
        0xbfd62df528fdeadf,
        0xd858f5d00e9bd47a,
        0x06b03d4d3476ec58,
        0x2b19daf4bcc936d1,
        0xa1a54e7a56f4299f,
        0xb533eee05adeaef1,
        0x170c812b84dda0b2,
    );
}

struct Bn254Fq6ParamsImpl {}

impl Bn254Fq6Params for Bn254Fq6ParamsImpl {}

pub type Fq6 = Field6Impl<Fq, Fq2, Bn254Fq6ParamsImpl>;
