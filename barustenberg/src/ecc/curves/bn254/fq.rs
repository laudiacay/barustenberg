use crate::ecc::fields::field::{Field, FieldParams};

pub trait Bn254FqParams: FieldParams {
    const modulus_0: u64 = 0x3C208C16D87CFD47;
    const modulus_1: u64 = 0x97816a916871ca8d;
    const modulus_2: u64 = 0xb85045b68181585d;
    const modulus_3: u64 = 0x30644e72e131a029;

    const r_squared_0: u64 = 0xF32CFC5B538AFA89;
    const r_squared_1: u64 = 0xB5E71911D44501FB;
    const r_squared_2: u64 = 0x47AB1EFF0A417FF6;
    const r_squared_3: u64 = 0x06D89F71CAB8351F;

    const cube_root_0: u64 = 0x71930c11d782e155;
    const cube_root_1: u64 = 0xa6bb947cffbe3323;
    const cube_root_2: u64 = 0xaa303344d4741444;
    const cube_root_3: u64 = 0x2c3b3f0d26594943;

    const primitive_root_0: u64 = 0;
    const primitive_root_1: u64 = 0;
    const primitive_root_2: u64 = 0;
    const primitive_root_3: u64 = 0;

    const endo_g1_lo: u64 = 0x7a7bd9d4391eb18d;
    const endo_g1_mid: u64 = 0x4ccef014a773d2cf;
    const endo_g1_hi: u64 = 0x0000000000000002;
    const endo_g2_lo: u64 = 0xd91d232ec7e0b3d2;
    const endo_g2_mid: u64 = 0x0000000000000002;
    const endo_minus_b1_lo: u64 = 0x8211bbeb7d4f1129;
    const endo_minus_b1_mid: u64 = 0x6f4d8248eeb859fc;
    const endo_b2_lo: u64 = 0x89d3256894d213e2;
    const endo_b2_mid: u64 = 0;

    const r_inv: u64 = 0x87d20782e4866389;

    const coset_generators_0: [u64; 8] = [
        0x7a17caa950ad28d7,
        0x4d750e37163c3674,
        0x20d251c4dbcb4411,
        0xf42f9552a15a51ae,
        0x4f4bc0b2b5ef64bd,
        0x22a904407b7e725a,
        0xf60647ce410d7ff7,
        0xc9638b5c069c8d94,
    ];
    const coset_generators_1: [u64; 8] = [
        0x1f6ac17ae15521b9,
        0x29e3aca3d71c2cf7,
        0x345c97cccce33835,
        0x3ed582f5c2aa4372,
        0x1a4b98fbe78db996,
        0x24c48424dd54c4d4,
        0x2f3d6f4dd31bd011,
        0x39b65a76c8e2db4f,
    ];
    const coset_generators_2: [u64; 8] = [
        0x334bea4e696bd284,
        0x99ba8dbde1e518b0,
        0x29312d5a5e5edc,
        0x6697d49cd2d7a508,
        0x5c65ec9f484e3a79,
        0xc2d4900ec0c780a5,
        0x2943337e3940c6d1,
        0x8fb1d6edb1ba0cfd,
    ];
    const coset_generators_3: [u64; 8] = [
        0x2a1f6744ce179d8e,
        0x3829df06681f7cbd,
        0x463456c802275bed,
        0x543ece899c2f3b1c,
        0x180a96573d3d9f8,
        0xf8b21270ddbb927,
        0x1d9598e8a7e39857,
        0x2ba010aa41eb7786,
    ];
}

pub struct Bn254FqParamsImpl {}

impl Bn254FqParams for Bn254FqParamsImpl {}

pub type Fq = Field<Bn254FqParamsImpl>;
