use crate::ecc::fields::field::{Field, FieldParams};

trait Bn254FrParams: FieldParams {
    // Note: limbs here are combined as concat(_3, _2, _1, _0)
    // E.g. this modulus forms the value:
    // 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
    // = 21888242871839275222246405745257275088548364400416034343698204186575808495617
    const modus_0: u64 = 0x43E1F593F0000001;
    const modus_1: u64 = 0x2833E84879B97091;
    const modus_2: u64 = 0xB85045B68181585D;
    const modus_3: u64 = 0x30644E72E131A029;

    const r_squared_0: u64 = 0x1BB8E645AE216DA7;
    const r_squared_1: u64 = 0x53FE3AB1E35C59E3;
    const r_squared_2: u64 = 0x8C49833D53BB8085;
    const r_squared_3: u64 = 0x216D0B17F4E44A5;

    const cube_root_0: u64 = 0x93e7cede4a0329b3;
    const cube_root_1: u64 = 0x7d4fdca77a96c167;
    const cube_root_2: u64 = 0x8be4ba08b19a750a;
    const cube_root_3: u64 = 0x1cbd5653a5661c25;

    const primitive_root_0: u64 = 0x636e735580d13d9c;
    const primitive_root_1: u64 = 0xa22bf3742445ffd6;
    const primitive_root_2: u64 = 0x56452ac01eb203d8;
    const primitive_root_3: u64 = 0x1860ef942963f9e7;

    const endo_g1_lo: u64 = 0x7a7bd9d4391eb18d;
    const endo_g1_mid: u64 = 0x4ccef014a773d2cf;
    const endo_g1_hi: u64 = 0x0000000000000002;
    const endo_g2_lo: u64 = 0xd91d232ec7e0b3d7;
    const endo_g2_mid: u64 = 0x0000000000000002;
    const endo_minus_b1_lo: u64 = 0x8211bbeb7d4f1128;
    const endo_minus_b1_mid: u64 = 0x6f4d8248eeb859fc;
    const endo_b2_lo: u64 = 0x89d3256894d213e3;
    const endo_b2_mid: u64 = 0;

    const r_inv: u64 = 0xc2e1f593efffffff;

    const coset_generators_0: [u64; 8] = [
        0x5eef048d8fffffe7,
        0xb8538a9dfffffe2,
        0x3057819e4fffffdb,
        0xdcedb5ba9fffffd6,
        0x8983e9d6efffffd1,
        0x361a1df33fffffcc,
        0xe2b0520f8fffffc7,
        0x8f46862bdfffffc2,
    ];
    const coset_generators_1: [u64; 8] = [
        0x12ee50ec1ce401d0,
        0x49eac781bc44cefa,
        0x307f6d866832bb01,
        0x677be41c0793882a,
        0x9e785ab1a6f45554,
        0xd574d1474655227e,
        0xc7147dce5b5efa7,
        0x436dbe728516bcd1,
    ];
    const coset_generators_2: [u64; 8] = [
        0x29312d5a5e5ee7,
        0x6697d49cd2d7a515,
        0x5c65ec9f484e3a89,
        0xc2d4900ec0c780b7,
        0x2943337e3940c6e5,
        0x8fb1d6edb1ba0d13,
        0xf6207a5d2a335342,
        0x5c8f1dcca2ac9970,
    ];
    const coset_generators_3: [u64; 8] = [
        0x463456c802275bed,
        0x543ece899c2f3b1c,
        0x180a96573d3d9f8,
        0xf8b21270ddbb927,
        0x1d9598e8a7e39857,
        0x2ba010aa41eb7786,
        0x39aa886bdbf356b5,
        0x47b5002d75fb35e5,
    ];
}

pub struct Bn254FrParamsImpl {}

impl Bn254FrParams for Bn254FrParamsImpl {}

pub type Fr = Field<Bn254FrParamsImpl>;
