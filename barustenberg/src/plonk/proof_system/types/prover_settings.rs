pub trait ProverSettings {}

pub trait SettingsBase: ProverSettings {
    fn requires_shifted_wire(wire_shift_settings: u64, wire_index: u64) -> bool {
        ((wire_shift_settings >> wire_index) & 1u64) == 1u64
    }
}

pub trait StandardSettings: SettingsBase {
    type Arithmetization: arithmetization::Standard;
    const NUM_CHALLENGE_BYTES: usize = 16;
    const HASH_TYPE: transcript::HashType = transcript::HashType::PedersenBlake3s;
    const PROGRAM_WIDTH: usize = 3;
    const NUM_SHIFTED_WIRE_EVALUATIONS: usize = 1;
    const WIRE_SHIFT_SETTINGS: u64 = 0b0100;
    const PERMUTATION_SHIFT: u32 = 30;
    const PERMUTATION_MASK: u32 = 0xC0000000;
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize = 4;
    const IS_PLOOKUP: bool = false;
}

pub trait TurboSettings: SettingsBase {
    const NUM_CHALLENGE_BYTES: usize = 16;
    const HASH_TYPE: transcript::HashType = transcript::HashType::PedersenBlake3s;
    const PROGRAM_WIDTH: usize = 4;
    const NUM_SHIFTED_WIRE_EVALUATIONS: usize = 4;
    const WIRE_SHIFT_SETTINGS: u64 = 0b1111;
    const PERMUTATION_SHIFT: u32 = 30;
    const PERMUTATION_MASK: u32 = 0xC0000000;
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize = 4;
    const IS_PLOOKUP: bool = false;
}

pub trait UltraSettings: SettingsBase {
    const NUM_CHALLENGE_BYTES: usize = 16;
    const HASH_TYPE: transcript::HashType = transcript::HashType::PlookupPedersenBlake3s;
    const PROGRAM_WIDTH: usize = 4;
    const NUM_SHIFTED_WIRE_EVALUATIONS: usize = 4;
    const WIRE_SHIFT_SETTINGS: u64 = 0b1111;
    const PERMUTATION_SHIFT: u32 = 30;
    const PERMUTATION_MASK: u32 = 0xC0000000;
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize = 4;
    const IS_PLOOKUP: bool = true;
}

pub trait UltraToStandardSettings: UltraSettings {
    const HASH_TYPE: transcript::HashType = transcript::HashType::PedersenBlake3s;
}

pub trait UltraWithKeccakSettings: UltraSettings {
    const NUM_CHALLENGE_BYTES: usize = 32;
    const HASH_TYPE: transcript::HashType = transcript::HashType::Keccak256;
}

pub struct Standard;
impl StandardSettings for Standard {}
pub struct Turbo;
impl TurboSettings for Turbo {}
pub struct Ultra;
impl UltraSettings for Ultra {}
pub struct UltraToStandard;
impl UltraToStandardSettings for UltraToStandard {}
pub struct UltraWithKeccak;
impl UltraWithKeccakSettings for UltraWithKeccak {}
