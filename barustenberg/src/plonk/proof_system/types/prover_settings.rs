use crate::{proof_system::arithmetization, transcript::HasherType};

pub trait SettingsBase<Hash: HasherType> {
    fn requires_shifted_wire(wire_shift_settings: u64, wire_index: u64) -> bool {
        ((wire_shift_settings >> wire_index) & 1u64) == 1u64
    }
}

pub trait StandardSettings<Hash: HasherType>: SettingsBase<Hash> {
    type Arithmetization: arithmetization::Standard;
    const NUM_CHALLENGE_BYTES: usize = 16;
    const PROGRAM_WIDTH: usize = 3;
    const NUM_SHIFTED_WIRE_EVALUATIONS: usize = 1;
    const WIRE_SHIFT_SETTINGS: u64 = 0b0100;
    const PERMUTATION_SHIFT: u32 = 30;
    const PERMUTATION_MASK: u32 = 0xC0000000;
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize = 4;
    const IS_PLOOKUP: bool = false;
}

pub trait TurboSettings: SettingsBase<PedersenBlake3s> {
    const NUM_CHALLENGE_BYTES: usize = 16;
    const PROGRAM_WIDTH: usize = 4;
    const NUM_SHIFTED_WIRE_EVALUATIONS: usize = 4;
    const WIRE_SHIFT_SETTINGS: u64 = 0b1111;
    const PERMUTATION_SHIFT: u32 = 30;
    const PERMUTATION_MASK: u32 = 0xC0000000;
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize = 4;
    const IS_PLOOKUP: bool = false;
}

pub trait UltraSettingsBase<Hash: HasherType>: SettingsBase<Hash> {
    const PROGRAM_WIDTH: usize = 4;
    const NUM_SHIFTED_WIRE_EVALUATIONS: usize = 4;
    const WIRE_SHIFT_SETTINGS: u64 = 0b1111;
    const PERMUTATION_SHIFT: u32 = 30;
    const PERMUTATION_MASK: u32 = 0xC0000000;
    const NUM_ROOTS_CUT_OUT_OF_VANISHING_POLYNOMIAL: usize = 4;
    const IS_PLOOKUP: bool = true;
}

pub trait UltraSettings: UltraSettingsBase< dyn HasherType::PlookupPedersenBlake3s> {
    const NUM_CHALLENGE_BYTES: usize = 16;
}

pub trait UltraToStandardSettings: UltraSettings<HasherType::PedersenBlake3s> {
    const NUM_CHALLENGE_BYTES: usize = 16;
}

pub trait UltraWithKeccakSettings: UltraSettings<HasherType::Keccak256> {
    const NUM_CHALLENGE_BYTES: usize = 32;
}
