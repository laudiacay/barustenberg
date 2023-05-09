use crate::transcript::{BarretenHasher, Keccak256, PedersenBlake3s, PlookupPedersenBlake3s};

pub trait Settings<H: BarretenHasher> {
    fn requires_shifted_wire(wire_shift_settings: u64, wire_index: u64) -> bool {
        ((wire_shift_settings >> wire_index) & 1u64) == 1u64
    }
    fn num_challenge_bytes(&self) -> usize;
    fn program_width(&self) -> usize;
    fn num_shifted_wire_evaluations(&self) -> usize;
    fn wire_shift_settings(&self) -> u64;
    fn permutation_shift(&self) -> u32;
    fn permutation_mask(&self) -> u32;
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize;
    fn is_plookup(&self) -> bool;
    fn hasher(&self) -> H;
}

pub struct StandardSettings<H: BarretenHasher> {
    hasher: H,
}

impl<H: BarretenHasher> StandardSettings<H> {
    pub fn new(h: H) -> Self {
        Self { hasher: h }
    }
}

impl<H: BarretenHasher> Settings<H> for StandardSettings<H> {
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    fn program_width(&self) -> usize {
        3
    }
    fn num_shifted_wire_evaluations(&self) -> usize {
        1
    }
    fn wire_shift_settings(&self) -> u64 {
        0b0100
    }
    fn permutation_shift(&self) -> u32 {
        30
    }
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    fn is_plookup(&self) -> bool {
        false
    }
    fn hasher(&self) -> dyn BarretenHasher {
        self.hasher
    }
}

pub struct TurboSettings {}

impl TurboSettings {
    pub fn new() -> Self {
        Self {}
    }
}

impl Settings<PedersenBlake3s> for TurboSettings {
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    fn program_width(&self) -> usize {
        4
    }
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    fn permutation_shift(&self) -> u32 {
        30
    }
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    fn is_plookup(&self) -> bool {
        false
    }
    fn hasher(&self) -> dyn BarretenHasher {
        PedersenBlake3s {}
    }
}

pub trait UltraSettingsBase {
    fn program_width(&self) -> usize;
    fn num_shifted_wire_evaluations(&self) -> usize;
    fn wire_shift_settings(&self) -> u64;
    fn permutation_shift(&self) -> u32;
    fn permutation_mask(&self) -> u32;
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize;
    fn is_plookup(&self) -> bool;
}

pub struct UltraSettings {}

impl Settings<PlookupPedersenBlake3s> for UltraSettings {
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    fn program_width(&self) -> usize {
        4
    }
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    fn permutation_shift(&self) -> u32 {
        30
    }
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    fn is_plookup(&self) -> bool {
        false
    }
    fn hasher(&self) -> dyn BarretenHasher {
        PlookupPedersenBlake3s {}
    }
}

pub struct UltraToStandardSettings {}

impl Settings<PedersenBlake3s> for UltraToStandardSettings {
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    fn program_width(&self) -> usize {
        4
    }
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    fn permutation_shift(&self) -> u32 {
        30
    }
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    fn is_plookup(&self) -> bool {
        false
    }
    fn hasher(&self) -> dyn BarretenHasher {
        PedersenBlake3s {}
    }
}

pub struct UltraWithKeccakSettings {}

impl Settings<Keccak256> for UltraWithKeccakSettings {
    fn num_challenge_bytes(&self) -> usize {
        32
    }
    fn program_width(&self) -> usize {
        4
    }
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    fn permutation_shift(&self) -> u32 {
        30
    }
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    fn is_plookup(&self) -> bool {
        false
    }
    fn hasher(&self) -> dyn BarretenHasher {
        Keccak256 {}
    }
}
