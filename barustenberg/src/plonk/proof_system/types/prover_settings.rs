use crate::transcript::{BarretenHasher, Keccak256, PedersenBlake3s, PlookupPedersenBlake3s};

// TODO bevy_reflect? or what
// todo at least inline it all
pub(crate) trait Settings<H: BarretenHasher> {
    #[inline]
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
    fn hasher(&self) -> &H;
}

pub(crate) struct StandardSettings<H: BarretenHasher> {
    hasher: H,
}

impl<H: BarretenHasher> StandardSettings<H> {
    pub(crate) fn new(h: H) -> Self {
        Self { hasher: h }
    }
}

impl<H: BarretenHasher> Settings<H> for StandardSettings<H> {
    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        3
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        1
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b0100
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &H {
        &self.hasher
    }

    #[inline]
    fn requires_shifted_wire(wire_shift_settings: u64, wire_index: u64) -> bool {
        ((wire_shift_settings >> wire_index) & 1u64) == 1u64
    }
}

pub(crate) struct TurboSettings {}

impl TurboSettings {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl Settings<PedersenBlake3s> for TurboSettings {
    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &PedersenBlake3s {
        &PedersenBlake3s {}
    }
}

pub(crate) struct UltraSettings {}

impl Settings<PlookupPedersenBlake3s> for UltraSettings {
    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &PlookupPedersenBlake3s {
        &PlookupPedersenBlake3s {}
    }
}

pub(crate) struct UltraToStandardSettings {}

impl Settings<PedersenBlake3s> for UltraToStandardSettings {
    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        16
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &PedersenBlake3s {
        &PedersenBlake3s {}
    }
}

pub(crate) struct UltraWithKeccakSettings {}

impl Settings<Keccak256> for UltraWithKeccakSettings {
    #[inline]
    fn num_challenge_bytes(&self) -> usize {
        32
    }
    #[inline]
    fn program_width(&self) -> usize {
        4
    }
    #[inline]
    fn num_shifted_wire_evaluations(&self) -> usize {
        4
    }
    #[inline]
    fn wire_shift_settings(&self) -> u64 {
        0b1111
    }
    #[inline]
    fn permutation_shift(&self) -> u32 {
        30
    }
    #[inline]
    fn permutation_mask(&self) -> u32 {
        0xC0000000
    }
    #[inline]
    fn num_roots_cut_out_of_vanishing_polynomial(&self) -> usize {
        4
    }
    #[inline]
    fn is_plookup(&self) -> bool {
        false
    }
    #[inline]
    fn hasher(&self) -> &Keccak256 {
        &Keccak256 {}
    }
}
