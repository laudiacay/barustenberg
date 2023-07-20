use ark_ff::Field;

// limb size when simulating a non-native FieldExt using bigFieldExt class
// (needs to be a universal constant to be used by native verifier)
pub(crate) const NUM_LIMB_BITS_IN_FIELD_SIMULATION: u64 = 68;
pub(crate) const NUM_QUOTIENT_PARTS: u32 = 4;

fn coset_generator<F: Field>(_k: usize) -> F {
    unimplemented!();
}
