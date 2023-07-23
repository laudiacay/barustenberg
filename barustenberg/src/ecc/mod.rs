use ark_bn254::Fq12;
use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use self::curves::pairings::MillerLines;

// TODO todo - stubs to get the compiler to cooperate.
pub(crate) mod curves;

pub(crate) fn reduced_ate_pairing_batch_precomputed<G: AffineRepr>(
    _p_affines: &[G],
    _miller_lines: &Vec<MillerLines>,
    _num_points: usize,
) -> Fq12 {
    // TODO compilation placeholder come back here bb
    todo!("see comment")
}

#[inline]
pub(crate) fn conditionally_subtract_from_double_modulus<Fr: Field + FftField>(
    _this: &Fr,
    _predicate: u64,
) -> Fr {
    todo!("see comment")
    // yikes man
}

#[inline]
pub(crate) fn tag_coset_generator<Fr: Field + FftField>() -> Fr {
    todo!("see comment")
    // yikes man
}
#[inline]
pub(crate) fn coset_generator<Fr: Field + FftField>(_n: u8) -> Fr {
    todo!("see comment")
    // yikes man
}
