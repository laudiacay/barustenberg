use self::{
    curves::bn254::{fq12::Fq12, g1::G1Affine},
    fields::field::{Field, FieldParams},
};

// TODO todo - stubs to get the compiler to cooperate.
pub(crate) mod curves;
pub(crate) mod fields;
pub(crate) mod groups;

// pub trait FieldElement {
//     type SizeInBytes: typenum::Unsigned; // do a typenum here
// }

// pub trait Field {
//     type Element: FieldElement;
// }

// trait GroupElement {
//     type SizeInBytes: typenum::Unsigned; // do a typenum here
// }

// pub trait Group {
//     type Element: GroupElement;
// }

// pub trait Pairing<G1: Group, G2: Group> {
//     type Output: Group;
// }

pub struct Pippenger {}

struct EllCoeffs<QuadFP: FieldParams> {
    o: Field<QuadFP>,
    vw: Field<QuadFP>,
    vv: Field<QuadFP>,
}

const PRECOMPUTED_COEFFICIENTS_LENGTH: usize = 87;

struct MillerLines {
    lines: [EllCoeffs<Fq12>; PRECOMPUTED_COEFFICIENTS_LENGTH],
}

pub fn reduced_ate_pairing_batch_precomputed(
    p_affines: &[G1Affine],
    miller_lines: &MillerLines,
    num_points: usize,
) -> Fq12 {
    // TODO compilation placeholder come back here bb
    todo!("see comment")
}
