use ark_ff::{FftField, Field};
use typenum::{U11, U3, U4, U5};

/// Specify the structure of a CircuitConstructor
/// This is typically passed as a template argument specifying the structure of a circuit constructor. It
/// should only ever contain circuit constructor data--it should not contain data that is particular to any
/// proving system.
///
/// It may make sense to say this is only partial arithmetization data, with the full data being
/// contained in the circuit constructor. We could change the name of this class if it conflicts with common usage.
pub(crate) trait Arithmetization {
    type NumWires: typenum::Unsigned;
    type NumSelectors: typenum::Unsigned;
    // Note: For even greater modularity, in each instantiation we could specify a list of components here, where a
    // component is a meaningful collection of functions for creating gates, as in:
    //
    // struct Component {
    //     using Arithmetic = component::Arithmetic3Wires;
    //     using RangeConstraints = component::Base4Accumulators or component::GenPerm or...
    //     using LooupTables = component::Plookup4Wire or component::CQ8Wire or...
    //     ...
    // };
    //
    // We should only do this if it becomes necessary or convenient.
}

// These are not magic numbers and they should not be written with global constants. These paraters are not accessible
// through clearly named static class members.

pub(crate) trait Standard: Arithmetization<NumWires = U3, NumSelectors = U5> {}
trait Turbo: Arithmetization<NumWires = U4, NumSelectors = U11> {}
trait Ultra: Arithmetization<NumWires = U4, NumSelectors = U11> {}

use serde::{Deserialize, Serialize};

pub(crate) struct AddTriple<Fr: Field + FftField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) a_scaling: Fr,
    pub(crate) b_scaling: Fr,
    pub(crate) c_scaling: Fr,
    pub(crate) const_scaling: Fr,
}

pub(crate) struct AddQuad<Fr: Field + FftField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) d: u32,
    pub(crate) a_scaling: Fr,
    pub(crate) b_scaling: Fr,
    pub(crate) c_scaling: Fr,
    pub(crate) d_scaling: Fr,
    pub(crate) const_scaling: Fr,
}

pub(crate) struct MulQuad<Fr: Field + FftField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) d: u32,
    pub(crate) mul_scaling: Fr,
    pub(crate) a_scaling: Fr,
    pub(crate) b_scaling: Fr,
    pub(crate) c_scaling: Fr,
    pub(crate) d_scaling: Fr,
    pub(crate) const_scaling: Fr,
}

pub(crate) struct MulTriple<Fr: Field + FftField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) mul_scaling: Fr,
    pub(crate) c_scaling: Fr,
    pub(crate) const_scaling: Fr,
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PolyTriple<Fr: Field + FftField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) q_m: Fr,
    pub(crate) q_l: Fr,
    pub(crate) q_r: Fr,
    pub(crate) q_o: Fr,
    pub(crate) q_c: Fr,
}

pub(crate) struct FixedGroupAddQuad<Fr: Field + FftField> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) d: u32,
    pub(crate) q_x_1: Fr,
    pub(crate) q_x_2: Fr,
    pub(crate) q_y_1: Fr,
    pub(crate) q_y_2: Fr,
}

pub(crate) struct FixedGroupInitQuad<Fr: Field + FftField> {
    pub(crate) q_x_1: Fr,
    pub(crate) q_x_2: Fr,
    pub(crate) q_y_1: Fr,
    pub(crate) q_y_2: Fr,
}

#[derive(Default)]
pub(crate) struct AccumulatorTriple {
    pub(crate) left: Vec<u32>,
    pub(crate) right: Vec<u32>,
    pub(crate) out: Vec<u32>,
}

pub(crate) struct EccAddGate<Fr: Field + FftField> {
    pub(crate) x1: u32,
    pub(crate) y1: u32,
    pub(crate) x2: u32,
    pub(crate) y2: u32,
    pub(crate) x3: u32,
    pub(crate) y3: u32,
    pub(crate) endomorphism_coefficient: Fr,
    pub(crate) sign_coefficient: Fr,
}
