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

pub(crate) mod gate_data {
    use ark_ff::Field;
    use serde::{Deserialize, Serialize};

    struct AddTriple<Fr: Field> {
        a: u32,
        b: u32,
        c: u32,
        a_scaling: Fr,
        b_scaling: Fr,
        c_scaling: Fr,
        const_scaling: Fr,
    }

    struct AddQuad<Fr: Field> {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        a_scaling: Fr,
        b_scaling: Fr,
        c_scaling: Fr,
        d_scaling: Fr,
        const_scaling: Fr,
    }

    struct MulQuad<Fr: Field> {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        mul_scaling: Fr,
        a_scaling: Fr,
        b_scaling: Fr,
        c_scaling: Fr,
        d_scaling: Fr,
        const_scaling: Fr,
    }

    struct MulTriple<Fr: Field> {
        a: u32,
        b: u32,
        c: u32,
        mul_scaling: Fr,
        c_scaling: Fr,
        const_scaling: Fr,
    }

    #[derive(PartialEq, Eq, Serialize, Deserialize)]
    struct PolyTriple<Fr: Field> {
        a: u32,
        b: u32,
        c: u32,
        q_m: Fr,
        q_l: Fr,
        q_r: Fr,
        q_o: Fr,
        q_c: Fr,
    }

    struct FixedGroupAddQuad<Fr: Field> {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
        q_x_1: Fr,
        q_x_2: Fr,
        q_y_1: Fr,
        q_y_2: Fr,
    }

    struct FixedGroupInitQuad<Fr: Field> {
        q_x_1: Fr,
        q_x_2: Fr,
        q_y_1: Fr,
        q_y_2: Fr,
    }

    struct AccumulatorTriple {
        left: Vec<u32>,
        right: Vec<u32>,
        out: Vec<u32>,
    }

    struct EccAddGate<Fr: Field> {
        x1: u32,
        y1: u32,
        x2: u32,
        y2: u32,
        x3: u32,
        y3: u32,
        endomorphism_coefficient: Fr,
        sign_coefficient: Fr,
    }
}
