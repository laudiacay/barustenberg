// TODO todo - stubs to get the compiler to cooperate.

pub trait FieldElement {
    type SizeInBytes: typenum::Unsigned; // do a typenum here
}

pub trait Field {
    type Element: FieldElement;
}

trait GroupElement {
    type SizeInBytes: typenum::Unsigned; // do a typenum here
}

pub trait Group {
    type Element: GroupElement;
}

pub trait Pairing<G1: Group, G2: Group> {
    type Output: Group;
}

pub struct Pippenger {}

pub mod curves {
    pub mod bn254 {
        pub struct Fr;
        impl super::super::FieldElement for Fr {
            // TODO compilation placeholder come back here bb
            type SizeInBytes = typenum::U0;
        }
        impl super::super::Field for Fr {
            // TODO compilation placeholder come back here bb
            type Element = Fr;
        }
    }

    pub mod grumpkin {
        pub struct Fr;
        impl super::super::FieldElement for Fr {
            // TODO compilation placeholder come back here bb
            type SizeInBytes = typenum::U0;
        }
        impl super::super::Field for Fr {
            // TODO compilation placeholder come back here bb
            type Element = Fr;
        }
    }
}
