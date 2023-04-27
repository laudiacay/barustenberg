// TODO todo - stubs to get the compiler to cooperate.

pub trait Field {
    type SizeInBytes: typenum::Unsigned; // do a typenum here
}

pub mod curves {
    pub mod bn254 {
        pub struct Fr;
        impl super::super::Field for Fr {
            // TODO compilation placeholder come back here bb
            type SizeInBytes = typenum::U0;
        }
    }

    pub mod grumpkin {
        pub struct Fr;
        impl super::super::Field for Fr {
            // TODO compilation placeholder come back here bb
            type SizeInBytes = typenum::U0;
        }
    }
}
