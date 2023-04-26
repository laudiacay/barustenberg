// TODO todo - stubs to get the compiler to cooperate.

pub trait Field {}

pub mod curves {
    pub mod bn254 {
        pub struct Fr;
        impl super::super::Field for Fr {}
    }

    pub mod grumpkin {
        pub struct Fr;
        impl super::super::Field for Fr {}
    }
}
