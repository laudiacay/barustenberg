#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_debug_implementations, missing_docs, rust_2018_idioms)]
#![deny(unreachable_pub, private_in_public)]

//! barustenberg
pub mod common;
pub mod ecc;
pub mod numeric;
pub mod plonk;
pub mod polynomials;
pub mod proof_system;
pub mod srs;
pub mod transcript;

// TODOs for claudia and waylon
// big error handling energy, type cleanup
// logging!
// tests!
// docstrings!
// perf
// why are things pub/not pub/pub(crate)? do things sensibly.
// claudia you cannot be hippity hopping through the vtable like this remove these dyns
// macros and consts and inlining for maxfast

/// Test utilities.
#[cfg(any(test, feature = "test_utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test_utils")))]
pub mod test_utils;

/// Add two integers together.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiplies two integers together.
pub fn mult(a: i32, b: i32) -> i32 {
    a * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mult() {
        assert_eq!(mult(3, 2), 6);
    }
}
