#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_debug_implementations, missing_docs, rust_2018_idioms)]
#![deny(unreachable_pub, private_in_public)]

//! barustenberg
/// random utilities
pub(crate) mod common;
/// crypto stuff
pub(crate) mod crypto;
/// elliptic curves stuff (currently a thin wrapper on other people's grumpkin and bn254)
pub(crate) mod ecc;
/// bitops mostly
pub(crate) mod numeric;
/// plonk
pub mod plonk;
/// math with polynomials
pub(crate) mod polynomials;
/// proof system
pub(crate) mod proof_system;
/// SRS utilities.
pub(crate) mod srs;
/// Transcript utilities.
pub(crate) mod transcript;

// TODOs for claudia and waylon
// big error handling energy, type cleanup
// no asserts only ensures. no unwraps at all.
// nice error messages abound
// logging!
// tests!
// docstrings!
// perf
// why are things pub/not pub/pub(crate)? do things sensibly.
// claudia you cannot be hippity hopping through the vtable like this remove these dyns
// macros and consts and inlining for maxfast
// remove all the extra trait bounds if you don't need them. sometimes you don't need them.
// there are some things that should be parallel that aren't. do a pass.
// pon de inlining
// audit how str versus String is used around the library. probably should have static strings.
// audit where rwlocks are used and where arcs are used?

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
    fn test_add() {
        assert_eq!(add(3, 2), 5);
    }
}
