#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_debug_implementations, missing_docs)]
#![deny(unreachable_pub, private_in_public)]
//! # Evaluation Domain
//!
//! This module contains the `EvaluationDomain` struct and related functionality.
//! It plays an important role in finite field and number-theoretic computations,
//! especially Fast Fourier Transforms (FFTs). In the context of FFTs, the evaluation
//! domain represents a set of points in a finite field where a polynomial is evaluated.
//!
//! The `EvaluationDomain` struct stores various properties of the finite field,
//! such as the size of the domain, the root of unity, and its inverse, among other things.
//! It also maintains lookup tables for the roots of unity required for each FFT round.
//!
//! Furthermore, this struct is designed to support parallel computation. It includes fields
//! that denote the number of threads and thread sizes for these computations.
//!
//! This module also provides specific types for evaluation domains over the fields
//! defined by the ark_bn254 and grumpkin crates, namely `BarretenbergEvaluationDomain`
//! and `GrumpkinEvaluationDomain`.
//!
//! The `compute_num_threads` function calculates the number of threads for parallel
//! computations, depending on the size of the domain and whether multithreading is enabled.
//! It is particularly useful when working with large polynomials and fields.
//!
//! The `compute_lookup_table_single` function is a utility for precomputing and storing
//! the roots of a polynomial in a lookup table, which can greatly accelerate subsequent FFT operations.

use ark_ff::{FftField, Field};

use crate::numeric::bitop::Msb;
use std::vec::Vec;

pub(crate) const MIN_GROUP_PER_THREAD: usize = 4;

#[derive(Default)]
pub(crate) struct EvaluationDomain<'a, F: Field + FftField> {
     /// The size of the domain, always a power of 2 (i.e., 2^n).
    pub(crate) size: usize,
    /// The number of threads used for parallel computation.
    pub(crate) num_threads: usize,
    /// The size of each thread.
    /// num_threads * thread_size = size
    pub(crate) thread_size: usize,
    /// The logarithm base 2 of the domain size (i.e., n).
    pub(crate) log2_size: usize,
    /// The logarithm base 2 of the thread size.
    pub(crate) log2_thread_size: usize,
    /// The logarithm base 2 of the number of threads
    pub(crate) log2_num_threads: usize,
    /// The size of the generator.
    pub(crate) generator_size: usize,
    /// The Nth root of unity,
    pub(crate) root: F,
    /// The inverse of the root of unity, omega^{-1}.
    pub(crate) root_inverse: F,
    /// The domain, equivalent to the size (i.e., 2^n).
    pub(crate) domain: F,
    /// Equivalent to the domain, but as an inverse (i.e., 2^{-n}).
    pub(crate) domain_inverse: F,
    /// The generator.
    pub(crate) generator: F,
    /// The inverse of the generator.
    pub(crate) generator_inverse: F,
    /// The inverse of four.
    pub(crate) four_inverse: F,
    /// Lookup table for roots of unity for each FFT round.
    /// An entry for each of the log(n) rounds: each entry is a pointer to
    /// the subset of the roots of unity required for that fft round.
    /// E.g. round_roots[0] = [1, ω^(n/2 - 1)],
    ///      round_roots[1] = [1, ω^(n/4 - 1), ω^(n/2 - 1), ω^(3n/4 - 1)]
    ///      ...
    pub(crate) round_roots: &'a [&'a [F]],
     /// Lookup table for the inverse of the roots of unity for each FFT round.
    pub(crate) inverse_round_roots: &'a [&'a [F]],
}

/// Computes the number of threads to be used based on the domain size and the
/// minimum number of groups per thread.
///
/// If multithreading feature is enabled, it uses the maximum number of threads
/// that can run simultaneously. If the domain size is smaller than the
/// product of the number of threads and the minimum number of groups per
/// thread, it uses a single thread.
///
/// Otherwise, it uses the number of threads computed by the `max_threads`
/// function from the `common` module.
fn compute_num_threads(size: usize) -> usize {
    #[cfg(feature = "multithreading")]
    let num_threads = crate::common::max_threads::compute_num_threads();
    #[cfg(not(feature = "multithreading"))]
    let num_threads = 1;
    if size <= num_threads * MIN_GROUP_PER_THREAD {
        1
    } else {
        num_threads
    }
}
/// This function computes a lookup table for the roots of a polynomial.
///
/// # Arguments
///
/// * `input_root` - An element of the field `Fr` that represents the root of the polynomial.
/// * `size` - The size of the polynomial. This is used to determine the number of rounds needed for computation.
/// * `roots` - A mutable vector of elements from the field `Fr`. This vector is used to store the roots computed in each round.
/// * `round_roots` - A mutable vector of `usize` values representing indices into `roots`. After each round, the index of the newly computed root is stored in this vector.
///
/// # Description
///
/// This function operates in several rounds. In each round, it computes a new root of the polynomial and stores it in the `roots` vector.
/// The index of the new root in the `roots` vector is then stored in the `round_roots` vector.
///
/// The new root in each round is computed by raising the `input_root` to a power that is determined by the current round and the size of the polynomial.
/// The result is then multiplied with the previous root to get the new root.
///
/// # Example
///
/// ```
/// use ark_ff::One;
/// use ark_bn254::Fr;
///
/// // Assume Fr, input_root are properly defined here
/// let size = 16;
/// let mut roots = vec![Fr::one(); size];
/// let mut round_roots = Vec::new();
/// compute_lookup_table_single(&input_root, size, &mut roots, &mut round_roots);
/// ```
fn compute_lookup_table_single<Fr: Field>(
    input_root: &Fr,
    size: usize,
    roots: &mut [Fr],
    round_roots: &mut Vec<usize>,
) {
    let num_rounds = (size as f64).log2().ceil() as usize;

    round_roots.push(0);
    for i in 1..num_rounds - 1 {
        let last = *round_roots.last().unwrap();
        round_roots.push(last + (1 << i));
    }
    for (i, round_root_i) in round_roots.iter().enumerate().take(num_rounds - 1) {
        let m = 1 << (i + 1);
        let exponent = [(size / (2 * m)) as u64];
        let round_root = input_root.pow(exponent);
        roots[*round_root_i] = Fr::one();
        for j in 1..m {
            roots[round_root_i + j] = roots[round_root_i + j - 1] * round_root;
        }
    }
}
impl<'a, F: Field + FftField> EvaluationDomain<'a, F> {
    pub(crate) fn new(domain_size: usize, _target_generator_size: Option<usize>) -> Self {
        // TODO: implement constructor logic

        let size = domain_size;
        let num_threads = compute_num_threads(size);
        let thread_size = size / num_threads;
        let _log2_size = size.get_msb();
        let _log2_thread_size = thread_size.get_msb();
        let _log2_num_threads = num_threads.get_msb();
        // let root = F::get_root_of_unity(log2_size);
        // let domain = F::new(size, 0,0,0).to_montgomery_form();
        // let domain_inverse = domain.inverse().unwrap();
        // let generator = F::coset_generator(0);
        // let generator_inverse = generator.inverse().unwrap();
        // let four_inverse = F::from(4).inverse().unwrap();
        // let roots = None;

        todo!("fix ");
        // assert!((1UL << log2_size) == size || (size == 0));
        // assert!((1UL << log2_thread_size) == thread_size || (size == 0));
        // assert!((1UL << log2_num_threads) == num_threads || (size == 0));

        // EvaluationDomain { size: size,
        //     num_threads,
        //     thread_size,
        //     log2_size,
        //     log2_thread_size,
        //     log2_num_threads,
        //     // TODO original was generator_size(target_generator_size ? target_generator_size : domain_size)- check me
        //     generator_size: if target_generator_size == 0 { size } else { target_generator_size },
        //     root,
        //     root_inverse: root.inverse().unwrap(),
        //     domain,
        //     domain_inverse,
        //     generator,
        //     generator_inverse,
        //     four_inverse,
        //     roots,
        //     round_roots: None,
        //     inverse_round_roots: None,
        // }
    }

    pub(crate) fn compute_lookup_table(&mut self) {
        // TODO: implement compute_lookup_table logic
    }

    pub(crate) fn compute_generator_table(&mut self, _target_generator_size: usize) {
        // TODO: implement compute_generator_table logic
    }

    pub(crate) fn get_round_roots(&self) -> &[&[F]] {
        self.round_roots
    }

    pub(crate) fn get_inverse_round_roots(&self) -> &[&[F]] {
        self.inverse_round_roots
    }
}

pub(crate) type BarretenbergEvaluationDomain<'a> = EvaluationDomain<'a, ark_bn254::Fr>;
pub(crate) type GrumpkinEvaluationDomain<'a> = EvaluationDomain<'a, grumpkin::Fr>;
