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

#[derive(Debug, Default)]
pub(crate) struct EvaluationDomain<F: Field + FftField> {
    /// n, always a power of 2
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
    /// An entry for each of the log(n) rounds: each entry is a range that
    /// specifies the subset of the roots of unity required for that fft round.
    /// E.g. round_roots[0] = [1, ω^(n/2 - 1)],
    ///      round_roots[1] = [1, ω^(n/4 - 1), ω^(n/2 - 1), ω^(3n/4 - 1)]
    ///      ...
    pub(crate) round_roots: Vec<std::ops::Range<usize>>,
    pub(crate) inverse_round_roots: Vec<std::ops::Range<usize>>,
    pub(crate) roots: Vec<F>,
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
/// * `input_root` - An element of the FieldExt `Fr` that represents the root of the polynomial.
/// * `size` - The size of the polynomial. This is used to determine the number of rounds needed for computation.
/// * `roots` - A mutable vector of elements from the FieldExt `Fr`. This vector is used to store the roots computed in each round.
/// * `roots_offset` - The offset from which to start storing the computed roots in the `roots` vector.
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
fn compute_lookup_table_single<Fr: Field + FftField>(
    input_root: &Fr,
    size: usize,
    roots: &mut [Fr],
    roots_offset: usize,
    round_roots: &mut Vec<std::ops::Range<usize>>,
) {
    let num_rounds = size.get_msb();

    // Creating index ranges
    round_roots.push(roots_offset..roots_offset + 2);
    for i in 1..(num_rounds - 1) {
        let start = round_roots.last().unwrap().end;
        round_roots.push(start..start + (1 << (i + 1)));
    }

    for (i, round_root_i) in round_roots.iter().enumerate().take(num_rounds - 1) {
        let m = 1 << (i + 1);
        let exponent = [(size / (2 * m)) as u64];
        let round_root = input_root.pow(exponent);
        let offset = round_root_i.start;
        roots[offset] = Fr::one();
        for j in 1..m {
            roots[offset + j] = roots[offset + (j - 1)] * round_root;
        }
    }
}

impl<F: Field + FftField> EvaluationDomain<F> {
    pub(crate) fn new(domain_size: usize, target_generator_size: Option<usize>) -> Self {
        let size = domain_size;
        let num_threads = compute_num_threads(size);
        let thread_size = size / num_threads;
        let log2_size = size.get_msb();
        let log2_thread_size = thread_size.get_msb();
        let log2_num_threads = num_threads.get_msb();
        let root = F::get_root_of_unity(domain_size as u64).unwrap();
        // TODO: I guess we don't need the conversion since arkworks internally uses a Montgomery form
        let domain = F::from(size as u64); // .to_montgomery_form();
        let domain_inverse = domain.inverse().unwrap();
        // TODO: Not sure if we need a specific generator or any generator will do
        let generator = F::GENERATOR; // F::coset_generator(0);
        let generator_inverse = generator.inverse().unwrap();
        let four_inverse = F::from(4u64).inverse().unwrap();
        let roots = Vec::new();

        assert!((1 << log2_size) == size || (size == 0));
        assert!((1 << log2_thread_size) == thread_size || (size == 0));
        assert!((1 << log2_num_threads) == num_threads || (size == 0));

        EvaluationDomain {
            size,
            num_threads,
            thread_size,
            log2_size,
            log2_thread_size,
            log2_num_threads,
            generator_size: target_generator_size.unwrap_or(size),
            root,
            root_inverse: root.inverse().unwrap(),
            domain,
            domain_inverse,
            generator,
            generator_inverse,
            four_inverse,
            round_roots: Vec::new(),
            inverse_round_roots: Vec::new(),
            roots,
        }
    }

    pub(crate) fn compute_lookup_table(&mut self) {
        assert!(self.roots.is_empty());
        self.roots = vec![F::default(); 2 * self.size];

        compute_lookup_table_single(
            &self.root,
            self.size,
            &mut self.roots,
            0,
            &mut self.round_roots,
        );
        compute_lookup_table_single(
            &self.root_inverse,
            self.size,
            &mut self.roots,
            self.size,
            &mut self.inverse_round_roots,
        );
    }

    pub(crate) fn compute_generator_table(&mut self, _target_generator_size: usize) {
        // TODO: implement compute_generator_table logic
    }

    pub(crate) fn get_round_roots(&self) -> Vec<&[F]> {
        // TODO: Not sure how to avoid this clone
        self.round_roots
            .iter()
            .map(|r| &self.roots[r.clone()])
            .collect()
    }

    pub(crate) fn get_inverse_round_roots(&self) -> Vec<&[F]> {
        self.inverse_round_roots
            .iter()
            .map(|r| &self.roots[r.clone()])
            .collect()
    }
}

pub(crate) type BarretenbergEvaluationDomain = EvaluationDomain<ark_bn254::Fr>;
pub(crate) type GrumpkinEvaluationDomain = EvaluationDomain<grumpkin::Fr>;
