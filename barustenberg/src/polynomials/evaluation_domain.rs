use ark_ff::{FftField, Field};

use crate::numeric::bitop::Msb;
use std::marker::PhantomData;
use std::vec::Vec;

pub(crate) const MIN_GROUP_PER_THREAD: usize = 4;

#[derive(Debug, Default)]
pub(crate) struct EvaluationDomain<'a, F: Field + FftField> {
    /// n, always a power of 2
    pub(crate) size: usize,
    /// num_threads * thread_size = size
    pub(crate) num_threads: usize,
    pub(crate) thread_size: usize,
    pub(crate) log2_size: usize,
    pub(crate) log2_thread_size: usize,
    pub(crate) log2_num_threads: usize,
    pub(crate) generator_size: usize,
    /// omega; the nth root of unity
    pub(crate) root: F,
    /// omega^{-1}
    pub(crate) root_inverse: F,
    /// n; same as size
    pub(crate) domain: F,
    /// n^{-1}
    pub(crate) domain_inverse: F,
    pub(crate) generator: F,
    pub(crate) generator_inverse: F,
    pub(crate) four_inverse: F,
    /// An entry for each of the log(n) rounds: each entry is a pointer to
    /// the subset of the roots of unity required for that fft round.
    /// E.g. round_roots[0] = [1, ω^(n/2 - 1)],
    ///      round_roots[1] = [1, ω^(n/4 - 1), ω^(n/2 - 1), ω^(3n/4 - 1)]
    ///      ...
    pub(crate) round_roots: Vec<std::ops::Range<usize>>,
    pub(crate) inverse_round_roots: Vec<std::ops::Range<usize>>,
    pub(crate) roots: Vec<F>,
    pub(crate) phantom: PhantomData<&'a ()>,
}

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

    for i in 0..(num_rounds - 1) {
        let m = 1 << (i + 1);
        let exponent = [(size / (2 * m)) as u64];
        let round_root = input_root.pow(exponent);
        let offset = round_roots[i].start - roots_offset;
        roots[offset] = Fr::one();
        for j in 1..m {
            roots[offset + j] = roots[offset + (j - 1)] * round_root;
        }
    }
}

impl<'a, F: Field + FftField> EvaluationDomain<'a, F> {
    pub(crate) fn new(domain_size: usize, target_generator_size: Option<usize>) -> Self {
        let size = domain_size;
        let num_threads = compute_num_threads(size);
        let thread_size = size / num_threads;
        let log2_size = size.get_msb();
        let log2_thread_size = thread_size.get_msb();
        let log2_num_threads = num_threads.get_msb();
        let root = F::get_root_of_unity(log2_size as u64).unwrap();
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
            size: size,
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
            phantom: PhantomData,
        }
    }

    pub(crate) fn compute_lookup_table(&mut self) {
        assert!(self.roots.is_empty());
        self.roots = vec![F::zero(); 2 * self.size];

        compute_lookup_table_single(
            &self.root,
            self.size,
            &mut self.roots[..self.size],
            0,
            &mut self.round_roots,
        );
        compute_lookup_table_single(
            &self.root_inverse,
            self.size,
            &mut self.roots[self.size..],
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

pub(crate) type BarretenbergEvaluationDomain<'a> = EvaluationDomain<'a, ark_bn254::Fr>;
pub(crate) type GrumpkinEvaluationDomain<'a> = EvaluationDomain<'a, grumpkin::Fr>;
