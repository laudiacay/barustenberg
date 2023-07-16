#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_debug_implementations, missing_docs)]
#![deny(unreachable_pub, private_in_public)]
//! This module provides various functions to perform operations on polynomials, with a focus on 
//! arithmetic and evaluation functionality. 
//!
//! Polynomials are represented as vectors of field elements where each element is a coefficient 
//! in the polynomial. The i-th element of the vector represents the coefficient for the x^i term.
//! 
//! # Main Functions
//! 
//! - `compute_efficient_interpolation`: Computes the polynomial that interpolates a given set 
//! of points in an efficient manner using the Lagrange Interpolation formula.
//! 
//! - `evaluate`: Evaluates a given polynomial at a specific point. This function can use parallel 
//! computation to speed up the evaluation.
//!
//! - `*_fft_*`: Computes the Fast Fourier Transform (FFT) of a polynomial in multiple different ways.
//!
//! # Helper Functions
//! 
//! - `compute_linear_polynomial_product`, `compute_num_threads`, `reverse_bits`, 
//! `is_power_of_two_usize`, `compute_barycentric_lagrange_weights`, `compute_multiexp`, 
//! `compute_inner_product`: These functions are used as helpers for the main functions to 
//! perform operations such as product of linear terms, computation of weights for interpolation, 
//! computing multi-exponentiation and inner products, as well as bit manipulation for the FFT.
//!
//! This module forms the computational backbone for performing operations on polynomials, 
//! including interpolation, evaluation, and transformations like the FFT. It is built with 
//! performance in mind, leveraging concurrent processing where possible.

use anyhow::ensure;
use ark_ff::{FftField, Field};

use crate::{common::max_threads::compute_num_threads, numeric::bitop::Msb};

pub(crate) struct LagrangeEvaluations<Fr: Field + FftField> {
    pub(crate) vanishing_poly: Fr,
    pub(crate) l_start: Fr,
    pub(crate) l_end: Fr,
}

#[inline]
/// Reverses the first `bit_length` bits of a `u32` number.
///
/// This function takes a `u32` number `x` and a bit length `bit_length`. It reverses the order of
/// the first `bit_length` bits in `x` and returns the result.
///
/// # Arguments
///
/// * `x` - A `u32` number whose bits are to be reversed.
/// * `bit_length` - A `u32` specifying the number of bits to be reversed.
///
/// # Returns
///
/// * A `u32` number with the first `bit_length` bits reversed.
///
/// # Examples
///
/// ```
/// assert_eq!(reverse_bits(0b1011, 4), 0b1101);
/// ```
fn reverse_bits(x: u32, bit_length: u32) -> u32 {
    let x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    let x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    let x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    let x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    ((x >> 16) | (x << 16)) >> (32 - bit_length)
}

#[inline]
/// Checks whether a `u64` number is a power of two.
///
/// # Arguments
///
/// * `x` - A `u64` number to be checked.
///
/// # Returns
///
/// * A `bool` that is `true` if `x` is a power of two and `false` otherwise.
///
/// # Examples
///
/// ```
/// assert_eq!(is_power_of_two(4), true);
/// assert_eq!(is_power_of_two(6), false);
/// ```
fn is_power_of_two(x: u64) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

#[inline]
/// Checks whether a `usize` number is a power of two.
///
/// # Arguments
///
/// * `x` - A `usize` number to be checked.
///
/// # Returns
///
/// * A `bool` that is `true` if `x` is a power of two and `false` otherwise.
///
/// # Examples
///
/// ```
/// assert_eq!(is_power_of_two_usize(4), true);
/// assert_eq!(is_power_of_two_usize(6), false);
/// ```
fn is_power_of_two_usize(x: usize) -> bool {
    x != 0 && (x & (x - 1)) == 0
}

/// Copies a polynomial from `src` to `dest` with specified numbers of coefficients.
///
/// This function copies the first `num_src_coefficients` coefficients from `src` to `dest`. If 
/// `num_target_coefficients` is greater than `num_src_coefficients`, the remaining coefficients in 
/// `dest` are set to zero.
///
/// # Type Parameters
///
/// * `Fr` - A type that implements `Copy` and `Default` traits.
///
/// # Arguments
///
/// * `src` - A slice of coefficients from the source polynomial.
/// * `dest` - A mutable slice of coefficients where the coefficients from `src` are copied.
/// * `num_src_coefficients` - The number of coefficients to copy from `src`.
/// * `num_target_coefficients` - The total number of coefficients in the target polynomial.
///
/// # Examples
///
/// ```
/// use ...;  // import the appropriate crate and function
///
/// let src = vec![Fr::from(1), Fr::from(2), Fr::from(3)];  // replace `Fr::from` with the appropriate constructor for your field type
/// let mut dest = vec![Fr::default(); 4];
///
/// copy_polynomial(&src, &mut dest, src.len(), dest.len());
///
/// assert_eq!(dest, vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::default()]);  // replace `Fr::from` and `Fr::default` with the appropriate constructor for your field type
/// ```
fn copy_polynomial<Fr: Copy + Default>(
    src: &[Fr],
    dest: &mut [Fr],
    num_src_coefficients: usize,
    num_target_coefficients: usize,
) {
    // TODO: fiddle around with avx asm to see if we can speed up
    dest[..num_src_coefficients].copy_from_slice(&src[..num_src_coefficients]);

    if num_target_coefficients > num_src_coefficients {
        // fill out the polynomial coefficients with zeroes
        for item in dest
            .iter_mut()
            .take(num_target_coefficients)
            .skip(num_src_coefficients)
        {
            *item = Fr::default();
        }
    }
}

use std::ops::{Add, Mul, Sub};

use super::{evaluation_domain::EvaluationDomain, Polynomial};

/// Performs a serial Fast Fourier Transform (FFT) on a set of polynomials.
///
/// This function computes the FFT on a set of polynomials represented by `coeffs`. The 
/// polynomials are evaluated at the `domain_size`-th roots of unity, given in the `root_table`.
///
/// It uses an iterative in-place Cooley-Tukey FFT algorithm, which requires that the number 
/// of polynomial coefficients (i.e., `domain_size`) is a power of two.
///
/// The computation is performed in serial, meaning it uses a single thread. For large inputs, 
/// a parallel FFT algorithm may be more efficient.
///
/// # Type Parameters
///
/// * `Fr` - A field type that implements `Copy`, `Default`, `Add`, `Sub` and `Mul`. The operations 
///          must take no arguments and return an instance of the same type.
///
/// # Arguments
///
/// * `coeffs` - A mutable slice of vectors representing the coefficients of the polynomials. 
///              Each vector is a polynomial and the FFT is performed in-place, meaning the output 
///              is stored in `coeffs`.
/// * `domain_size` - The size of the domain over which the FFT is computed. It must be a power 
///                   of two and divide evenly into the number of polynomials.
/// * `root_table` - A slice of vectors representing the `domain_size`-th roots of unity. These 
///                  are used in the FFT computation.
///
/// # Panics
///
/// Panics if `domain_size` is not a power of two, or if `domain_size` does not divide evenly 
/// into the number of polynomials.
///
/// # Examples
///
/// ```
/// use ...;  // import the appropriate crate and function
///
/// let mut coeffs = vec![vec![Fr::from(1), Fr::from(2)], vec![Fr::from(3), Fr::from(4)]];
/// let root_table = ...;  // compute or provide the roots of unity
/// let domain_size = 2;
///
/// fft_inner_serial(&mut coeffs, domain_size, &root_table);
///
/// // `coeffs` now contains the FFT of the input polynomials
/// ```
fn fft_inner_serial<Fr: Copy + Default + Add<Output = Fr> + Sub<Output = Fr> + Mul<Output = Fr>>(
    coeffs: &mut [Vec<Fr>],
    domain_size: usize,
    root_table: &[Vec<Fr>],
) {
    // Assert that the number of polynomials is a power of two.
    let num_polys = coeffs.len();
    assert!(is_power_of_two_usize(num_polys));
    let poly_domain_size = domain_size / num_polys;
    assert!(is_power_of_two_usize(poly_domain_size));

    let log2_size = domain_size.get_msb();
    let log2_poly_size = poly_domain_size.get_msb();

    for i in 0..=domain_size {
        let swap_index = reverse_bits(i as u32, log2_size as u32) as usize;

        if i < swap_index {
            let even_poly_idx = i >> log2_poly_size;
            let even_elem_idx = i % poly_domain_size;
            let odd_poly_idx = swap_index >> log2_poly_size;
            let odd_elem_idx = swap_index % poly_domain_size;
            coeffs[even_poly_idx][even_elem_idx] = coeffs[odd_poly_idx][odd_elem_idx];
        }
    }

    for coeffs_l in coeffs.iter_mut().take(num_polys) {
        for k in (0..poly_domain_size).step_by(2) {
            let temp = coeffs_l[k + 1];
            coeffs_l[k + 1] = coeffs_l[k] - coeffs_l[k + 1];
            coeffs_l[k] = coeffs_l[k] + temp;
        }
    }

    for m in (2..domain_size).step_by(2) {
        let i = m.get_msb();
        for k in (0..domain_size).step_by(2 * m) {
            for j in 0..m {
                let even_poly_idx = (k + j) >> log2_poly_size;
                let even_elem_idx = (k + j) & (poly_domain_size - 1);
                let odd_poly_idx = (k + j + m) >> log2_poly_size;
                let odd_elem_idx = (k + j + m) & (poly_domain_size - 1);

                let temp = root_table[i - 1][j] * coeffs[odd_poly_idx][odd_elem_idx];
                coeffs[odd_poly_idx][odd_elem_idx] = coeffs[even_poly_idx][even_elem_idx] - temp;
                coeffs[even_poly_idx][even_elem_idx] = coeffs[even_poly_idx][even_elem_idx] + temp;
            }
        }
    }
}

impl<Fr: Field + FftField> EvaluationDomain<Fr> {
    /// modifies target[..generator_size]
    fn scale_by_generator(
        &self,
        _coeffs: &mut [Fr],
        _target: &mut [Fr],
        _generator_start: Fr,
        _generator_shift: Fr,
        generator_size: usize,
    ) {
        let _generator_size_per_thread = generator_size / self.num_threads;
        todo!("parallelism");
    }

    /// Compute multiplicative subgroup (g.X)^n.
    /// Compute the subgroup for X in roots of unity of (2^log2_subgroup_size)*n.
    /// X^n will loop through roots of unity (2^log2_subgroup_size).
    /// @param log2_subgroup_size Log_2 of the subgroup size.
    /// @param src_domain The domain of size n.
    /// @param subgroup_roots Pointer to the array for saving subgroup members.

    fn compute_multiplicative_subgroup(
        &self,
        log2_subgroup_size: usize,
        subgroup_roots: &mut [Fr],
    ) -> anyhow::Result<()> {
        let subgroup_size = 1 << log2_subgroup_size;

        // Step 1: get primitive 4th root of unity
        let subgroup_root = Fr::get_root_of_unity(log2_subgroup_size as u64)
            .ok_or_else(|| anyhow::anyhow!("Failed to find root of unity"))?;

        // Step 2: compute the cofactor term g^n
        let mut accumulator = self.generator;
        for _ in 0..self.log2_size {
            accumulator.square_in_place();
        }

        // Step 3: fill array with subgroup_size values of (g.X)^n, scaled by the cofactor
        subgroup_roots[0] = accumulator;
        for i in 1..subgroup_size {
            subgroup_roots[i] = subgroup_roots[i - 1] * subgroup_root;
        }
        Ok(())
    }

    // TODO readd pragma omp parallel
    pub(crate) fn fft_inner_parallel_vec_inplace(
        &self,
        coeffs: &mut [&mut [Fr]],
        _fr: &Fr,
        root_table: &[&[Fr]],
    ) {
        //let scratch_space = Self::get_scratch_space(self.size); // Implement the get_scratch_space function

        let mut scratch_space = vec![Fr::zero(); self.size];

        let num_polys = coeffs.len();
        assert!(num_polys.is_power_of_two());
        let poly_size = self.size / num_polys;
        assert!(poly_size.is_power_of_two());
        let poly_mask = poly_size - 1;
        let log2_poly_size = poly_size.trailing_zeros() as usize;

        // First FFT round is a special case - no need to multiply by root table, because all entries are 1.
        // We also combine the bit reversal step into the first round, to avoid a redundant round of copying data
        for j in 0..self.num_threads {
            let mut temp_1;
            let mut temp_2;
            for i in (j * self.thread_size..(j + 1) * self.thread_size).step_by(2) {
                //let next_index_1 = reverse_bits((i + 2) as u32, self.log2_size as u32) as usize;
                //let next_index_2 = reverse_bits((i + 3) as u32, self.log2_size as u32) as usize;
                // TODO builtin prefetch stuff here
                let swap_index_1 = reverse_bits(i as u32, self.log2_size as u32) as usize;
                let swap_index_2 = reverse_bits((i + 1) as u32, self.log2_size as u32) as usize;

                let poly_idx_1 = swap_index_1 >> log2_poly_size;
                let elem_idx_1 = swap_index_1 & poly_mask;
                let poly_idx_2 = swap_index_2 >> log2_poly_size;
                let elem_idx_2 = swap_index_2 & poly_mask;

                temp_1 = coeffs[poly_idx_1][elem_idx_1];
                temp_2 = coeffs[poly_idx_2][elem_idx_2];
                scratch_space[i + 1] = temp_1 - temp_2;
                scratch_space[i] = temp_1 + temp_2;
            }
        }

        // hard code exception for when the domain size is tiny - we won't execute the next loop, so need to manually
        // reduce + copy
        if self.size <= 2 {
            coeffs[0][0] = scratch_space[0];
            coeffs[0][1] = scratch_space[1];
        }
        // Outer FFT loop - iterates over the FFT rounds
        for m in (2..=self.size).step_by(2) {
            for j in 0..self.num_threads {
                let mut temp: Fr;

                // Ok! So, what's going on here? This is the inner loop of the FFT algorithm, and we want to break it
                // out into multiple independent threads. For `num_threads`, each thread will evaluation `domain.size /
                // num_threads` of the polynomial. The actual iteration length will be half of this, because we leverage
                // the fact that \omega^{n/2} = -\omega (where \omega is a root of unity)

                // Here, `start` and `end` are used as our iterator limits, so that we can use our iterator `i` to
                // directly access the roots of unity lookup table
                let start = j * (self.thread_size >> 1);
                let end = (j + 1) * (self.thread_size >> 1);

                // For all but the last round of our FFT, the roots of unity that we need, will be a subset of our
                // lookup table. e.g. for a size 2^n FFT, the 2^n'th roots create a multiplicative subgroup of order 2^n
                //      the 1st round will use the roots from the multiplicative subgroup of order 2 : the 2'th roots of
                //      unity the 2nd round will use the roots from the multiplicative subgroup of order 4 : the 4'th
                //      roots of unity
                // i.e. each successive FFT round will double the set of roots that we need to index.
                // We have already laid out the `root_table` container so that each FFT round's roots are linearly
                // ordered in memory. For all FFT rounds, the number of elements we're iterating over is greater than
                // the size of our lookup table. We need to access this table in a cyclical fasion - i.e. for a subgroup
                // of size x, the first x iterations will index the subgroup elements in order, then for the next x
                // iterations, we loop back to the start.

                // We could implement the algorithm by having 2 nested loops (where the inner loop iterates over the
                // root table), but we want to flatten this out - as for the first few rounds, the inner loop will be
                // tiny and we'll have quite a bit of unneccesary branch checks For each iteration of our flattened
                // loop, indexed by `i`, the element of the root table we need to access will be `i % (current round
                // subgroup size)` Given that each round subgroup size is `m`, which is a power of 2, we can index the
                // root table with a very cheap `i & (m - 1)` Which is why we have this odd `block_mask` variable

                let block_mask = m - 1;

                // The next problem to tackle, is we now need to efficiently index the polynomial element in
                // `scratch_space` in our flattened loop If we used nested loops, the outer loop (e.g. `y`) iterates
                // from 0 to 'domain size', in steps of 2 * m, with the inner loop (e.g. `z`) iterating from 0 to m. We
                // have our inner loop indexer with `i & (m - 1)`. We need to add to this our outer loop indexer, which
                // is equivalent to taking our indexer `i`, masking out the bits used in the 'inner loop', and doubling
                // the result. i.e. polynomial indexer = (i & (m - 1)) + ((i & ~(m - 1)) >> 1) To simplify this, we
                // cache index_mask = ~block_mask, meaning that our indexer is just `((i & index_mask) << 1 + (i &
                // block_mask)`

                let index_mask = !block_mask;

                // `round_roots` fetches the pointer to this round's lookup table. We use `numeric::get_msb(m) - 1` as
                // our indexer, because we don't store the precomputed root values for the 1st round (because they're
                // all 1).

                let round_roots = root_table[(m.trailing_zeros() - 1) as usize];

                // Finally, we want to treat the final round differently from the others,
                // so that we can reduce out of our 'coarse' reduction and store the output in `coeffs` instead of
                // `scratch_space`

                if m != (self.size >> 1) {
                    for i in start..end {
                        let k1 = (i & index_mask) << 1;
                        let j1 = i & block_mask;
                        temp = round_roots[j1] * scratch_space[k1 + j1 + m];
                        scratch_space[k1 + j1 + m] = scratch_space[k1 + j1] - temp;
                        scratch_space[k1 + j1] += temp;
                    }
                } else {
                    for i in start..end {
                        let k1 = (i & index_mask) << 1;
                        let j1 = i & block_mask;

                        let poly_idx_1 = (k1 + j1) >> log2_poly_size;
                        let elem_idx_1 = (k1 + j1) & poly_mask;
                        let poly_idx_2 = (k1 + j1 + m) >> log2_poly_size;
                        let elem_idx_2 = (k1 + j1 + m) & poly_mask;

                        temp = round_roots[j1] * scratch_space[k1 + j1 + m];
                        coeffs[poly_idx_2][elem_idx_2] = scratch_space[k1 + j1] - temp;
                        coeffs[poly_idx_1][elem_idx_1] = scratch_space[k1 + j1] + temp;
                    }
                }
            }
        }
    }

    // TODO readd pragma omp parallel
    pub(crate) fn fft_inner_parallel(
        &self,
        coeffs: &mut [Fr],
        target: &mut [Fr],
        _fr: &Fr,
        root_table: &[&[Fr]],
    ) {
        // TODO parallelize
        // First FFT round is a special case - no need to multiply by root table, because all entries are 1.
        // We also combine the bit reversal step into the first round, to avoid a redundant round of copying data
        (0..self.num_threads).for_each(|j| {
            let mut temp_1;
            let mut temp_2;
            let thread_start = j * self.thread_size;
            let thread_end = (j + 1) * self.thread_size;
            for i in (thread_start..thread_end).step_by(2) {
                //let next_index_1 = reverse_bits((i + 2) as u32, self.log2_size as u32) as usize;
                //let next_index_2 = reverse_bits((i + 3) as u32, self.log2_size as u32) as usize;

                // TODO builtin prefetch :|

                let swap_index_1 = reverse_bits(i as u32, self.log2_size as u32) as usize;
                let swap_index_2 = reverse_bits((i + 1) as u32, self.log2_size as u32) as usize;

                temp_1 = coeffs[swap_index_1];
                temp_2 = coeffs[swap_index_2];
                target[i + 1] = temp_1 - temp_2;
                target[i] = temp_1 + temp_2;
            }
        });

        // hard code exception for when the domain size is tiny - we won't execute the next loop, so need to manually
        // reduce + copy
        if self.size <= 2 {
            coeffs[0] = target[0];
            coeffs[1] = target[1];
        }

        // outer FFT loop
        // TODO this is super incorrect
        for m in (2..self.size).step_by(2) {
            (0..self.num_threads).for_each(|j| {
                let mut temp;

                let start = j * (self.thread_size >> 1);
                let end = (j + 1) * (self.thread_size >> 1);

                let block_mask = m - 1;
                let index_mask = !block_mask;

                let round_roots = &root_table[m.get_msb() - 1];

                for i in start..end {
                    let k1 = (i & index_mask) << 1;
                    let j1 = i & block_mask;
                    temp = round_roots[j1] * target[k1 + j1 + m];
                    target[k1 + j1 + m] = target[k1 + j1] - temp;
                    target[k1 + j1] += temp;
                }
            });
        }
    }

    fn partial_fft_serial_inner(&self, coeffs: &mut [Fr], target: &mut [Fr], root_table: &[&[Fr]]) {
        let n = self.size >> 2;
        let full_mask = self.size - 1;
        let m = self.size >> 1;
        let half_mask = m - 1;
        let round_roots = &root_table[((m as f64).log2() as usize) - 1];
        let mut root_index;

        for i in 0..n {
            for s in 0..4 {
                target[(3 - s) * n + i] = Fr::zero();
                for j in 0..4 {
                    let index = i + j * n;
                    root_index = (index * (s + 1)) & full_mask;
                    target[(3 - s) * n + i] += (if root_index < m {
                        Fr::one()
                    } else {
                        -Fr::one()
                    }) * coeffs[index]
                        * round_roots[root_index & half_mask];
                }
            }
        }
    }

    pub(crate) fn partial_fft_parallel_inner(
        &self,
        coeffs: &mut [Fr],
        root_table: &[&[Fr]],
        constant: Fr,
        is_coset: bool,
    ) {
        let n = self.size >> 2;
        let full_mask = self.size - 1;
        let m = self.size >> 1;
        let half_mask = m - 1;
        let round_roots = &root_table[((m as f64).log2() as usize) - 1];

        let small_domain = EvaluationDomain::<Fr>::new(n, None);

        for i in 0..small_domain.size {
            let temp = [
                coeffs[i],
                coeffs[i + n],
                coeffs[i + 2 * n],
                coeffs[i + 3 * n],
            ];
            coeffs[i] = Fr::zero();
            coeffs[i + n] = Fr::zero();
            coeffs[i + 2 * n] = Fr::zero();
            coeffs[i + 3 * n] = Fr::zero();

            let mut index;
            let mut root_index;
            let mut root_multiplier;
            let mut temp_constant = constant;

            for s in 0..4 {
                for (j, t_j) in temp.iter().enumerate() {
                    index = i + j * n;
                    root_index = index * (s + 1);
                    if is_coset {
                        root_index -= 4 * i;
                    }
                    root_index &= full_mask;
                    root_multiplier = round_roots[root_index & half_mask];
                    if root_index >= m {
                        root_multiplier = -round_roots[root_index & half_mask];
                    }
                    coeffs[(3 - s) * n + i] += root_multiplier * t_j;
                }
                if is_coset {
                    temp_constant *= self.generator;
                    coeffs[(3 - s) * n + i] *= temp_constant;
                }
            }
        }
    }

    pub(crate) fn partial_fft_serial(&self, coeffs: &mut [Fr], target: &mut [Fr]) {
        self.partial_fft_serial_inner(coeffs, target, &self.get_round_roots()[..]);
    }

    pub(crate) fn partial_fft(&self, coeffs: &mut [Fr], constant: Fr, is_coset: bool) {
        self.partial_fft_parallel_inner(coeffs, &self.get_round_roots()[..], constant, is_coset);
    }

    fn fft_inplace(&self, coeffs: &mut [Fr]) {
        self.fft_inner_parallel_vec_inplace(&mut [coeffs], &self.root, &self.get_round_roots()[..]);
    }

    fn fft(&self, coeffs: &mut [Fr], target: &mut [Fr]) {
        self.fft_inner_parallel(coeffs, target, &self.root, &self.get_round_roots()[..]);
    }

    fn fft_vec_inplace(&self, _coeffs: &mut [&mut [Fr]]) {
        todo!();
    }

    // The remaining functions require you to create a version of `fft_inner_parallel` that accepts a Vec<&[T]> as the first parameter.

    pub(crate) fn ifft_inplace(&self, coeffs: &mut Polynomial<Fr>) {
        self.fft_inner_parallel_vec_inplace(
            &mut [coeffs.coefficients.as_mut_slice()],
            &self.root_inverse,
            &self.get_inverse_round_roots()[..],
        );
        todo!("parallelize")
        // for i in 0..self.size {
        //     coeffs[i] *= self.domain_inverse;
        // }
    }

    pub(crate) fn ifft(&self, coeffs: &mut [Fr], target: &mut [Fr]) {
        self.fft_inner_parallel(
            coeffs,
            target,
            &self.root_inverse,
            &self.get_round_roots()[..],
        );
        // TODO parallelize me
        todo!("parallelize here")
        // for i in 0..self.size {
        //     target[i] *= self.domain_inverse;
        // }
    }

    fn ifft_vec_inplace(&self, _coeffs: &mut [&mut [Fr]]) {
        todo!();
    }
    fn ifft_with_constant(&self, _coeffs: &mut [Fr], _value: Fr) {
        todo!();
    }

    pub(crate) fn coset_ifft(&self, _coeffs: &mut [Fr]) {
        todo!()
    }
    pub(crate) fn coset_ifft_vec(&self, _coeffs: &[&mut [&mut Fr]]) {
        todo!()
    }

    fn fft_with_constant(&self, coeffs: &mut [Fr], target: &mut [Fr], value: Fr) {
        self.fft_inner_parallel(coeffs, target, &self.root, &self.get_round_roots()[..]);
        for item in coeffs.iter_mut().take(self.size) {
            *item *= value;
        }
    }

    // The remaining `coset_fft` functions require you to create a version of `scale_by_generator` that accepts a Vec<&[T]> as the first parameter.
    fn coset_fft_inplace_extension(
        coeffs: &mut [Fr],
        small_domain: &Self,
        _large_domain: &Self,
        domain_extension: usize,
    ) -> anyhow::Result<()> {
        let log2_domain_extension = domain_extension.get_msb();
        let primitive_root =
            Fr::get_root_of_unity((small_domain.log2_size + log2_domain_extension) as u64)
                .ok_or_else(|| anyhow::anyhow!("Failed to get root of unity"))?;

        let scratch_space_len = small_domain.size * domain_extension;
        let mut scratch_space = vec![Fr::zero(); scratch_space_len];

        let mut coset_generators = vec![Fr::zero(); domain_extension];
        coset_generators[0] = small_domain.generator;
        for i in 1..domain_extension {
            coset_generators[i] = coset_generators[i - 1] * primitive_root;
        }

        for i in (0..domain_extension).rev() {
            let mut target = vec![Fr::zero(); small_domain.size];
            small_domain.scale_by_generator(
                coeffs,
                &mut target,
                Fr::one(),
                coset_generators[i],
                small_domain.size,
            );
            coeffs[(i * small_domain.size)..(i + 1) * small_domain.size]
                .copy_from_slice(target.as_slice());
        }

        for i in 0..domain_extension {
            small_domain.fft_inner_parallel(
                &mut coeffs[(i * small_domain.size)..],
                &mut scratch_space[(i * small_domain.size)..],
                &small_domain.root,
                &small_domain.get_round_roots()[..],
            );
        }

        if domain_extension == 4 {
            // TODO parallelism
            for j in 0..small_domain.num_threads {
                let start = j * small_domain.thread_size;
                let end = (j + 1) * small_domain.thread_size;
                for i in start..end {
                    scratch_space[i] = coeffs[i << 2];
                    scratch_space[i + (1 << small_domain.log2_size)] = coeffs[(i << 2) + 1];
                    scratch_space[i + (2 << small_domain.log2_size)] = coeffs[(i << 2) + 2];
                    scratch_space[i + (3 << small_domain.log2_size)] = coeffs[(i << 2) + 3];
                }
            }
            for i in 0..small_domain.size {
                for j in 0..domain_extension {
                    scratch_space[i + (j << small_domain.log2_size)] =
                        coeffs[(i << log2_domain_extension) + j];
                }
            }
        } else {
            for i in 0..small_domain.size {
                for j in 0..domain_extension {
                    scratch_space[i + (j << small_domain.log2_size)] =
                        coeffs[(i << log2_domain_extension) + j];
                }
            }
        }
        Ok(())
    }
    pub(crate) fn coset_fft_inplace(&self, _coeffs: &mut [Fr]) {
        unimplemented!()
    }
    fn coset_fft_vec_inplace(&self, _coeffs: &mut [&mut [Fr]]) {
        unimplemented!()
    }
    pub(crate) fn coset_fft(&self, _coeffs: &[Fr], _target: &mut [Fr]) {
        unimplemented!()
    }
    pub(crate) fn coset_fft_with_generator_shift(&self, _coeffs: &mut [Fr], _constant: Fr) {
        unimplemented!()
    }

    pub(crate) fn divide_by_pseudo_vanishing_polynomial(
        &self,
        _coeffs: &[&mut [&mut Fr]],
        _target: &EvaluationDomain<Fr>,
        _num_roots_cut_out_of_vanishing_poly: usize,
    ) {
        unimplemented!()
    }

    /// Computes evaluations of vanishing polynomial Z_H, l_start, l_end at ʓ.
    ///
    /// Note that as we modify the vanishing polynomial by cutting out some roots, we must simultaneously ensure that
    /// the lagrange polynomials we require would be l_1(ʓ) and l_{n-k}(ʓ) where k =
    /// num_roots_cut_out_of_vanishing_polynomial. For notational simplicity, we call l_1 as l_start and l_{n-k} as
    /// l_end.
    ///
    /// # Arguments:
    ///
    /// - `zeta``: the name given (in our code) to the evaluation challenge ʓ from the Plonk paper.
    /// - `domain`: evaluation domain on which said polynomials will be evaluated.
    /// - `num_roots_cut_out_of_vanishing_poly`: num of roots left out of vanishing polynomial.
    ///
    /// # Returns
    ///
    /// - A struct containing lagrange evaluation of Z_H, l_start, l_end poly.
    pub(crate) fn get_lagrange_evaluations(
        &self,
        z: &Fr,
        num_roots_cut_out_of_vanishing_poly: Option<usize>,
    ) -> LagrangeEvaluations<Fr> {
        // NOTE: If in future, there arises a need to cut off more zeros, this method will not require any changes.
        let num_roots_cut_out_of_vanishing_poly = num_roots_cut_out_of_vanishing_poly.unwrap_or(0);

        let z_pow_n = z.pow([self.size as u64]);

        let mut numerator = z_pow_n - Fr::one();

        let mut denominators = vec![Fr::default(); 3];

        // Compute the denominator of Z_H*(ʓ)
        //   (ʓ - ω^{n-1})(ʓ - ω^{n-2})...(ʓ - ω^{n - num_roots_cut_out_of_vanishing_poly})
        // = (ʓ - ω^{ -1})(ʓ - ω^{ -2})...(ʓ - ω^{  - num_roots_cut_out_of_vanishing_poly})
        let mut work_root = self.root_inverse;
        denominators[0] = Fr::one();
        for _ in 0..num_roots_cut_out_of_vanishing_poly {
            denominators[0] *= *z - work_root;
            work_root *= self.root_inverse;
        }

        // The expressions of the lagrange polynomials are:
        //
        //           ω^0.(X^n - 1)      (X^n - 1)
        // L_1(X) = --------------- =  -----------
        //            n.(X - ω^0)       n.(X - 1)
        //
        // Notice: here (in this comment), the index i of L_i(X) counts from 1 (not from 0). So L_1 corresponds to the
        // _first_ root of unity ω^0, and not to the 1-th root of unity ω^1.
        //
        //
        //             ω^{i-1}.(X^n - 1)         X^n - 1          X^n.(ω^{-i+1})^n - 1
        // L_{i}(X) = ------------------ = -------------------- = -------------------- = L_1(X.ω^{-i+1})
        //              n.(X - ω^{i-1})    n.(X.ω^{-(i-1)} - 1) |  n.(X.ω^{-i+1} - 1)
        //                                                      |
        //                                                      since (ω^{-i+1})^n = 1 trivially
        //
        //                                                          (X^n - 1)
        // => L_{n-k}(X) = L_1(X.ω^{k-n+1}) = L_1(X.ω^{k+1}) =  -----------------
        //                                                      n.(X.ω^{k+1} - 1)
        //
        denominators[1] = *z - Fr::one();

        // Compute ω^{num_roots_cut_out_of_vanishing_polynomial + 1}
        let l_end_root = self
            .root
            .pow([(num_roots_cut_out_of_vanishing_poly + 1) as u64]);
        denominators[2] = (*z * l_end_root) - Fr::one();

        batch_inversion(denominators.as_mut_slice());

        let vanishing_poly = numerator * denominators[0]; // (ʓ^n - 1) / (ʓ-ω^{-1}).(ʓ-ω^{-2})...(ʓ-ω^{-k}) =: Z_H*(ʓ)
        numerator *= self.domain_inverse; // (ʓ^n - 1) / n
        let l_start = numerator * denominators[1]; // (ʓ^n - 1) / (n.(ʓ - 1))         =: L_1(ʓ)
        let l_end = numerator * denominators[2]; // (ʓ^n - 1) / (n.(ʓ.ω^{k+1} - 1)) =: L_{n-k}(ʓ)
        LagrangeEvaluations {
            vanishing_poly,
            l_start,
            l_end,
        }
    }
    /// Compute evaluations of lagrange polynomial L_1(X) on the specified domain.
    ///
    /// # Arguments
    ///
    /// - `l_1_coefficients`: A mutable pointer to a buffer of type `Fr` representing the evaluations of L_1(X) for all X = k*n'th roots of unity.
    /// - `src_domain`: The source domain multiplicative generator g.
    /// - `target_domain`: The target domain (kn'th) root of unity w'.
    ///
    /// # Details
    ///
    /// Let the size of the target domain be k*n, where k is a power of 2.
    /// Evaluate L_1(X) = (X^{n} - 1 / (X - 1)) * (1 / n) at the k*n points X_i = w'^i.g,
    /// i = 0, 1,..., k*n-1, where w' is the target domain (kn'th) root of unity, and g is the
    /// source domain multiplicative generator. The evaluation domain is taken to be the coset
    /// w'^i.g, rather than just the kn'th roots, to avoid division by zero in L_1(X).
    /// The computation is done in three steps:
    /// Step 1) (Parallelized) Compute the evaluations of 1/denominator of L_1 at X_i using
    /// Montgomery batch inversion.
    /// Step 2) Compute the evaluations of the numerator of L_1 using the fact that (X_i)^n forms
    /// a subgroup of order k.
    /// Step 3) (Parallelized) Construct the evaluations of L_1 on X_i using the numerator and
    /// denominator evaluations from Steps 1 and 2.
    ///
    /// Note 1: Let w = n'th root of unity. When evaluated at the k*n'th roots of unity, the term
    /// X^{n} forms a subgroup of order k, since (w'^i)^n = w^{in/k} = w^{1/k}. Similarly, for X_i
    /// we have (X_i)^n = (w'^i.g)^n = w^{in/k}.g^n = w^{1/k}.g^n.
    /// For example, if k = 2:
    /// for even powers of w', X^{n} = w^{2in/2} = 1
    /// for odd powers of w', X = w^{i}w^{n/2} -> X^{n} = w^{in}w^{n/2} = -1
    /// The numerator term, therefore, can only take two values (for k = 2):
    /// For even indices: (X^{n} - 1)/n = (g^n - 1)/n
    /// For odd indices: (X^{n} - 1)/n = (-g^n - 1)/n
    ///
    /// Note 2: We can use the evaluations of L_1 to compute the k*n-fft evaluations of any L_i(X).
    /// We can consider `l_1_coefficients` to be a k*n-sized vector of the evaluations of L_1(X),
    /// for all X = k*n'th roots of unity. To compute the vector for the k*n-fft transform of
    /// L_i(X), we perform a (k*i)-left-shift of this vector.
    pub(crate) fn compute_lagrange_polynomial_fft(
        &self,
        l_1_coefficients: &mut Polynomial<Fr>,
        target_domain: &EvaluationDomain<Fr>,
    ) -> anyhow::Result<()> {
        // Step 1: Compute the 1/denominator for each evaluation: 1 / (X_i - 1)
        let multiplicand = target_domain.root; // kn'th root of unity w'

        // First compute X_i - 1, i = 0,...,kn-1
        for j in 0..target_domain.num_threads {
            let root_shift = multiplicand.pow([(j * target_domain.thread_size) as u64]);
            let mut work_root = self.generator * root_shift; // g.(w')^{j*thread_size}
            let offset = j * target_domain.thread_size;
            for i in offset..offset + target_domain.thread_size {
                l_1_coefficients[i] = work_root - Fr::one(); // (w')^{j*thread_size + i}.g - 1
                work_root *= multiplicand; // (w')^{j*thread_size + i + 1}
            }
        }

        // Compute 1/(X_i - 1) using Montgomery batch inversion
        // Note: This is a placeholder, replace with actual batch invert function.
        // TODO add batch invert
        // invert them all
        let result: Result<(), anyhow::Error> =
            l_1_coefficients.coefficients.iter_mut().try_for_each(|x| {
                let inverse = x
                    .inverse()
                    .ok_or_else(|| anyhow::anyhow!("Failed to find inverse"))?;
                *x = inverse;
                Ok(())
            });
        result?;

        // Step 2: Compute numerator (1/n)*(X_i^n - 1)
        // First compute X_i^n (which forms a multiplicative subgroup of order k)
        let log2_subgroup_size = target_domain.log2_size - self.log2_size; // log_2(k)
        let subgroup_size = 1usize << log2_subgroup_size; // k
        assert!(target_domain.log2_size >= self.log2_size);
        let mut subgroup_roots = vec![Fr::one(); subgroup_size];
        // Note: compute_multiplicative_subgroup function is missing, replace this with your own.
        self.compute_multiplicative_subgroup(log2_subgroup_size, &mut subgroup_roots)?;

        // Subtract 1 and divide by n to get the k elements (1/n)*(X_i^n - 1)
        for root in &mut subgroup_roots {
            *root -= Fr::one();
            *root *= self.domain_inverse;
        }
        // Step 3: Construct L_1(X_i) by multiplying the 1/denominator evaluations in
        // l_1_coefficients by the numerator evaluations in subgroup_roots
        let subgroup_mask = subgroup_size - 1;
        for i in 0..target_domain.num_threads {
            for j in 0..target_domain.thread_size {
                let eval_idx = i * target_domain.thread_size + j;
                l_1_coefficients[eval_idx] *= subgroup_roots[eval_idx & subgroup_mask];
            }
        }
        Ok(())
    }

    /// Computes r = \sum_{i=0}^{num_coeffs-1} (L_{i+1}(ʓ).f_i)
    ///
    /// # Details
    ///
    /// L_i represents ith coefficient of the lagrange polynomial and calculated using first
    /// lagrange coefficient.
    /// `L_1(ʓ)` := (ʓ^n - 1) / n.(ʓ - 1)
    /// `L_i(ʓ)` := L_1(ʓ.ω^{1-i}) = ʓ^n-1 / n.(ʓ.ω^{1-i} - 1)
    ///
    /// # Arguments
    ///
    /// - `coeffs`: f_i, coefficients of the polynomial
    /// - `z`: evaluation point
    /// - `domain`: evaluation domain
    pub(crate) fn compute_barycentric_evaluation(
        &self,
        coeffs: &[Fr],
        num_coeffs: usize,
        z: &Fr,
    ) -> Fr {
        let mut denominators = vec![Fr::default(); num_coeffs];

        let mut numerator = z.pow([self.size as u64]);
        numerator -= Fr::one();
        numerator *= self.domain_inverse; // (ʓ^n - 1) / n

        denominators[0] = *z - Fr::one();
        let mut work_root = self.root_inverse; // ω^{-1}
        for denominator in denominators.iter_mut().take(num_coeffs).skip(1) {
            *denominator = work_root * *z; // denominators[i] will correspond to L_[i+1] (since our 'commented maths' notation indexes
                                           // L_i from 1). So ʓ.ω^{-i} = ʓ.ω^{1-(i+1)} is correct for L_{i+1}.
            *denominator -= Fr::one();
            work_root *= self.root_inverse;
        }

        batch_inversion(denominators.as_mut_slice());

        let mut result = Fr::zero();
        for i in 0..num_coeffs {
            result += coeffs[i] * denominators[i]; // f_i * 1/(ʓ.ω^{-i} - 1)
        }

        result *= numerator; //   \sum_{i=0}^{num_coeffs-1} f_i * [ʓ^n - 1]/[n.(ʓ.ω^{-i} - 1)]
                             // = \sum_{i=0}^{num_coeffs-1} f_i * L_{i+1}
                             // (with our somewhat messy 'commented maths' convention that L_1 corresponds to the 0th coeff).

        result
    }
}

/// Compute the sum of field elements in a given slice.
///
/// This function computes the sum of all elements in the input slice `slice` of type `Fr`. It
/// is useful for computing the sum of coefficients of a polynomial represented in coefficient form.
///
/// # Type Parameters
///
/// * `Fr` - A type that implements the `Field` trait.
///
/// # Arguments
///
/// * `slice` - A slice containing elements of the field `Fr`.
///
/// # Returns
///
/// * A field element of type `Fr` representing the sum of all elements in `slice`.
///
/// # Examples
///
/// ```
/// use ...;  // import the appropriate crate and function
///
/// let coeffs = vec![Fr::from(1), Fr::from(2), Fr::from(3)];  // replace `Fr::from` with the appropriate constructor for your field type
///
/// let sum = compute_sum(&coeffs);
///
/// assert_eq!(sum, Fr::from(6));  // replace `Fr::from` with the appropriate constructor for your field type
/// ```
fn compute_sum<Fr: Field>(slice: &[Fr]) -> Fr {
    slice.iter().copied().fold(Fr::zero(), Add::add)
}

/// Compute the coefficients of a polynomial that is the product of linear polynomials.
///
/// This function computes the coefficients of a polynomial which is the product of linear (degree 1)
/// polynomials with roots specified by `roots`. Each linear polynomial is of the form `(x - root)`,
/// where `root` is an element from the `roots` slice. The resulting polynomial is therefore of the
/// form `(x - root_1) * (x - root_2) * ... * (x - root_n)`.
///
/// # Arguments
///
/// * `roots` - A slice that holds the roots of the linear polynomials.
///
/// * `dest` - A mutable slice where the computed polynomial coefficients will be stored. It should
///   have length `n + 1` where `n` is the number of roots.
///
/// * `n` - The number of roots and hence the degree of the resulting polynomial.
///
/// # Examples
///
/// ```
/// use ...;  // import the appropriate crate and function
///
/// let roots = vec![Fr::from(1), Fr::from(2), Fr::from(3)];  // replace `Fr::from` with the appropriate constructor for your field type
/// let mut dest = vec![Fr; 4];
/// let n = 3;
///
/// compute_linear_polynomial_product(&roots, &mut dest, n);
///
/// assert_eq!(dest, vec![24, -50, 35, 1]); 
/// ```
///
/// # Panics
///
/// This function will panic if the length of `dest` is not `n + 1`.

pub(crate) fn compute_linear_polynomial_product<Fr: Field>(
    roots: &[Fr],
    dest: &mut [Fr],
    n: usize,
) {
    // Equivalent of getting scratch_space
    let mut scratch_space = vec![Fr::default(); n];
    scratch_space.clone_from_slice(&roots[0..n]);

    dest[n] = Fr::one();
    dest[n - 1] = -compute_sum(&scratch_space[..n]);

    let mut temp;
    let mut constant = Fr::one();
    for i in 0..n - 1 {
        temp = Fr::default();
        for j in 0..n - 1 - i {
            scratch_space[j] = roots[j] * compute_sum(&scratch_space[j + 1..n - 1 - i - j]);
            temp += scratch_space[j];
        }
        dest[n - 2 - i] = temp * constant;
        constant = constant.neg();
    }
}

/// Computes efficient interpolation of a polynomial using Lagrange's formula.
///
/// This function computes an efficient interpolation of a polynomial using Lagrange's technique.
/// Given a set of points (x_i, y_i), it computes the function f(X) such that f(x_i) = y_i for all i.
/// The polynomial is computed as a product of linear factors, divided by a denominator computed from the 
/// evaluation points.
///
/// It uses a technique similar to the Kate commitment scheme for dividing by (X - x_i).
///
/// # Type Parameters
///
/// * `Fr` - A field type that implements the `Field` trait. The operations must take no arguments and return an 
///          instance of the same type.
///
/// # Arguments
///
/// * `src` - A slice representing the coefficients of the polynomial (y_i values).
/// * `dest` - A mutable slice where the interpolated polynomial will be stored.
/// * `evaluation_points` - A slice representing the points at which the polynomial is evaluated (x_i values).
/// * `n` - The number of points.
///
/// # Returns
///
/// This function returns a `Result<(), anyhow::Error>`. If the interpolation is successful, it returns `Ok(())`. 
/// If an error occurs (e.g., if it fails to find an inverse during computation), it returns `Err(anyhow::Error)`.
///
/// # Examples
///
/// ```
/// use ...;  // import the appropriate crate and function
///
/// let src = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
/// let evaluation_points = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
/// let mut dest = vec![Fr::zero(); src.len()];
/// let n = src.len();
///
/// compute_efficient_interpolation(&src, &mut dest, &evaluation_points, n).unwrap();
///
/// // `dest` now contains the interpolated polynomial
/// ```
pub(crate) fn compute_efficient_interpolation<Fr: Field>(
    src: &[Fr],
    dest: &mut [Fr],
    evaluation_points: &[Fr],
    n: usize,
) -> anyhow::Result<()> {
    /*
        We use Lagrange technique to compute polynomial interpolation.
        Given: (x_i, y_i) for i ∈ {0, 1, ..., n} =: [n]
        Compute function f(X) such that f(x_i) = y_i for all i ∈ [n].
                   (X - x1)(X - x2)...(X - xn)             (X - x0)(X - x2)...(X - xn)
        F(X) = y0--------------------------------  +  y1----------------------------------  + ...
                 (x0 - x_1)(x0 - x_2)...(x0 - xn)       (x1 - x_0)(x1 - x_2)...(x1 - xn)
        We write this as:
                      [          yi        ]
        F(X) = N(X) * |∑_i --------------- |
                      [     (X - xi) * di  ]
        where:
        N(X) = ∏_{i \in [n]} (X - xi),
        di = ∏_{j != i} (xi - xj)
        For division of N(X) by (X - xi), we use the same trick that was used in compute_opening_polynomial()
        function in the kate commitment scheme.
    */
    let mut numerator_polynomial = vec![Fr::zero(); n + 1];

    compute_linear_polynomial_product(evaluation_points, &mut numerator_polynomial, n);

    let mut roots_and_denominators = vec![Fr::zero(); 2 * n];
    let mut temp_src = vec![Fr::zero(); n];

    for i in 0..n {
        roots_and_denominators[i] = -evaluation_points[i];
        temp_src[i] = src[i];
        dest[i] = Fr::zero();

        // compute constant denominator
        roots_and_denominators[n + i] = Fr::one();
        for j in 0..n {
            if j == i {
                continue;
            }
            roots_and_denominators[n + i] *= evaluation_points[i] - evaluation_points[j];
        }
    }

    // TODO make this a batch_invert
    // invert them all
    let result: Result<(), anyhow::Error> = roots_and_denominators.iter_mut().try_for_each(|x| {
        let inverse = x
            .inverse()
            .ok_or_else(|| anyhow::anyhow!("Failed to find inverse"))?;
        *x = inverse;
        Ok(())
    });
    result?;

    let mut z;
    let mut multiplier;
    let mut temp_dest = vec![Fr::zero(); n];
    for i in 0..n {
        z = roots_and_denominators[i];
        multiplier = temp_src[i] * roots_and_denominators[n + i];
        temp_dest[0] = multiplier * numerator_polynomial[0];
        temp_dest[0] *= z;
        dest[0] += temp_dest[0];
        for j in 1..n {
            temp_dest[j] = multiplier * numerator_polynomial[j] - temp_dest[j - 1];
            temp_dest[j] *= z;
            dest[j] += temp_dest[j];
        }
    }
    Ok(())
}

/// Computes coefficients of opening polynomial in Kate commitments
/// i.e. W(X) = F(X) - F(z) / (X - z)
///
/// # Details
///
/// if `coeffs` represents F(X), we want to compute W(X)
/// where W(X) = F(X) - F(z) / (X - z)
/// i.e. divide by the degree-1 polynomial [-z, 1]
///
/// # Arguments
///
/// - `src`: coeffcients of F(X)
/// - `dest`: coefficients of W(X)
/// - `z`: evaluation point
/// - `n`: polynomial size
pub(crate) fn compute_kate_opening_coefficients<Fr: Field>(
    src: &[Fr],
    dest: &mut [Fr],
    z: &Fr,
    n: usize,
) -> anyhow::Result<Fr> {
    // We assume that the commitment is well-formed and that there is no remainder term.
    // Under these conditions we can perform this polynomial division in linear time with good constants let f = evaluate(src, z, n);
    let f = evaluate(src, z, n);

    let divisor = z
        .neg()
        .inverse()
        .ok_or_else(|| anyhow::anyhow!("Failed to find inverse"))?;

    // we're about to shove these coefficients into a pippenger multi-exponentiation routine, where we need
    // to convert out of montgomery form. So, we can use lazy reduction techniques here without triggering overflows
    dest[0] = src[0] - f;
    dest[0] *= divisor;
    for i in 1..n {
        dest[i] = src[i] - dest[i - 1];
        dest[i] *= divisor;
    }

    Ok(f)
}

/// Evaluates a polynomial at a given point.
///
/// This function evaluates a polynomial represented by `coeffs` at a given point `z`.
/// The computation is parallelized into threads for efficiency. The number of threads is determined by 
/// the `compute_num_threads` function, which is assumed to be appropriately defined elsewhere.
///
/// The polynomial is evaluated using the formula:
/// 
/// f(z) = c0 * z^0 + c1 * z^1 + c2 * z^2 + ... + cn * z^n
///
/// where `ci` are the coefficients of the polynomial.
///
/// # Type Parameters
///
/// * `F` - A field type that implements the `Field` trait. The operations must take no arguments and return an 
///          instance of the same type.
///
/// # Arguments
///
/// * `coeffs` - A slice representing the coefficients of the polynomial.
/// * `z` - A reference to the point at which the polynomial should be evaluated.
/// * `n` - The degree of the polynomial plus one (i.e., the number of coefficients).
///
/// # Returns
///
/// This function returns an instance of type `F`, which is the result of the polynomial evaluation at point `z`.
///
/// # Examples
///
/// ```
/// use ...;  // import the appropriate crate and function
///
/// let coeffs = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
/// let z = Fr::from(5);
/// let n = coeffs.len();
///
/// let result = evaluate(&coeffs, &z, n);
///
/// // `result` is the value of the polynomial at z
/// ```
pub(crate) fn evaluate<F: Field>(coeffs: &[F], z: &F, n: usize) -> F {
    let num_threads = compute_num_threads();
    let range_per_thread = n / num_threads;
    let leftovers = n - (range_per_thread * num_threads);
    let mut evaluations = vec![F::default(); num_threads];
    for (j, eval_j) in evaluations.iter_mut().enumerate().take(num_threads) {
        let mut z_acc = z.pow([(j * range_per_thread) as u64]);
        let offset = j * range_per_thread;
        *eval_j = F::default();
        let end = if j == num_threads - 1 {
            offset + range_per_thread + leftovers
        } else {
            offset + range_per_thread
        };
        for coeffs_i in coeffs.iter().take(end).skip(offset) {
            let work_var = z_acc * coeffs_i;
            *eval_j += work_var;
            z_acc *= z;
        }
    }
    let mut r = F::default();
    for evaluation in evaluations {
        r += evaluation;
    }
    r
}
