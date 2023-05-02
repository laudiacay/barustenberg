pub mod polynomial_arithmetic {
    use lazy_static::lazy_static;
    use std::sync::Mutex;

    use crate::{
        ecc::curves::bn254::Fr,
        numeric::{self, bitop::Msb},
    }; // NOTE: This might not be the right Fr, need to check vs gumpkin
    struct ScratchSpace<T> {
        working_memory: Mutex<Option<Vec<T>>>,
    }

    impl<T> ScratchSpace<T> {
        pub fn get_scratch_space(&self, num_elements: usize) -> &mut [T] {
            let mut working_memory = self.working_memory.lock().unwrap();
            let current_size = working_memory.as_ref().map(|v| v.len()).unwrap_or(0);

            if num_elements > current_size {
                *working_memory = Some(Vec::with_capacity(num_elements));
            }

            working_memory.as_mut().unwrap().as_mut_slice()
        }
    }

    lazy_static! {
        static ref SCRATCH_SPACE: ScratchSpace<Fr> = ScratchSpace {
            working_memory: Mutex::new(None),
        };
    }

    pub fn get_scratch_space(num_elements: usize) -> &'static mut [Fr] {
        SCRATCH_SPACE.get_scratch_space(num_elements)
    }

    #[inline]
    fn reverse_bits(x: u32, bit_length: u32) -> u32 {
        let x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
        let x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
        let x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
        let x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
        ((x >> 16) | (x << 16)) >> (32 - bit_length)
    }
    #[inline]
    fn is_power_of_two(x: u64) -> bool {
        x != 0 && (x & (x - 1)) == 0
    }

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
            for i in num_src_coefficients..num_target_coefficients {
                dest[i] = Fr::default();
            }
        }
    }

    use std::ops::{Add, Mul, Sub};

    fn fft_inner_serial<
        Fr: Copy + Default + Add<Output = Fr> + Sub<Output = Fr> + Mul<Output = Fr>,
    >(
        coeffs: &mut [Vec<Fr>],
        domain_size: usize,
        root_table: &[Vec<Fr>],
    ) {
        // Assert that the number of polynomials is a power of two.
        let num_polys = coeffs.len();
        assert!(is_power_of_two(num_polys));
        let poly_domain_size = domain_size / num_polys;
        assert!(is_power_of_two(poly_domain_size));

        // TODO Implement the msb from numeric/bitop/get_msb.cpp
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

        for l in 0..num_polys {
            for k in (0..poly_domain_size).step_by(2) {
                let temp = coeffs[l][k + 1];
                coeffs[l][k + 1] = coeffs[l][k] - coeffs[l][k + 1];
                coeffs[l][k] = coeffs[l][k] + temp;
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
                    coeffs[odd_poly_idx][odd_elem_idx] =
                        coeffs[even_poly_idx][even_elem_idx] - temp;
                    coeffs[even_poly_idx][even_elem_idx] =
                        coeffs[even_poly_idx][even_elem_idx] + temp;
                }
            }
        }
    }

    pub struct EvaluationDomain<Fr> {
        pub num_threads: usize,
        pub size: usize,
        pub log2_size: usize,
        pub roots: Vec<Fr>,
        pub inverse_roots: Vec<Fr>,
    }

    fn scale_by_generator<Fr: Copy + Mul<Output = Fr>>(
        coeffs: &[Fr],
        target: &mut [Fr],
        domain: &EvaluationDomain<Fr>,
        generator_start: Fr,
        generator_shift: Fr,
        generator_size: usize,
    ) {
        let generator_size_per_thread = generator_size / domain.num_threads;

        target
            .par_chunks_mut(generator_size_per_thread)
            .enumerate()
            .for_each(|(j, chunk)| {
                let thread_shift = generator_shift.pow(j as u64 * generator_size_per_thread as u64);
                let mut work_generator = generator_start * thread_shift;

                for (i, coeff) in chunk.iter_mut().enumerate() {
                    let index = j * generator_size_per_thread + i;
                    *coeff = coeffs[index] * work_generator;
                    work_generator = work_generator * generator_shift;
                }
            });
    }

    /// Compute multiplicative subgroup (g.X)^n.
    /// Compute the subgroup for X in roots of unity of (2^log2_subgroup_size)*n.
    /// X^n will loop through roots of unity (2^log2_subgroup_size).
    /// @param log2_subgroup_size Log_2 of the subgroup size.
    /// @param src_domain The domain of size n.
    /// @param subgroup_roots Pointer to the array for saving subgroup members.

    pub trait FieldElement: Sized + Copy + Mul<Output = Self> {
        fn get_root_of_unity(log2_subgroup_size: usize) -> Self;
        fn self_sqr(&mut self);
    }

    fn compute_multiplicative_subgroup<Fr: FieldElement>(
        log2_subgroup_size: usize,
        src_domain: &EvaluationDomain<Fr>,
        subgroup_roots: &mut [Fr],
    ) {
        let subgroup_size = 1 << log2_subgroup_size;

        // Step 1: get primitive 4th root of unity
        let subgroup_root = Fr::get_root_of_unity(log2_subgroup_size);

        // Step 2: compute the cofactor term g^n
        let mut accumulator = src_domain.generator;
        for _ in 0..src_domain.log2_size {
            accumulator.self_sqr();
        }

        // Step 3: fill array with subgroup_size values of (g.X)^n, scaled by the cofactor
        subgroup_roots[0] = accumulator;
        for i in 1..subgroup_size {
            subgroup_roots[i] = subgroup_roots[i - 1] * subgroup_root;
        }
    }

    fn fft_inner_parallel<Fr: FieldElement>(
        coeffs: &mut Vec<&mut [Fr]>,
        domain: &EvaluationDomain<Fr>,
        _: &Fr,
        root_table: &Vec<&[Fr]>,
    ) {
        let scratch_space = get_scratch_space(domain.size); // Implement the get_scratch_space function

        let num_polys = coeffs.len();
        assert!(num_polys.is_power_of_two());
        let poly_size = domain.size / num_polys;
        assert!(poly_size.is_power_of_two());
        let poly_mask = poly_size - 1;
        let log2_poly_size = poly_size.trailing_zeros() as usize;

        // First FFT round is a special case - no need to multiply by root table, because all entries are 1.
        // We also combine the bit reversal step into the first round, to avoid a redundant round of copying data
        for j in 0..domain.num_threads {
            let mut temp_1 = Fr::__copy(coeffs[0][0]); // Just initializing with an element, any element will do
            let mut temp_2 = temp_1;
            for i in (j * domain.thread_size..(j + 1) * domain.thread_size).step_by(2) {
                let next_index_1 = reverse_bits((i + 2) as u32, domain.log2_size as u32) as usize;
                let next_index_2 = reverse_bits((i + 3) as u32, domain.log2_size as u32) as usize;

                let swap_index_1 = reverse_bits(i as u32, domain.log2_size as u32) as usize;
                let swap_index_2 = reverse_bits((i + 1) as u32, domain.log2_size as u32) as usize;

                let poly_idx_1 = swap_index_1 >> log2_poly_size;
                let elem_idx_1 = swap_index_1 & poly_mask;
                let poly_idx_2 = swap_index_2 >> log2_poly_size;
                let elem_idx_2 = swap_index_2 & poly_mask;

                temp_1 = Fr::__copy(coeffs[poly_idx_1][elem_idx_1]);
                temp_2 = Fr::__copy(coeffs[poly_idx_2][elem_idx_2]);
                scratch_space[i + 1] = temp_1 - temp_2;
                scratch_space[i] = temp_1 + temp_2;
            }
        }

        // hard code exception for when the domain size is tiny - we won't execute the next loop, so need to manually
        // reduce + copy
        if domain.size <= 2 {
            coeffs[0][0] = scratch_space[0];
            coeffs[0][1] = scratch_space[1];
        }
        // Outer FFT loop - iterates over the FFT rounds
        for m in (2..=domain.size).step_by(2) {
            for j in 0..domain.num_threads {
                let mut temp: Fr;

                // Ok! So, what's going on here? This is the inner loop of the FFT algorithm, and we want to break it
                // out into multiple independent threads. For `num_threads`, each thread will evaluation `domain.size /
                // num_threads` of the polynomial. The actual iteration length will be half of this, because we leverage
                // the fact that \omega^{n/2} = -\omega (where \omega is a root of unity)

                // Here, `start` and `end` are used as our iterator limits, so that we can use our iterator `i` to
                // directly access the roots of unity lookup table
                let start = j * (domain.thread_size >> 1);
                let end = (j + 1) * (domain.thread_size >> 1);

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

                let round_roots = &root_table[((m.trailing_zeros() - 1) as usize)];

                // Finally, we want to treat the final round differently from the others,
                // so that we can reduce out of our 'coarse' reduction and store the output in `coeffs` instead of
                // `scratch_space`

                if m != (domain.size >> 1) {
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
}
