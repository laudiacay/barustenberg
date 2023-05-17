use crate::{
    ecc::{
        curves::bn254::fr::Fr,
        fields::field::{Field, FieldParams},
    },
    numeric::bitop::Msb,
};
use std::vec::Vec;

pub const MIN_GROUP_PER_THREAD: usize = 4;

pub struct EvaluationDomain<F: FieldParams> {
    /// n, always a power of 2
    pub size: usize,
    /// num_threads * thread_size = size
    num_threads: usize,
    thread_size: usize,
    log2_size: usize,
    log2_thread_size: usize,
    log2_num_threads: usize,
    generator_size: usize,
    /// omega; the nth root of unity
    root: Field<F>,
    /// omega^{-1}
    root_inverse: Field<F>,
    /// n; same as size
    domain: Field<F>,
    /// n^{-1}
    domain_inverse: Field<F>,
    generator: Field<F>,
    generator_inverse: Field<F>,
    four_inverse: Field<F>,
    /// An entry for each of the log(n) rounds: each entry is a pointer to
    /// the subset of the roots of unity required for that fft round.
    /// E.g. round_roots[0] = [1, ω^(n/2 - 1)],
    ///      round_roots[1] = [1, ω^(n/4 - 1), ω^(n/2 - 1), ω^(3n/4 - 1)]
    ///      ...
    round_roots: Vec<Vec<Field<F>>>,
    inverse_round_roots: Vec<Vec<Field<F>>>,
}

fn compute_num_threads(size: usize) -> usize {
    #[cfg(feature = "multithreading")]
    let num_threads = crate::common::max_threads::compute_num_threads();
    #[cfg(not(feature = "multithreading"))]
    let num_threads = 1;
    if size <= num_threads * MIN_GROUP_PER_THREAD {
        return 1;
    }
    return num_threads;
}

fn compute_lookup_table_single<F: FieldParams>(
    input_root: &Field<F>,
    size: usize,
    roots: &[Field<F>],
    round_roots: &mut Vec<&mut [Field<F>]>,
) {
    todo!("unimplemented, see comment below");
    // ORIGINAL CODE:
    /*
    void compute_lookup_table_single(const Fr& input_root,
                                     const size_t size,
                                     Fr* const roots,
                                     std::vector<Fr*>& round_roots)
    {
        const size_t num_rounds = static_cast<size_t>(numeric::get_msb(size));

        round_roots.emplace_back(&roots[0]);
        for (size_t i = 1; i < num_rounds - 1; ++i) {
            round_roots.emplace_back(round_roots.back() + (1UL << i));
        }

        for (size_t i = 0; i < num_rounds - 1; ++i) {
            const size_t m = 1UL << (i + 1);
            const Fr round_root = input_root.pow(static_cast<uint64_t>(size / (2 * m)));
            Fr* const current_round_roots = round_roots[i];
            current_round_roots[0] = Fr::one();
            for (size_t j = 1; j < m; ++j) {
                current_round_roots[j] = current_round_roots[j - 1] * round_root;
            }
        }
    }
     */

    // MAYBE a solution- from chatgpt
    // let num_rounds = size.get_msb();

    // round_roots.push(&mut roots[0..1]);
    // for i in 1..(num_rounds - 1) {
    //     let prev_round_roots = round_roots.last().unwrap();
    //     let next_start = prev_round_roots.as_ptr() as usize + (1 << i) * std::mem::size_of::<Fr>();
    //     let next_round_roots =
    //         unsafe { std::slice::from_raw_parts_mut(next_start as *mut Fr, 1 << i) };
    //     round_roots.push(next_round_roots);
    // }

    // for i in 0..(num_rounds - 1) {
    //     let m = 1 << (i + 1);
    //     let round_root = input_root.pow((size / (2 * m)) as u64);
    //     let current_round_roots = round_roots[i];
    //     current_round_roots[0] = Fr::one();
    //     for j in 1..m {
    //         current_round_roots[j] = current_round_roots[j - 1] * round_root;
    //     }
    // }
}

impl<F: FieldParams> EvaluationDomain<F> {
    pub fn new(domain_size: usize, target_generator_size: Option<usize>) -> Self {
        // TODO: implement constructor logic

        let size = domain_size;
        let num_threads = compute_num_threads(size);
        let thread_size = size / num_threads;
        let log2_size = size.get_msb();
        let log2_thread_size = thread_size.get_msb();
        let log2_num_threads = num_threads.get_msb();
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

    pub fn compute_lookup_table(&mut self) {
        // TODO: implement compute_lookup_table logic
    }

    pub fn compute_generator_table(&mut self, target_generator_size: usize) {
        // TODO: implement compute_generator_table logic
    }

    pub fn get_round_roots(&self) -> &Vec<Vec<F>> {
        &self.round_roots
    }

    pub fn get_inverse_round_roots(&self) -> &Vec<Vec<F>> {
        &self.inverse_round_roots
    }
}

pub type BarretenbergEvaluationDomain = EvaluationDomain<Fr>;
pub type GrumpkinEvaluationDomain = EvaluationDomain<crate::ecc::curves::grumpkin::Fr>;
