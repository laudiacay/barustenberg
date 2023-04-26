use crate::numeric::bitop::Msb;

/// This method will compute the number of threads which would be used
/// for computation in barretenberg. We set it to the max number of threads
/// possible for a system (using the openmp package). However, if any system
/// has max number of threads which is NOT a power of two, we set number of threads
/// to be used as the previous power of two.
pub fn compute_num_threads() -> usize {
    #[cfg(feature = "multithreading")]
    // TODO: we used omp_get_max_threads() from openmp in the c++. no idea if this is the same thing- will check later.
    let num_threads: usize = std::thread::available_parallelism().unwrap().get();
    #[cfg(not(feature = "multithreading"))]
    let num_threads: usize = 1;

    // ensure that num_threads is a power of two
    1 << num_threads.get_msb()
}
