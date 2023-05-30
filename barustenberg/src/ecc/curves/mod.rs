//pub(crate) mod grumpkin;

use ark_ff::Field;

pub(crate) fn coset_generator<F: Field>(idx: usize) -> F {
    /*
        ASSERT(idx < 7);
    const field result{
        Params::coset_generators_0[idx],
        Params::coset_generators_1[idx],
        Params::coset_generators_2[idx],
        Params::coset_generators_3[idx],
    };
    return result; */
    unimplemented!("coset_generator")
}

pub(crate) fn external_coset_generator<F: Field>() -> F {
    unimplemented!("external_coset_generator")
}
