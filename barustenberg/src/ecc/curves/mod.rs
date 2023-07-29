use ark_ff::{FftField, Field};

//pub(crate) mod grumpkin;
pub(crate) mod bn254_scalar_multiplication;

pub(crate) fn coset_generator<F: Field + FftField>(_idx: usize) -> F {
    /*
        ASSERT(idx < 7);
    const FieldExt result{
        Params::coset_generators_0[idx],
        Params::coset_generators_1[idx],
        Params::coset_generators_2[idx],
        Params::coset_generators_3[idx],
    };
    return result; */
    unimplemented!("coset_generator")
}

pub(crate) fn external_coset_generator<F: Field + FftField>() -> F {
    unimplemented!("external_coset_generator")
}
