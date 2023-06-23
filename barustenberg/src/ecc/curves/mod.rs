//pub(crate) mod grumpkin;

use super::fieldext::FieldExt;

pub(crate) fn coset_generator<F: FieldExt>(idx: usize) -> F {
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

pub(crate) fn external_coset_generator<F: FieldExt>() -> F {
    unimplemented!("external_coset_generator")
}
