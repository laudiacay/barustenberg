pub(crate) mod lookup;
use std::vec::Vec;

pub(crate) type GeneratorIndexT = usize;

pub(crate) fn commit_single(_in_value: ark_bn254::Fr, _index: GeneratorIndexT) -> grumpkin::Affine {
    // TODO: Implement this function
    todo!("commit_single")
}

pub(crate) fn commit_native(
    _inputs: Vec<grumpkin::Fq>,
    hash_index: Option<usize>,
) -> grumpkin::Affine {
    let _hash_index = hash_index.unwrap_or(0);

    // TODO: Implement this function
    todo!("commit_native")
}

pub(crate) fn commit_native_with_pairs(
    _input_pairs: Vec<(grumpkin::Fq, GeneratorIndexT)>,
) -> grumpkin::Affine {
    // TODO: Implement this function
    todo!("commit_native_with_pairs")
}

pub(crate) fn compress_native_with_index(
    _inputs: Vec<grumpkin::Fq>,
    hash_index: Option<usize>,
) -> grumpkin::Fq {
    let _hash_index = hash_index.unwrap_or(0);
    // TODO: Implement this function
    todo!("compress_native")
}

pub(crate) fn compress_native_array<const T: usize>(_inputs: [grumpkin::Fq; T]) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native_array")
}

pub(crate) fn compress_native(_input: &[u8]) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native")
}

pub(crate) fn compress_native_with_pairs(
    _input_pairs: Vec<(grumpkin::Fq, GeneratorIndexT)>,
) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native_with_pairs")
}
