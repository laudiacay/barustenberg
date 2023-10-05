use std::vec::Vec;

pub(crate) fn merkle_damgard_compress(_inputs: Vec<grumpkin::Fq>, _iv: usize) -> grumpkin::Affine {
    // TODO: Implement this function
    todo!("merkle_damgard_compress with single iv")
}

pub(crate) fn merkle_damgard_compress_with_multiple_ivs(
    _inputs: Vec<grumpkin::Fq>,
    _ivs: Vec<usize>,
) -> grumpkin::Affine {
    // TODO: Implement this function
    todo!("merkle_damgard_compress with multiple ivs")
}

pub(crate) fn merkle_damgard_tree_compress(
    _inputs: Vec<grumpkin::Fq>,
    _ivs: Vec<usize>,
) -> grumpkin::Affine {
    // TODO: Implement this function
    todo!("merkle_damgard_tree_compress")
}

pub(crate) fn compress_native_index(
    _inputs: Vec<grumpkin::Fq>,
    _hash_index: Option<usize>,
) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native with single hash index")
}

pub(crate) fn compress_native_with_multiple_indices(
    _inputs: Vec<grumpkin::Fq>,
    _hash_indices: Vec<usize>,
) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native with multiple hash indices")
}

pub(crate) fn compress_native(_input: &[u8]) -> Vec<u8> {
    // TODO: Implement this function
    todo!("compress_native")
}

pub(crate) fn compress_native_buffer_to_field(_input: Vec<u8>) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native_buffer_to_field")
}

pub(crate) fn compress_native_array<const T: usize>(_inputs: [grumpkin::Fq; T]) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native_array")
}

pub(crate) fn commit_native(_inputs: Vec<grumpkin::Fq>, _hash_index: usize) -> grumpkin::Affine {
    // TODO: Implement this function
    todo!("commit_native with single hash index")
}

pub(crate) fn commit_native_with_multiple_indices(
    _inputs: Vec<grumpkin::Fq>,
    _hash_indices: Vec<usize>,
) -> grumpkin::Affine {
    // TODO: Implement this function
    todo!("commit_native with multiple hash indices")
}
