use std::vec::Vec;
use ark_ec::short_weierstrass::Affine;

pub(crate) type GeneratorIndexT = usize;

pub(crate) fn commit_single(input: grumpkin::Fq, index: GeneratorIndexT) -> grumpkin::SWAffine {
    // TODO: Implement this functio§n
    let gen_data = get_generator_data(index);
    let mut scalar_multiplier = input.from_montgomery_form();

    const NUM_BITS: usize = 254;
    const NUM_QUADS_BASE: usize = (NUM_BITS - 1) >> 1;
    const NUM_QUADS: usize = if (NUM_QUADS_BASE << 1) + 1 < NUM_BITS { NUM_QUADS_BASE + 1 } else { NUM_QUADS_BASE };
    const NUM_WNAF_BITS: usize = (NUM_QUADS << 1) + 1;

    let ladder = gen_data.get_hash_ladder(NUM_BITS);

    let mut wnaf_entries = [0u64; NUM_QUADS + 2];
    let mut skew = false;
    fixed_wnaf::<NUM_WNAF_BITS, 1, 2>(&mut scalar_multiplier.data[0], &mut wnaf_entries[0], &mut skew, 0);

    let mut accumulator = grumpkin::SWAffine::from(ladder[0].one);
    if skew {
        accumulator.sub_assign(gen_data.skew_generator);
    }

    for i in 0..NUM_QUADS {
        let entry = wnaf_entries[i + 1];
        let point_to_add = if (entry & WNAF_MASK) == 1 { 
            ladder[i + 1].three 
        } else { 
            ladder[i + 1].one 
        };
        let predicate = (entry >> 31) & 1;
        accumulator.self_mixed_add_or_sub(point_to_add, predicate);
    }
    accumulator
    todo!("Need to implement generator functions")

}

pub(crate) fn commit_native(
    inputs: Vec<grumpkin::Fq>,
    hash_index: Option<usize>,
) -> grumpkin::SWAffine {
    let hash_index = hash_index.unwrap_or(0);
    assert!()

    todo!("commit_native")
}

pub(crate) fn commit_native_with_pairs(
    _input_pairs: Vec<(grumpkin::Fq, GeneratorIndexT)>,
) -> grumpkin::SWAffine {
    // TODO: Implement this function
    todo!("commit_native_with_pairs")
}

pub(crate) fn compress_native_with_index(
    inputs: Vec<grumpkin::Fq>,
    hash_index: Option<usize>,
) -> grumpkin::Fq {
    commit_native(inputs, hash_index)
}

pub(crate) fn compress_native_array<const T: usize>(_inputs: [grumpkin::Fq; T]) -> grumpkin::Fq {
    // TODO: Implement this function
    commit_native(inputs, hash_index)
}

pub(crate) fn compress_native(_input: &[grumpkin::Fq]) -> Vec<u8> {
    // TODO: Implement this function
    commit_native(inputs, hash_index)
}

pub(crate) fn compress_native_with_pairs(
    _input_pairs: Vec<(grumpkin::Fq, GeneratorIndexT)>,
) -> grumpkin::Fq {
    // TODO: Implement this function
    todo!("compress_native_with_pairs")
}
/**
 * Given an arbitrary length of bytes, convert them to fields and compress the result using the default generators.
 */
pub(crate) fn compress_native_buffer_to_field(input: &[u8], hash_index: usize) -> grumpkin::Fq
{
    todo!()
}