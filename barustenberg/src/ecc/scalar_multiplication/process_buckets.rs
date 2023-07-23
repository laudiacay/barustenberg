// use ark_bn254::{G1Affine, G1Projective};
use std::mem;

const NUM_BITS: usize = 8;
// From what I've seen UL == u64
const NUM_BUCKETS: usize = (1u64 << NUM_BITS) as usize;
const MASK: u64 = NUM_BUCKETS as u64 - 1u64;

pub(crate) const fn get_optimal_bucket_width(num_points: usize) -> usize {
    if num_points >= 14617149 {
        return 21;
    }
    if num_points >= 1139094 {
        return 18;
    }
    // if (num_points >= 100000)
    if num_points >= 155975 {
        return 15;
    }
    if num_points >= 144834
    // if (num_points >= 100000)
    {
        return 14;
    }
    if num_points >= 25067 {
        return 12;
    }
    if num_points >= 13926 {
        return 11;
    }
    if num_points >= 7659 {
        return 10;
    }
    if num_points >= 2436 {
        return 9;
    }
    if num_points >= 376 {
        return 7;
    }
    if num_points >= 231 {
        return 6;
    }
    if num_points >= 97 {
        return 5;
    }
    if num_points >= 35 {
        return 4;
    }
    if num_points >= 10 {
        return 3;
    }
    if num_points >= 2 {
        return 2;
    }
    return 1;
}
const SCALAR_BITS: usize = 127;
const fn wnaf_size(bits: usize) -> usize {
    SCALAR_BITS + bits - 1 / bits
}

pub(crate) const fn get_num_buckets(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    return (1u64 << bits_per_bucket) as usize;
}

pub(crate) const fn get_num_rounds(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    wnaf_size(bits_per_bucket + 1)
}

//TLDR: For each round we preprocess which keys go into which bucket
//we iterate over each glv decomposed scalar and encode it as a number of wnaf keys. This contains 31 bytes of the scalar so
//we can slice it to grab the bucket key and the last 32 bytes represent its naf form for
//optimization purposes
//
//
//TODO: Move this into Pippenger runtime as a assc function
//TODO: Make this more idiomatic maybe???
pub(crate) fn radix_sort<'a>(keys: &'a mut [u64], num_entries: usize, shift: usize) {
    let mut bucket_counts = [0; NUM_BUCKETS];

    //Store counts of occurences of keys
    for i in 0..num_entries {
        //We get the key for the bucket by right shifting and mask the 31 LSD og the u64 wnaf
        //encoded key. When then cast to usize to index the correct bucket count and increment the
        //number of wnaf encoded point keys in that bucket
        bucket_counts[((keys[i] >> shift) & MASK) as usize] += 1;
    }

    let mut offsets = [0usize; NUM_BUCKETS + 1];
    offsets[0] = 0;

    //Change bucket_counts[i] so that count[i] now contains actual position of this bucket
    for i in 0..NUM_BUCKETS - 1 {
        bucket_counts[i + 1] += bucket_counts[i];
    }

    //copy positions of the bucket to offsets.
    for i in 1..NUM_BUCKETS + 1 {
        offsets[i] = bucket_counts[i - 1];
    }
    let offsets_copy = offsets.clone();

    //Not sure if the indexing is correct
    for i in 0..NUM_BUCKETS {
        let mut bucket_start = &keys[offsets[i] as usize];
        let bucket_end = &keys[offsets[i + 1] as usize];
        while bucket_start != bucket_end {
            for mut it in *bucket_start..*bucket_end {
                let value = ((it >> shift) & MASK) as usize;
                let offset = offsets[value];
                offsets[value] += 1;
                //Instead of grabbing the pointer &keys[0] we just normal array access
                mem::swap(&mut it, &mut keys[offset]);
            }
            bucket_start = &keys[offsets[i]];
        }
    }

    if shift > 0 {
        for i in 0..NUM_BUCKETS {
            if (offsets_copy[i + 1] - offsets_copy[i]) > 1 {
                radix_sort(
                    &mut keys[offsets_copy[i]..],
                    (offsets[i + 1] - offsets[i]) as usize,
                    shift - 8,
                );
            }
        }
    }
}

pub(crate) fn process_buckets<'a>(wnaf_entries: &'a mut [u64], num_entries: usize, num_bits: u32) {
    const BITS_PER_ROUND: usize = 8;
    const BASE: usize = NUM_BITS & 7;
    const TOTAL_BITS: usize = if BASE == 0 {
        NUM_BITS
    } else {
        NUM_BITS - BASE + 8
    };
    let shift = TOTAL_BITS - BITS_PER_ROUND;

    radix_sort(wnaf_entries, num_entries, shift);
}
