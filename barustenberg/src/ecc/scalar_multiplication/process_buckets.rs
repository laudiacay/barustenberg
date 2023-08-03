use std::mem;

const NUM_BITS: usize = 8;
// From what I've seen UL == u64
const NUM_BUCKETS: usize = (1u64 << NUM_BITS) as usize;
const MASK: u64 = NUM_BUCKETS as u64 - 1u64;

//TLDR: For each round we preprocess which keys go into which bucket
//we iterate over each glv decomposed scalar and encode it as a number of wnaf keys. This contains 31 bytes of the scalar so
//we can slice it to grab the bucket key and the last 32 bytes represent its naf form for
//optimization purposes
//
//
//TODO: Move this into Pippenger runtime as a assc function
//TODO: Make this more idiomatic maybe???
fn radix_sort(keys: &mut [u64], num_entries: usize, shift: usize) {
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

pub(crate) fn process_buckets(wnaf_entries: &mut [u64], num_entries: usize, num_bits: u32) {
    const BITS_PER_ROUND: usize = 8;
    let base: usize = (num_bits as usize) & 7;
    let total_bits: usize = if base == 0 {
        num_bits as usize
    } else {
        (num_bits as usize) - base + 8
    };
    let shift = total_bits - BITS_PER_ROUND;

    radix_sort(wnaf_entries, num_entries, shift);
}
