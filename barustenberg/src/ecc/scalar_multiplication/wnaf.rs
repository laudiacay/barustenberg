use crate::numeric::bitop::get_msb::get_msb64;

pub(crate) const SCALAR_BITS: usize = 127;
pub(crate) const fn WNAF_SIZE(x: usize) -> usize {
    (SCALAR_BITS + x - 1) / x
}

#[inline]
pub(crate) fn get_num_scalar_bits(scalar: &u128) -> u64 {
    //Since msb operates on u64 we have to split then recombine and in the process do some fun bit
    //shifting
    let hi: u64 = (scalar >> 64) as u64;
    let lo: u64 = *scalar as u64;

    let msb_1 = get_msb64(hi);
    let msb_0 = get_msb64(lo);

    let scalar_1_mask = 0u64 - (hi > 0) as u64;
    let scalar_0_mask = (0u64 - (lo > 0) as u64) & !scalar_1_mask;

    (scalar_1_mask & (msb_1 + 64)) | (scalar_0_mask & (msb_0))
}

///Returns the 64 bit wnaf value from a scalar value
//
#[inline]
pub(crate) fn get_wnaf_bits<'a>(scalar: &[u64], bits: usize, bit_position: usize) -> u64 {
    /*
     * we want to take a 128 bit scalar and shift it down by (bit_position).
     * We then wish to mask out `bits` number of bits.
     * Low limb contains first 64 bits, so we wish to shift this limb by (bit_position mod 64), which is also
     * (bit_position & 63) If we require bits from the high limb, these need to be shifted left, not right. Actual bit
     * position of bit in high limb = `b`. Desired position = 64 - (amount we shifted low limb by) = 64 - (bit_position
     * & 63)
     *
     * So, step 1:
     * get low limb and shift right by (bit_position & 63)
     * get high limb and shift left by (64 - (bit_position & 63))
     *
     */
    //NOTE: This implementation in C++ computes the index of limb of the u128 scalar of interest
    //scalars are stored as an array of u64 limb values with two representing a u128 of the

    println!("get_wnaf_bits");
    println!("bits: {:x?}, {:?}", bits, bits);
    println!("bit_positon: {:x?}, {:?}", bit_position, bit_position);
    //TODO: eliminate this indexing once everything is worked out
    let lo_limb_idx = bit_position >> 6;
    let hi_limb_idx = (bit_position + bits - 1) >> 6;
    let lo_shift: u64 = (bit_position & 63) as u64;
    let bit_mask: u64 = (1 << bits) - 1;
    println!("bit_mask: {:?}", bit_mask);
    let lo = scalar[lo_limb_idx] >> lo_shift;
    //NOTE: This could be backward
    let hi_shift = if bit_position != 0 {
        64 - (bit_position & 63)
    } else {
        0
    };
    println!("hi_shift: {:x?}", hi_shift);
    let hi: u64 = scalar[hi_limb_idx] << hi_shift;
    let is_same_idx = lo_limb_idx != hi_limb_idx;
    //Note: in C++ implementaion this is done using 0 - 1. Where 1 is the result of the above
    //boolean. This works as unsigned types in c++ use wraparound modular arith for negative
    //numbers. This means that 0 - 1 = U64::max aka every bit is 1 aka use the entire bit mask.
    //
    //Otherwise if false use 0.
    //
    let hi_mask = if is_same_idx == true { bit_mask } else { 0u64 };
    //let hi_mask: u64 = bit_mask & (0u64 - ((lo_limb_idx != hi_limb_idx) as u64));
    println!("hi_mask: {:x?}", hi_mask);
    return (lo & bit_mask) | (hi & hi_mask);
}

pub(crate) fn fixed_wnaf(
    scalar: &mut [u64],
    wnaf: &mut [u64],
    skew_map: &mut bool,
    point_index: u64,
    num_points: usize,
    wnaf_bits: usize,
) {
    println!("fixed wnaf");
    *skew_map = (scalar[0] & 1) == 0;
    let mut previous = get_wnaf_bits(scalar, wnaf_bits, 0) + *skew_map as u64;
    println!("previous: {:?} {:x?}", previous, previous);
    let wnaf_entries = WNAF_SIZE(wnaf_bits);
    println!("wnaf_entries: {:?} {:x?}", wnaf_entries, wnaf_entries);
    for round in 1..(wnaf_entries - 1) {
        let slice = get_wnaf_bits(scalar, wnaf_bits, round * wnaf_bits);
        println!("slice: {:?} {:x?}", slice, slice);

        let predicate = slice & 1;
        println!("predicate: {:?} {:x?}", predicate, predicate);
        wnaf[(wnaf_entries - round) * num_points] =
            ((((previous - (predicate << wnaf_bits)) ^ (0 - predicate)) >> 1) | (predicate << 31))
                | point_index;
        println!(
            "wnaf[wnaf_entries - round]: {:?} {:x?}",
            wnaf[(wnaf_entries - round) * num_points],
            wnaf[(wnaf_entries - round) * num_points]
        );
        previous = slice + predicate;
        println!("previous: {:?} {:x?}", previous, previous);
    }

    let final_bits = SCALAR_BITS - (wnaf_bits * (wnaf_entries - 1));
    println!("final_bits: {:?} {:x?}", final_bits, final_bits);
    let slice = get_wnaf_bits(scalar, final_bits, (wnaf_entries - 1) * wnaf_bits);
    println!("slice: {:?} {:x?}", slice, slice);
    let predicate = slice & 1;
    println!("predicate: {:?} {:x?}", predicate, predicate);
    wnaf[num_points] = ((((previous - (predicate << wnaf_bits)) ^ (0 - predicate)) >> 1)
        | (predicate << 31))
        | point_index;
    println!(
        "wnaf[num_points]: {:?} {:x?}",
        wnaf[num_points], wnaf[num_points]
    );
    wnaf[0] = ((slice + predicate) >> 1) | point_index;
    println!("wnaf[0]: {:?} {:x?}", wnaf[0], wnaf[0]);
}

pub(crate) fn fixed_wnaf_with_counts(
    scalar: &mut u128,
    wnaf: &mut [u64],
    skew_map: &mut bool,
    wnaf_round_counts: &mut [u64],
    point_index: u64,
    num_points: usize,
    wnaf_bits: usize,
) {
    let max_wnaf_entries = (SCALAR_BITS + wnaf_bits - 1) / wnaf_bits;

    //If the scalar is 0
    if *scalar == 0 {
        *skew_map = false;
        for round in 0..max_wnaf_entries {
            wnaf[round * num_points] = u64::MAX;
        }
        return;
    }

    let current_scalar_bits = get_num_scalar_bits(scalar) + 1;
    let hi: u64 = (*scalar >> 64) as u64;
    let lo: u64 = *scalar as u64;

    //If the first bit of scalars hi limb is set set its wnaf skew to 1
    *skew_map = (hi & 1) == 0;
    let mut previous = get_wnaf_bits(&[hi, lo], wnaf_bits, 0) + *skew_map as u64;
    let wnaf_entries = (current_scalar_bits as usize + wnaf_bits - 1) / wnaf_bits;

    //If there is 1 window
    if wnaf_entries == 1 {
        wnaf[(max_wnaf_entries - 1) * num_points] = (previous >> 1) | point_index;
        wnaf_round_counts[max_wnaf_entries - 1] += 1;
        for j in wnaf_entries..max_wnaf_entries {
            wnaf[(max_wnaf_entries - 1 - j) * num_points] = u64::MAX;
        }
        return;
    }

    //If there are several windows
    for round in 1..(wnaf_entries - 1) {
        // Get wnaf bit slice of a scalar for the window
        let slice = get_wnaf_bits(&[hi, lo], wnaf_bits, round * wnaf_bits);

        // Get the predicate of the slice -> Whether its last bit is zero
        let predicate = ((slice & 1) == 0) as u64;

        // Update round count
        wnaf_round_counts[max_wnaf_entries - round] += 1;

        // Compute wnaf entry value
        // If the last bit of current slice is 1, we simply put the previous value with the point index
        // If the last bit of the current slice is 0, we negate everything, so that we subtract from the WNAF form and
        // make it 0
        // NOTE: Not sure what the commented out 1 is for but it was in the C++ implementation
        wnaf[(max_wnaf_entries - round) * num_points] =
            ((((previous - (predicate << (wnaf_bits/*+ 1*/))) ^ (0 - predicate)) >> 1)
                | (predicate << 31))
                | (point_index);

        // Update the previous value to the next windows
        previous = slice + predicate;
    }

    // Final iteration for top bits
    let final_bits = current_scalar_bits as usize - (wnaf_bits * (wnaf_entries - 1));
    let slice = get_wnaf_bits(&[hi, lo], final_bits, (wnaf_entries - 1) * wnaf_bits);
    let predicate = ((slice & 1) == 0) as u64;

    wnaf_round_counts[(max_wnaf_entries - wnaf_entries + 1)] += 1;
    wnaf[((max_wnaf_entries - wnaf_entries + 1) * num_points)] =
        ((((previous - (predicate << (wnaf_bits/*+ 1*/))) ^ (0 - predicate)) >> 1)
            | (predicate << 31))
            | (point_index);

    // Saving top bits
    wnaf_round_counts[max_wnaf_entries - wnaf_entries] += 1;
    wnaf[(max_wnaf_entries - wnaf_entries) * num_points] =
        ((slice + predicate) >> 1) | (point_index);

    // Fill all unused slots with -1
    for j in wnaf_entries..max_wnaf_entries {
        wnaf[(max_wnaf_entries - 1 - j) * num_points] = u64::MAX;
    }
}

