const SCALAR_BITS: usize = 127;

pub const fn WNAF_SIZE(x: usize) -> usize {
    (SCALAR_BITS + x - 1) / (x)
}

const fn get_optimal_bucket_width(num_points: usize) -> usize {
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

const fn get_num_buckets(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    return 1 << bits_per_bucket;
}

const fn get_num_rounds(num_points: usize) -> usize {
    let bits_per_bucket = get_optimal_bucket_width(num_points / 2);
    return WNAF_SIZE(bits_per_bucket + 1);
}

const fn get_wnaf_bits_const(scalar: &[u64], bits: usize, bit_position: usize) -> u64 {
    if bits == 0 {
        return 0;
    } else {
        /*
         *  we want to take a 128 bit scalar and shift it down by (bit_position).
         * We then wish to mask out `bits` number of bits.
         * Low limb contains first 64 bits, so we wish to shift this limb by (bit_position mod 64), which is also
         * (bit_position & 63) If we require bits from the high limb, these need to be shifted left, not right. Actual
         * bit position of bit in high limb = `b`. Desired position = 64 - (amount we shifted low limb by) = 64 -
         * (bit_position & 63)
         *
         * So, step 1:
         * get low limb and shift right by (bit_position & 63)
         * get high limb and shift left by (64 - (bit_position & 63))
         *
         */
        let lo_limb_idx = bit_position / 64;
        let hi_limb_idx = (bit_position + bits - 1) / 64;
        let lo_shift = bit_position & 63;
        let bit_mask = (1 << bits) - 1;

        let lo = scalar[lo_limb_idx] >> lo_shift;
        if lo_limb_idx == hi_limb_idx {
            return lo & bit_mask;
        } else {
            let hi_shift = 64 - (bit_position & 63);
            let hi = scalar[hi_limb_idx] << (hi_shift);
            return (lo | hi) & bit_mask;
        }
    }
}

const fn get_wnaf_bits(scalar: &[u64], bits: u64, bit_position: u64) -> u64 {
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
    let lo_limb_idx = bit_position >> 6;
    let hi_limb_idx = (bit_position + bits - 1) >> 6;
    let lo_shift = bit_position & 63;
    let bit_mask = (1 << bits) - 1;

    let lo = scalar[lo_limb_idx] >> lo_shift;
    let hi_shift = if bit_position == 0 {
        0
    } else {
        64 - (bit_position & 63)
    };
    let hi = scalar[hi_limb_idx] << (hi_shift);
    let hi_mask = bit_mask & (0 - (lo_limb_idx != hi_limb_idx));

    return (lo & bit_mask) | (hi & hi_mask);
}

const fn fixed_wnaf_packed(
    scalar: &[u64],
    wnaf: &[u64],
    skew_map: &[bool],
    point_index: u64,
    wnaf_bits: usize,
) {
    unimplemented!()
}

const fn fixed_wnaf(
    scalar: &[u64],
    wnaf: &[u64],
    skew_map: &[bool],
    point_index: u64,
    num_points: u64,
    wnaf_bits: usize,
) {
    unimplemented!()
}

/**
 * Current flow...
 *
 * If a wnaf entry is even, we add +1 to it, and subtract 32 from the previous entry.
 * This works if the previous entry is odd. If we recursively apply this process, starting at the least significant
 *window, this will always be the case.
 *
 * However, we want to skip over windows that are 0, which poses a problem.
 *
 * Scenario 1:  even window followed by 0 window followed by any window 'x'
 *
 *   We can't add 1 to the even window and subtract 32 from the 0 window, as we don't have a bucket that maps to -32
 *   This means that we have to identify whether we are going to borrow 32 from 'x', requiring us to look at least 2
 *steps ahead
 *
 * Scenario 2: <even> <0> <0> <x>
 *
 *   This problem proceeds indefinitely - if we have adjacent 0 windows, we do not know whether we need to track a
 *borrow flag until we identify the next non-zero window
 *
 * Scenario 3: <odd> <0>
 *
 *   This one works...
 *
 * Ok, so we should be a bit more limited with when we don't include window entries.
 * The goal here is to identify short scalars, so we want to identify the most significant non-zero window
 **/

const fn get_num_scalar_bits(scalar: &[u64]) -> u64 {
    unimplemented!()
}

/**
 * How to compute an x-bit wnaf slice?
 *
 * Iterate over number of slices in scalar.
 * For each slice, if slice is even, ADD +1 to current slice and SUBTRACT 2^x from previous slice.
 * (for 1st slice we instead add +1 and set the scalar's 'skew' value to 'true' (i.e. need to subtract 1 from it at the
 * end of our scalar mul algo))
 *
 * In *wnaf we store the following:
 *  1. bits 0-30: ABSOLUTE value of wnaf (i.e. -3 goes to 3)
 *  2. bit 31: 'predicate' bool (i.e. does the wnaf value need to be negated?)
 *  3. bits 32-63: position in a point array that describes the elliptic curve point this wnaf slice is referencing
 *
 * N.B. IN OUR STDLIB ALGORITHMS THE SKEW VALUE REPRESENTS AN ADDITION NOT A SUBTRACTION (i.e. we add +1 at the end of
 * the scalar mul algo we don't sub 1) (this is to eliminate situations which could produce the point at infinity as an
 * output as our circuit logic cannot accomodate this edge case).
 *
 * Credits: Zac W.
 *
 * @param scalar Pointer to the 128-bit non-montgomery scalar that is supposed to be transformed into wnaf
 * @param wnaf Pointer to output array that needs to accomodate enough 64-bit WNAF entries
 * @param skew_map Reference to output skew value, which if true shows that the point should be added once at the end of
 * computation
 * @param wnaf_round_counts Pointer to output array specifying the number of points participating in each round
 * @param point_index The index of the point that should be multiplied by this scalar in the point array
 * @param num_points Total points in the MSM (2*num_initial_points)
 *
 */

fn fixed_wnaf_with_counts(
    scalar: &[u64],
    wnaf: &[u64],
    skew_map: &[bool],
    wnaf_round_counts: &[u64],
    point_index: u64,
    num_points: u64,
    wnaf_bits: usize,
) {
    unimplemented!()
}

const fn wnaf_round(
    scalar: &[u64],
    wnaf: &[u64],
    point_index: usize,
    previous: u64,
    wnaf_bits: usize,
    round_i: usize,
    num_points: usize,
) {
    unimplemented!()
}

const fn wnaf_round_packed(
    scalar: &[u64],
    wnaf: &[u64],
    point_index: usize,
    previous: u64,
    wnaf_bits: usize,
    round_i: usize,
) {
    unimplemented!()
}

const fn wnaf_round_with_restricted_first_slice() {
    unimplemented!()
}

const fn fixed_wnaf_with_restricted_first_slice() {
    unimplemented!()
}
