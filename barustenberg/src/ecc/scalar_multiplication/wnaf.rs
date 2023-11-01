use crate::numeric::bitop::get_msb;
use get_msb::Msb;

pub(crate) const SCALAR_BITS: usize = 127;
pub(crate) const fn wnaf_size(x: usize) -> usize {
    (SCALAR_BITS + x - 1) / x
}

#[inline]
pub(crate) fn get_num_scalar_bits(scalar: &[u64]) -> u64 {
    //Since msb operates on u64 we have to split then recombine and in the process do some fun bit
    //shifting
    let hi: u64 = scalar[1];
    let lo: u64 = scalar[0];

    let msb_1 = hi.get_msb();
    let msb_0 = lo.get_msb();

    let scalar_1_mask = 0u64 - (hi > 0) as u64;
    let scalar_0_mask = (0u64 - (lo > 0) as u64) & !scalar_1_mask;

    (scalar_1_mask & (msb_1 + 64)) | (scalar_0_mask & (msb_0))
}

///Returns the 64 bit wnaf value from a scalar value
//
#[inline]
pub(crate) fn get_wnaf_bits(scalar: &[u64], bits: usize, bit_position: usize) -> u64 {
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
    println!(
        "bit_mask: {:?}, lo_limb_idx: {}, hi_limb_idx: {}",
        bit_mask, lo_limb_idx, hi_limb_idx
    );
    let lo = scalar[lo_limb_idx] >> lo_shift;
    //NOTE: This could be backward
    let hi_shift = if bit_position != 0 {
        64 - (bit_position & 63)
    } else {
        0
    };
    println!("hi_shift: {:x?}", hi_shift);
    let mut hi = 0;
    if hi_shift == 64 {
        hi = 0;
    } else {
        hi = scalar[hi_limb_idx] << hi_shift as u64;
    }

    let hi_mask = if lo_limb_idx != hi_limb_idx {
        bit_mask
    } else {
        0u64
    };
    //let hi_mask: u64 = bit_mask & (0u64 - ((lo_limb_idx != hi_limb_idx) as u64));
    println!("hi_mask: {:x?}", hi_mask);
    (lo & bit_mask) | (hi & hi_mask)
}

pub(crate) fn fixed_wnaf<const S: usize>(
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
    let wnaf_entries = (S + wnaf_bits - 1) / wnaf_bits;
    println!("wnaf_entries: {:?} {:x?}", wnaf_entries, wnaf_entries);
    for round in 1..(wnaf_entries - 1) {
        let slice = get_wnaf_bits(scalar, wnaf_bits, round * wnaf_bits);
        println!("slice: {:?} {:x?}", slice, slice);

        let predicate = ((slice & 1) == 0) as u64;
        println!(
            "previous: {}, predicate: {:?} {:x?}",
            previous, predicate, predicate
        );
        let negation: u64 = (u64::MAX - predicate) % u64::MAX;
        let previous_neg = ((previous as u128 + u64::MAX as u128
            - (predicate << wnaf_bits) as u128)
            % u64::MAX as u128) as u64;
        wnaf[(wnaf_entries - round) * num_points] =
            (((previous_neg ^ negation) >> 1) | (predicate << 31)) | point_index;
        println!(
            "round: {}, wnaf[wnaf_entries - round]: {:?} {:x?}, negation: {:x}, previous_neg: {:x}",
            round,
            wnaf[(wnaf_entries - round) * num_points],
            wnaf[(wnaf_entries - round) * num_points],
            negation,
            previous_neg,
        );
        previous = slice + predicate;
        println!("previous: {:?} {:x?}", previous, previous);
    }

    let final_bits = S - (wnaf_bits * (wnaf_entries - 1));
    println!("final_bits: {:?} {:x?}", final_bits, final_bits);
    let slice = get_wnaf_bits(scalar, final_bits, (wnaf_entries - 1) * wnaf_bits);
    println!("slice: {:?} {:x?}", slice, slice);
    let predicate = (slice & 1 == 0) as u64;
    println!("predicate: {:?} {:x?}", predicate, predicate);
    let negation: u64 = (u64::MAX - predicate) % u64::MAX;
    let previous_neg: u64 = ((previous as u128 + u64::MAX as u128
        - (predicate << wnaf_bits) as u128)
        % u64::MAX as u128) as u64;
    wnaf[num_points] = (((previous_neg ^ negation) >> 1) | (predicate << 31)) | point_index;
    println!(
        "wnaf[num_points]: {:?} {:x?}",
        wnaf[num_points], wnaf[num_points]
    );
    wnaf[0] = ((slice + predicate) >> 1) | point_index;
    println!("wnaf[0]: {:?} {:x?}", wnaf[0], wnaf[0]);
}

pub(crate) fn fixed_wnaf_with_counts(
    scalar: Vec<u64>,
    wnaf: &mut [u64],
    skew_map: &mut bool,
    wnaf_round_counts: &mut [u64],
    point_index: u64,
    num_points: usize,
    wnaf_bits: usize,
) {
    let max_wnaf_entries = (SCALAR_BITS + wnaf_bits - 1) / wnaf_bits;

    //If the scalar is 0
    if scalar[0] == 0 && scalar[1] == 0 {
        *skew_map = false;
        for round in 0..max_wnaf_entries {
            wnaf[round * num_points] = u64::MAX;
        }
        return;
    }

    let current_scalar_bits = get_num_scalar_bits(&scalar) + 1;
    let hi: u64 = scalar[1];
    let lo: u64 = scalar[0];

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
        let negation: u64 = (u64::MAX - predicate) % u64::MAX;

        wnaf[(max_wnaf_entries - round) * num_points] =
            ((((previous - (predicate << (wnaf_bits/*+ 1*/))) ^ negation) >> 1)
                | (predicate << 31))
                | (point_index);

        // Update the previous value to the next windows
        previous = slice + predicate;
    }

    // Final iteration for top bits
    let final_bits = current_scalar_bits as usize - (wnaf_bits * (wnaf_entries - 1));
    let slice = get_wnaf_bits(&[hi, lo], final_bits, (wnaf_entries - 1) * wnaf_bits);
    let predicate = ((slice & 1) == 0) as u64;

    wnaf_round_counts[max_wnaf_entries - wnaf_entries + 1] += 1;
    let negation: u64 = (u64::MAX - predicate) % u64::MAX;

    wnaf[(max_wnaf_entries - wnaf_entries + 1) * num_points] =
        ((((previous - (predicate << (wnaf_bits/*+ 1*/))) ^ negation) >> 1) | (predicate << 31))
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ecc::scalar_multiplication::cube_root_of_unity;
    use ark_bn254::Fr;
    use ark_ec::scalar_mul::glv::GLVConfig;
    use ark_ff::BigInt;
    use ark_ff::PrimeField;
    use ark_ff::UniformRand;
    use num_bigint::BigUint;

    fn recover_fixed_wnaf(wnaf: &mut [u64], skew: bool, wnaf_bits: usize) -> (u64, u64) {
        let wnaf_entries: usize = (SCALAR_BITS + wnaf_bits - 1) / wnaf_bits;
        let mut scalar: u128 = 0;

        for (i, entry_formatted) in wnaf.iter().enumerate().take(wnaf_entries) {
            println!("entry_formatted: {:?}", entry_formatted);
            let negative = entry_formatted >> 31;
            let entry = (((entry_formatted & 0x0fffffffu64) << 1) + 1) as u128;
            println!("i: {}, negative: {}, entry: {:?}", i, negative, entry);
            if negative == 1 {
                scalar -= entry << (wnaf_bits * (wnaf_entries - 1 - i)) as u128;
            } else {
                scalar = scalar.wrapping_add(entry << (wnaf_bits * (wnaf_entries - 1 - i)) as u128);
            }
            println!("i: {}, scalar: {:x?}", i, scalar);
        }
        scalar -= skew as u128;
        println!("scalar - skew: {:?}", scalar);
        let hi = (scalar >> 64) as u64;
        let lo = (scalar & u64::MAX as u128) as u64;
        (hi, lo)
    }

    #[test]
    fn wnaf_zero() {
        let mut input = Fr::from(0);
        let mut wnaf = [0u64; wnaf_size(5)];
        let mut skew = false;

        fixed_wnaf::<SCALAR_BITS>(&mut input.0 .0, &mut wnaf, &mut skew, 0, 1, 5);

        let (recovered_lo, recovered_hi) = recover_fixed_wnaf(&mut wnaf, skew, 5);

        assert_eq!(input.0 .0[0], recovered_lo);
        assert_eq!(input.0 .0[0], recovered_hi);
    }

    #[test]
    fn wnaf_two_bit_window() {
        let mut rng = ark_std::test_rng();
        let mut input = Fr::rand(&mut rng);
        let window = 2;
        const NUM_BITS: usize = 254;
        const NUM_QUADS: usize = (NUM_BITS >> 1) + 1;
        let mut wnaf = [0u64; NUM_QUADS];
        let mut skew = false;

        fixed_wnaf::<256>(&mut input.0 .0, &mut wnaf, &mut skew, 0, 1, window);

        // For representing even numbers, we define a skew:
        //
        // ```
        //        / false   if input is odd
        // skew = |
        //        \ true    if input is even
        // ```
        // The i-th quad value is defined as:
        //
        // ```
        //        / -(2b + 1)   if sign = 1
        // q[i] = |
        //        \ (2b + 1)    if sign = 0
        // ```
        //
        // where sign = `((wnaf[i] >> 31) == 0)` and b = `(wnaf[i] & 1)`.
        // We can compute back the original number from the quads as:
        // ```
        //                 127
        //               -----
        //               \
        // R = -skew  +  |    4^{127 - i} . q[i].
        //               /
        //               -----
        //                i=0
        // ```
        let mut recovered = BigUint::from(0u64);
        let mut four_power = BigUint::from(1u64) << NUM_BITS;
        for (_, wnaf_entry) in wnaf.iter().enumerate().take(NUM_QUADS) {
            let extracted = 2 * (wnaf_entry & 1) + 1;
            let sign = wnaf_entry >> 31 == 0;
            if sign {
                //Note cast to uint256
                recovered += extracted * &four_power;
            } else {
                recovered -= extracted * &four_power;
            }
        }

        recovered -= skew as u32;

        assert_eq!(recovered, input.0.into());
    }

    #[test]
    fn wnaf_fixed_rand() {
        let mut rng = ark_std::test_rng();
        let mut buffer = Fr::rand(&mut rng);
        buffer.0 .0[1] &= 0x7fffffffffffffffu64;

        let mut wnaf = [0u64; wnaf_size(5)];
        let mut skew = false;
        fixed_wnaf::<SCALAR_BITS>(&mut buffer.0 .0, &mut wnaf, &mut skew, 0, 1, 5);

        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);

        assert_eq!(recovered_lo, buffer.0 .0[0]);
        assert_eq!(recovered_hi, buffer.0 .0[1]);
    }

    #[test]
    #[ignore]
    fn wnaf_fixed_simple_lo() {
        let mut buffer = [1u64, 0u64];
        let mut wnaf = [0u64; wnaf_size(5)];
        let mut skew = false;
        fixed_wnaf::<256>(&mut buffer, &mut wnaf, &mut skew, 0, 1, 5);

        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);

        assert_eq!(recovered_lo, buffer[0]);
        assert_eq!(recovered_hi, buffer[1]);
    }

    #[test]

    fn wnaf_fixed_simple_hi() {
        let mut buffer = [0u64, 1u64];
        let mut wnaf = [0u64; wnaf_size(5)];
        let mut skew = false;
        fixed_wnaf::<SCALAR_BITS>(&mut buffer, &mut wnaf, &mut skew, 0, 1, 5);

        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);

        assert_eq!(recovered_lo, buffer[0]);
        assert_eq!(recovered_hi, buffer[1]);
    }

    #[test]
    fn wnaf_fixed_with_endo_split() {
        let mut rng = ark_std::test_rng();
        let mut k = Fr::rand(&mut rng);
        println!("k: {:x?}", k);
        // k.0 .0[0] = 0x7d1faf1a18c7788fu64;
        // k.0 .0[1] = 0x4e53984ebf57f9au64;
        // k.0 .0[2] = 0xcf6d1069ea03ff3cu64;
        // k.0 .0[3] = 0x2f01189eb498b10u64;
        // k.0 .0[3] &= 0x0fffffffffffffffu64;

        let ((sgn_t1, mut t1), (sgn_t2, mut t2)) = ark_bn254::g1::Config::scalar_decomposition(k);

        println!("t1: {}, {:?}", sgn_t1, t1);
        println!("t2: {}, {:?}", sgn_t2, t2);

        if !sgn_t1 {
            t1 = -t1;
        }
        if !sgn_t2 {
            t2 = -t2;
        }

        println!("k: {:x?}", k);
        println!("t1: {:?}", t1);
        println!("t2: {:?}", t2);

        let mut wnaf = [0u64; wnaf_size(5)];
        let mut endo_wnaf = [0u64; wnaf_size(5)];
        let mut skew = false;
        let mut endo_skew = false;

        fixed_wnaf::<SCALAR_BITS>(&mut t1.0 .0, &mut wnaf, &mut skew, 0, 1, 5);
        fixed_wnaf::<SCALAR_BITS>(&mut t2.0 .0, &mut endo_wnaf, &mut endo_skew, 0, 1, 5);

        println!("wnaf: {:x?}", wnaf);
        println!("wnaf_endo: {:x?}", endo_wnaf);
        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);
        let (endo_recovered_hi, endo_recovered_lo) =
            recover_fixed_wnaf(&mut endo_wnaf, endo_skew, 5);
        println!("recovered_lo: {:x?}", recovered_lo);
        println!("recovered_hi: {:x?}", recovered_hi);

        println!("endo_recovered_lo: {:x?}", endo_recovered_lo);
        println!("endo_recovered_hi: {:x?}", endo_recovered_hi);

        let mut recovered = Fr::from(0);
        (recovered.0 .0[0], recovered.0 .0[1]) = (recovered_lo, recovered_hi);
        let mut endo_recovered = Fr::from(0);
        (endo_recovered.0 .0[0], endo_recovered.0 .0[1]) = (endo_recovered_lo, endo_recovered_hi);

        let lambda: Fr = cube_root_of_unity();
        println!("lamdba: {:x?}", wnaf);
        let result = endo_recovered * lambda;
        println!("k2_recovered * lambda: {:x?}", result);
        let result = recovered - result;
        println!("k1_recovered - result: {:x?}", result);
        assert_eq!(result, k);
    }
}
