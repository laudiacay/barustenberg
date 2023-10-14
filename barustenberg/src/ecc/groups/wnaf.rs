use ark_ec::{AffineRepr, AdditiveGroup, CurveGroup, short_weierstrass::{Affine, Projective, SWCurveConfig}, CurveConfig};
use ark_ff::{BigInt, Fp256, BigInteger, Field, PrimeField};
use crate::numeric::bitop::get_msb::get_msb64;

const SCALAR_BITS: u64 = 127u64;

//Some assumptions are made here.. First and furmost the number of limbs > 2. Also since barretenberg operates on u128 scalers we convert at this stage by using get_wnaf_bits_const
pub fn fixed_wnaf<Curve: CurveConfig, const NUM_BITS: u64, const NUM_POINTS: u64, const WNAF_BITS: u64>(scalar: Curve::ScalarField, wnaf: &mut [u64], skew_map: &u64, point_index: u64) {
    //TODO: convert from montgomery to non montgomery
    let scalar = scalar.into_bigint().as_ref();
    let skew_map = ((scalar[0] & 1) == 0) as u64;
    let prev = get_wnaf_bits(scalar, WNAF_BITS, 0) + skew_map;
    wnaf_round::<NUM_POINTS, WNAF_BITS>(scalar, wnaf, point_index, prev, 1)
}

#[inline(always)]
fn wnaf_round<const NUM_POINTS: u64, const WNAF_BITS: u64>(scalar: &[u64], wnaf: &mut [u64], point_index: u64, prev: u64, round_i: u64) {
   let wnaf_entries = SCALAR_BITS + WNAF_BITS - 1;
   let log2_num_points = get_msb64(NUM_POINTS);

    if round_i < wnaf_entries - 1 {
        let slice = get_wnaf_bits(scalar, WNAF_BITS,round_i * WNAF_BITS);
        let predicate = ((slice & 1) == 0) as u64;
        let index = (wnaf_entries - round_i) << log2_num_points;
        wnaf[index as usize] = (prev.wrapping_sub(predicate << WNAF_BITS)) ^ (((0u64.wrapping_sub(predicate)) >> 1u64) | (predicate.wrapping_shl(31))) | (point_index.wrapping_shl(31));
        wnaf_round(scalar, wnaf, point_index, prev, round_i + 1)
    } else {
        let final_bits = SCALAR_BITS - (SCALAR_BITS / WNAF_BITS) * WNAF_BITS;
        let slice = get_wnaf_bits(scalar, final_bits, (wnaf_entries - 1) * WNAF_BITS);
        let predicate = ((slice & 1) == 0) as u64;
        wnaf[NUM_POINTS as usize] = (prev.wrapping_sub(predicate << WNAF_BITS)) ^ (((0u64.wrapping_sub(predicate)) >> 1u64) | (predicate.wrapping_shl(31))) | (point_index.wrapping_shl(31));
        wnaf[0] = (slice.wrapping_add(predicate)).wrapping_shr(1) | point_index.wrapping_shl(32);
    }
}

#[inline(always)]
const fn get_wnaf_bits(scalar: &[u64], bits: u64, bit_position: u64) -> u64 {
    //NOTE: This implementation in C++ computes the index of limb of the u128 scalar of interest
    //scalars are stored as an array of u64 limb values with two representing a u128 of the
    if bits == 0 { 
        return 0 
    } else {
        println!("get_wnaf_bits");
        println!("bits: {:x?}, {:?}", bits, bits);
        println!("bit_positon: {:x?}, {:?}", bit_position, bit_position);
        //TODO: eliminate this indexing once everything is worked out
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
        let lo_limb_idx = bit_position / 64;
        let hi_limb_idx = (bit_position + bits - 1) / 64;
        let lo_shift = (bit_position & 63) as u64;
        let bit_mask = (1u64 << bits).wrapping_sub(1);
        
        println!("lo_limb_idx: {:?}", lo_limb_idx);
        println!("hi_limb_idx: {:?}", hi_limb_idx);
        println!("lo_shift {:?}", lo_shift);
        println!("bit_mask: {:?}", bit_mask);

        let lo = scalar[lo_limb_idx as usize] >> lo_shift;
        println!("lo: {:?}", lo);
        if lo_limb_idx == hi_limb_idx {
            return lo & bit_mask;
        } else {
            let hi_shift = 64 - (bit_position & 63);
            println!("hi_shift: {:x?}", hi_shift);
            let hi: u64 = scalar[hi_limb_idx as usize] << hi_shift;
            return (lo | hi) & bit_mask;
        }
    }
}
#[cfg(test)]
mod tests {
    //use crate::ecc::scalar_multiplication::cube_root_of_unity;

    use super::*;
    use ark_bn254::{g1::Config, Fr, G1Affine, G1Projective};
    use ark_ec::scalar_mul::glv::GLVConfig;
    use ark_ff::{BigInteger, Field, PrimeField};
    use ark_std::{One, UniformRand, Zero};

    //TODO: implement GLVConfig for Bn254;

    //Have this return hi and lo

    fn recover_fixed_wnaf(wnaf: &mut [u64], skew: bool, wnaf_bits: usize) -> (u64, u64) {
        let wnaf_entries: usize = (127 + wnaf_bits - 1) / wnaf_bits;
        let mut scalar: u128 = 0;
        let skew = if skew == true { 1u128 } else { 0u128 };
        for i in 0..wnaf_entries {
            let entry_formatted: u64 = wnaf[i];
            println!("entry_formatted: {:?}", entry_formatted);
            //This doesn't work in rust
            let negative = entry_formatted >> 31;
            println!("negative: {:?}", negative);
            let entry: u128 = (((entry_formatted & 0x0fffffffu64) << 1) + 1) as u128;
            println!("entry: {:?}", entry);
            if negative == 1 {
                scalar -= entry << (wnaf_bits * (wnaf_entries - 1 - i)) as u128
            } else {
                scalar += entry << (wnaf_bits * (wnaf_entries - 1 - i)) as u128
            }
            println!("scalar: {:?}", scalar);
        }
        scalar -= skew;
        println!("scalar - skew: {:?}", scalar);
        let hi = (scalar >> 64) as u64;
        let lo = (scalar & u128::MAX) as u64;
        (hi, lo)
    }

    #[test]
    fn wnaf_zero() {
        //Print outs of Fr from arkworks to figure shit out.
        // Scalar values
        // let rng =
        let mut input = Fr::from(0);
        let mut wnaf = [0u64; WNAF_SIZE(5)];
        let mut skew = false;
        println!("Fr Internal BigInt [u64; 4]");
        println!("{:?}", input.0 .0);
        println!("wnaf: {:?}", wnaf);
        fixed_wnaf(&mut input.0 .0, &mut wnaf, &mut skew, 0, 1, 5);
        println!("wnaf: {:?}", wnaf);
        let (recovered_lo, recovered_hi) = recover_fixed_wnaf(&mut wnaf, skew, 5);
        println!("recovered_lo {:?}", recovered_lo);
        println!("recovered_hi {:?}", recovered_hi);

        assert_eq!(recovered_lo, 0);
        assert_eq!(recovered_hi, 0);
        assert_eq!(input.0 .0[0], recovered_lo);
        assert_eq!(input.0 .0[0], recovered_hi);
        /*
        let buffer = [0u64, 0u64];
        // Wnaf values
        let wnaf = [0u64; WNAF_SIZE(5)];
        let skew = 0;
        let wnaf = fixed_wnaf();
        let (rec_hi, rec_lo) = recover_fixed_wnaf(wnaf, skew, 5);
        assert_eq()
        */
    }

    #[test]
    #[ignore]
    fn wnaf_two_bit_window() {
        let mut rng = ark_std::test_rng();
        let input = Fr::rand(&mut rng);
        let window = 2;
        const num_bits: usize = 254;
        const num_quads: usize = ((num_bits >> 1) + 1) as usize;
        let wnaf = [0u64; num_quads];
        let skew = false;
        let out = fixed_wnaf(&mut input.0 .0, &mut wnaf, &mut skew, 0, 1, 5);

        //Note cast to uint256

        /*
        For representing even numbers, we define a skew:
               / false   if input is odd
        skew = |
               \ true    if input is even
        The i-th quad value is defined as:
               / -(2b + 1)   if sign = 1
        q[i] = |
               \ (2b + 1)    if sign = 0
        where sign = ((wnaf[i] >> 31) == 0) and b = (wnaf[i] & 1).
        We can compute back the original number from the quads as:
                       127
                      -----
                      \
        R = -skew  +  |    4^{127 - i} . q[i].
                      /
                      -----
                       i=0
        */
        let recovered = 0u64;
        //NOTE this is cast to uint256 in C++
        let four_power = 1 << num_bits;
        for i in 0..num_quads {
            let extracted = 2 * ((wnaf[i] as u64) & 1) + 1;
            let sign = wnaf[i] >> 31;
            if sign != 0 {
                //Note cast to uint256
                recovered += extracted * four_power;
            } else {
                recovered -= extracted * four_power;
            }

            four_power >>= 2;
        }

        let skew = if skew == false { 0 } else { 1 };
        recovered -= skew;

        assert_eq!(Fr::from(recovered), input);
    }

    #[test]
    #[ignore]
    fn wnaf_fixed_rand() {
        let mut rng = ark_std::test_rng();
        let mut buffer = Fr::rand(&mut rng);
        buffer.0 .0[1] = 0x7fffffffu64;
        let mut wnaf = [0u64; WNAF_SIZE(5)];
        let mut skew = false;
        fixed_wnaf(&mut buffer.0 .0, &mut wnaf, &mut skew, 0, 1, 5);
        println!("wnaf: {:?}", wnaf);
        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);
        println!("recovered_lo {:?}", recovered_lo);
        println!("recovered_hi {:?}", recovered_hi);

        assert_eq!(recovered_lo, buffer.0 .0[0]);
        assert_eq!(recovered_hi, buffer.0 .0[1]);
    }

    #[test]
    #[ignore]
    fn wnaf_fixed_simple_lo() {
        let mut buffer = [1u64, 0u64];
        let mut wnaf = [0u64; WNAF_SIZE(5)];
        let mut skew = false;
        fixed_wnaf(&mut buffer, &mut wnaf, &mut skew, 0, 1, 5);

        println!("wnaf: {:?}", wnaf);
        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);
        println!("recovered_lo {:?}", recovered_lo);
        println!("recovered_hi {:?}", recovered_hi);

        assert_eq!(recovered_lo, buffer[0]);
        assert_eq!(recovered_hi, buffer[1]);
    }

    #[test]
    #[ignore]

    fn wnaf_fixed_simple_hi() {
        let mut buffer = [0u64, 1u64];
        let mut wnaf = [0u64; WNAF_SIZE(5)];
        let mut skew = false;
        fixed_wnaf(&mut buffer, &mut wnaf, &mut skew, 0, 1, 5);

        println!("wnaf: {:?}", wnaf);
        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);
        println!("recovered_lo {:?}", recovered_lo);
        println!("recovered_hi {:?}", recovered_hi);

        assert_eq!(recovered_lo, buffer[0]);
        assert_eq!(recovered_hi, buffer[1]);
    }

    #[test]
    #[ignore]
    fn wnaf_fixed_with_endo_split() {
        let mut rng = ark_std::test_rng();
        let k = Fr::rand(&mut rng);
        k.0 .0[3] &= 0x0fffffffu64;
        let k1 = Fr::from(0);
        let k2 = Fr::from(0);

        //TODO: implement endomorphism split
        let wnaf = [0u64; WNAF_SIZE(5)];
        let endo_wnaf = [0u64; WNAF_SIZE(5)];
        let skew = false;
        let endo_skew = false;

        fixed_wnaf(&mut k1.0 .0, &mut wnaf, &mut skew, 0, 1, 5);
        fixed_wnaf(&mut k2.0 .0, &mut endo_wnaf, &mut endo_skew, 0, 1, 5);

        println!("wnaf: {:?}", wnaf);
        println!("wnaf_endo: {:?}", wnaf);
        let (recovered_hi, recovered_lo) = recover_fixed_wnaf(&mut wnaf, skew, 5);
        let (endo_recovered_hi, endo_recovered_lo) = recover_fixed_wnaf(&mut endo_wnaf, skew, 5);
        println!("recovered_lo: {:?}", recovered_lo);
        println!("recovered_hi: {:?}", recovered_hi);

        println!("endo_recovered_lo: {:?}", endo_recovered_lo);
        println!("endo_recovered_hi: {:?}", endo_recovered_hi);

        let lambda: Fr = cube_root_of_unity();
        println!("lamdba: {:?}", wnaf);
        let result = k2 * lambda;
        println!("k2_recovered * lambda: {:?}", result);
        let result = k1 - result;
        println!("k1_recovered - result: {:?}", result);
        assert_eq!(result, k);
    }
}