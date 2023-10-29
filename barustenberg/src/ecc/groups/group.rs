use ark_ec::short_weierstrass::{Affine, SWCurveConfig};

use super::affine_element::hash_to_curve;

/**
 * @brief Derives generator points via hash-to-curve
 *
 * ALGORITHM DESCRIPTION:
 *      1. Each generator has an associated "generator index" described by its location in the vector
 *      2. a 64-byte preimage buffer is generated with the following structure:
 *          bytes 0-31: BLAKE3 hash of domain_separator
 *          bytes 32-63: generator index in big-endian form
 *      3. The hash-to-curve algorithm is used to hash the above into a group element:
 *           a. iterate `count` upwards from `0`
 *           b. append `count` to the preimage buffer as a 1-byte integer in big-endian form
 *           c. compute BLAKE3 hash of concat(preimage buffer, 0)
 *           d. compute BLAKE3 hash of concat(preimage buffer, 1)
 *           e. interpret (c, d) as (hi, low) limbs of a 512-bit integer
 *           f. reduce 512-bit integer modulo coordinate_field to produce x-coordinate
 *           g. attempt to derive y-coordinate. If not successful go to step (a) and continue
 *           h. if parity of y-coordinate's least significant bit does not match parity of most significant bit of
 *              (d), invert y-coordinate.
 *           j. return (x, y)
 *
 * NOTE: In step 3b it is sufficient to use 1 byte to store `count`.
 *       Step 3 has a 50% chance of returning, the probability of `count` exceeding 256 is 1 in 2^256
 * NOTE: The domain separator is included to ensure that it is possible to derive independent sets of
 * index-addressable generators.
 * NOTE: we produce 64 bytes of BLAKE3 output when producing x-coordinate field
 * element, to ensure that x-coordinate is uniformly randomly distributed in the field. Using a 256-bit input adds
 * significant bias when reducing modulo a ~256-bit coordinate_field
 * NOTE: We ensure y-parity is linked to preimage
 * hash because there is no canonical deterministic square root algorithm (i.e. if a field element has a square
 * root, there are two of them and `field::sqrt` may return either one)
 * @param num_generators
 * @param domain_separator
 * @return std::vector<affine_element>
 */
pub(crate) fn derive_generators<E: SWCurveConfig>(
    domain_separator_bytes: &[u8],
    num_generators: usize,
    starting_index: usize,
) -> Vec<Affine<E>> {
    let mut res = Vec::new();
    let mut generator_preimage = [0u8; 64];
    let domain_hash = blake3::hash(domain_separator_bytes);
    //1st 32 bytes are blak3 domain_hash
    generator_preimage[..32].copy_from_slice(domain_hash.as_bytes());

    // Convert generator index in big-endian form
    for i in starting_index..(starting_index + num_generators) {
        let mask = 0xffu32;
        let generator_index = i as u32;

        generator_preimage[32] = (generator_index >> 24) as u8;
        generator_preimage[33] = ((generator_index >> 16) & mask) as u8;
        generator_preimage[34] = ((generator_index >> 8) & mask) as u8;
        generator_preimage[35] = (generator_index & mask) as u8;
        res.push(hash_to_curve(&generator_preimage, 0));
    }
    res
}

#[cfg(test)]
pub(crate) mod test {

    use grumpkin::GrumpkinConfig;

    use super::*;

    #[test]
    fn test_derive_generators() {
        let res = derive_generators::<GrumpkinConfig>("test domain".as_bytes(), 128, 0);

        let is_unique = |y: Affine<GrumpkinConfig>, j: usize| -> bool {
            for (i, res) in res.iter().enumerate() {
                if i != j && *res == y {
                    return false;
                }
            }
            true
        };

        for (i, res) in res.iter().enumerate() {
            assert_eq!(is_unique(*res, i), true);
            assert_eq!(res.is_on_curve(), true);
        }
    }
}
