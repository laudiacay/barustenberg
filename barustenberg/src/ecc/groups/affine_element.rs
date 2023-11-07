use ark_ec::short_weierstrass::{Affine, SWCurveConfig};

use ark_ff::Field;

/**
 * @brief Hash a seed buffer into a point
 *
 * @details ALGORITHM DESCRIPTION:
 *          1. Initialize unsigned integer `attempt_count = 0`
 *          2. Copy seed into a buffer whose size is 2 bytes greater than `seed` (initialized to 0)
 *          3. Interpret `attempt_count` as a byte and write into buffer at [buffer.size() - 2]
 *          4. Compute Blake3s hash of buffer
 *          5. Set the end byte of the buffer to `1`
 *          6. Compute Blake3s hash of buffer
 *          7. Interpret the two hash outputs as the high / low 256 bits of a 512-bit integer (big-endian)
 *          8. Derive x-coordinate of point by reducing the 512-bit integer modulo the curve's field modulus (Fq)
 *          9. Compute y^2 from the curve formula y^2 = x^3 + ax + b (a, b are curve params. for BN254, a = 0, b = 3)
 *          10. IF y^2 IS NOT A QUADRATIC RESIDUE
 *              10a. increment `attempt_count` by 1 and go to step 2
 *          11. IF y^2 IS A QUADRATIC RESIDUE
 *              11a. derive y coordinate via y = sqrt(y)
 *              11b. Interpret most significant bit of 512-bit integer as a 'parity' bit
 *              In Barretenberg:
 *                  11c. If parity bit is set AND y's most significant bit is not set, invert y
 *                  11d. If parity bit is not set AND y's most significant bit is set, invert y
 *              In Barustenberg we use arkworks https://github.com/arkworks-rs/algebra/blob/master/ec/src/models/short_weierstrass/affine.rs#L110:
 *                  11c. If parity bit is set AND y < -y lexographically, invert y
 *                  11d. If parity bit is not set AND y >= -y lexographically, invert y
 *              N.B. last 2 steps are because the sqrt() algorithm can return 2 values,
 *                   we need to a way to canonically distinguish between these 2 values and select a "preferred" one
 *              11e. return (x, y)
 *
 * @note This algorihm is constexpr: we can hash-to-curve (and derive generators) at compile-time!
 * @tparam Fq
 * @tparam Fr
 * @tparam T
 * @param seed Bytes that uniquely define the point being generated
 * @param attempt_count
 * @return constexpr affine_element<Fq, Fr, T>
 */
pub(crate) fn hash_to_curve<E: SWCurveConfig>(seed: &[u8], attempt_count: u8) -> Affine<E> {
    let seed_size = seed.len();
    // expand by 2 bytes to cover incremental hash attempts
    let mut target_seed = seed.to_vec();
    target_seed.extend_from_slice(&[0u8; 2]);

    target_seed[seed_size] = attempt_count;
    target_seed[seed_size + 1] = 0;
    let hash_hi = blake3::hash(&target_seed);
    target_seed[seed_size + 1] = 1;
    let hash_lo = blake3::hash(&target_seed);

    // custom serialize methods as common/serialize.hpp is not constespr
    //TODO: this should be double checked the provides enough entropy as its supposed to reduce from 512
    // see
    let mut hash = hash_hi.as_bytes().to_vec();
    hash.extend_from_slice(hash_lo.as_bytes());
    //TODO: handle unwrap()
    if let Some(x) = E::BaseField::from_random_bytes(&hash) {
        let sign_bit = hash_hi.as_bytes()[0] > 127 & hash[0] & 1;
        if let Some(res) = Affine::get_point_from_x_unchecked(x, sign_bit) {
            res
        } else {
            hash_to_curve(seed, attempt_count + 1)
        }
    } else {
        hash_to_curve(seed, attempt_count + 1)
    }
}
