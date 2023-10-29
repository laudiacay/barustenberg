use ark_ec::{
    short_weierstrass::{Affine, Projective, SWCurveConfig},
    CurveConfig, CurveGroup,
};
use grumpkin::GrumpkinConfig;

use crate::{crypto::generator::GeneratorContext, ecc::groups::group::derive_generators};

use super::pederson::commit_native;

/**
 * @brief Given a vector of fields, generate a pedersen commitment using the indexed generators.
 *
 * @details This method uses `Curve::BaseField` members as inputs. This aligns with what we expect when creating
 * grumpkin commitments to field elements inside a BN254 SNARK circuit.
 * @param inputs
 * @param context
 * @return Curve::AffineElement
 */
//TODO: confirm we can do this with scalar field
pub(crate) fn hash(
    inputs: &[grumpkin::Fq],
    context: &mut GeneratorContext<GrumpkinConfig>,
) -> <GrumpkinConfig as CurveConfig>::BaseField {
    let res: Affine<GrumpkinConfig> = (length_generator()
        * <GrumpkinConfig as CurveConfig>::ScalarField::from(inputs.len() as u64))
    .into_affine();
    (res + commit_native(inputs, context)).x
}

//Note: this can be abstracted to a lazy_static!()
fn length_generator<E: SWCurveConfig>() -> Affine<E> {
    derive_generators::<E>("pedersen_hash_length".as_bytes(), 1, 0)[0]
}

//TODO: unneeded
/*
pub(crate) fn hash_buffer<E: SWCurveConfig>(input: &[u8], /* context: GeneratorContext */) -> E::BaseField {
std::vector<Fq> converted = convert_buffer(input);

if (converted.size() < 2) {
    return hash(converted, context);
}
auto result = hash({ converted[0], converted[1] }, context);
for (size_t i = 2; i < converted.size(); ++i) {
    result = hash({ result, converted[i] }, context);
}
return result;
}
*/

#[cfg(test)]
pub(crate) mod test {
    use crate::crypto::generator::GENERATOR_CONTEXT;

    use super::*;

    use ark_ff::{AdditiveGroup, BigInteger, PrimeField, Zero};
    use ark_serialize::CanonicalSerialize;
    use ark_std::{One, UniformRand};
    use grumpkin::{Fq, Fr};

    #[test]
    fn zero_one() {
        let res = commit_native(
            &[Fq::zero(), Fq::one()],
            &mut GENERATOR_CONTEXT.lock().unwrap(),
        );
        let mut res_bytes = Vec::new();
        res.serialize_uncompressed(&mut res_bytes).unwrap();
        assert_eq!(
            res_bytes,
            "0x0c5e1ddecd49de44ed5e5798d3f6fb7c71fe3d37f5bee8664cf88a445b5ba0af".as_bytes()
        );
    }
}
