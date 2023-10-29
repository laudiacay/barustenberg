use ark_ec::{
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr, CurveGroup,
};
use ark_ff::{Field, Fp, PrimeField};
use grumpkin::{Fq, Fr, GrumpkinConfig};

use crate::crypto::generator::GeneratorContext;

/**
 * @brief Given a vector of fields, generate a pedersen commitment using the indexed generators.
 *
 * @details This method uses `Curve::BaseField` members as inputs. This aligns with what we expect when creating
 * grumpkin commitments to field elements inside a BN254 SNARK circuit.
 * @param inputs
 * @param context
 * @return Curve::AffineElement
 */
// NOTE: this could be generalized using SWCurveConfig but since we perform the operation over grumpkin its explicit
pub(crate) fn commit_native(
    inputs: &[Fq],
    context: &mut GeneratorContext<GrumpkinConfig>,
) -> Affine<GrumpkinConfig> {
    let generators = context
        .generators
        .get(inputs.len(), context.offset, context.domain_separator);

    inputs
        .iter()
        .enumerate()
        .fold(Affine::zero(), |mut acc, (i, input)| {
            //TODO: this is a sketch conversion do better
            acc = (acc
                + (generators[i] * Fr::from_bigint(input.into_bigint()).unwrap()).into_affine())
            .into_affine();
            acc
        })
}
