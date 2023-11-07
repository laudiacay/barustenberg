use ark_ec::{short_weierstrass::Affine, AffineRepr, CurveGroup};
use ark_ff::PrimeField;
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

#[cfg(test)]
pub(crate) mod test {
    use crate::crypto::generator::GENERATOR_CONTEXT;

    use super::*;

    use ark_ff::MontFp;
    use ark_std::One;
    use grumpkin::Fq;

    //TODO: double check that the generators are the same. They could be slightly different due to the way we canonically
    // decide how to invert y which was done to prevent a headache of having to deseialize an Fq element... Long story.
    // We compute sum_{gen * input}; -> 1 * gen[0] + 1 * gen[1] - Given Group ops -> (gen[0].operate_with_self(1)).operate_with(gen[1].operate_with_self(1))
    // -> gen[0].operate_with(gen[1])
    // Compute naive generators sage:
    /*
       r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
       p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
       a = 0
       b = -17
       Fr = GF(r)
       grumpkin = EllipticCurve(Fr, [a,b])
       grumpkin.set_order(p)
       # generators[1]
       e = grumpkin(
           18427940726016234078985946418448280648870225692973225849694456867521160726934,
            21532357112255058987590902028734969864671062849942210368353786847928073297018
           )
       # generators[1]
       d = grumpkin(
           391479787776068058408721993510469975463547477513640094152105077479335183379,
           11137044286323765227152527641563178484030256868339213989437323640137135753514
       )
       c = e + d
       #c
       #(
       #    231570567088489780672426506353362499554225005377301234298356723277158049403
       #    : 15307670091902218669505377418137932514463250251528740589240008994863263703888
       #    : 1
       #)
    */
    #[test]
    fn commitment() {
        let res = commit_native(
            &[Fq::one(), Fq::one()],
            &mut GENERATOR_CONTEXT.lock().unwrap(),
        );
        let expected = Affine::new(
            // 2f7a8f9a6c96926682205fb73ee43215bf13523c19d7afe36f12760266cdfe15
            MontFp!("231570567088489780672426506353362499554225005377301234298356723277158049403"),
            // 01916b316adbbf0e10e39b18c1d24b33ec84b46daddf72f43878bcc92b6057e6
            MontFp!(
                "15307670091902218669505377418137932514463250251528740589240008994863263703888"
            ),
        );

        assert_eq!(res, expected);
    }
}
