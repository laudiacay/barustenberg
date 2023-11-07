use ark_ec::{
    short_weierstrass::{Affine, SWCurveConfig},
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
pub(crate) fn hash(
    inputs: &[grumpkin::Fq],
    context: &mut GeneratorContext<GrumpkinConfig>,
) -> <GrumpkinConfig as CurveConfig>::BaseField {
    let res: Affine<GrumpkinConfig> = (length_generator(0)
        * <GrumpkinConfig as CurveConfig>::ScalarField::from(inputs.len() as u64))
    .into_affine();
    //Note output is x in projective coordinates not affine.
    (res + commit_native(inputs, context)).into_affine().x
}

pub(crate) fn hash_with_index(
    inputs: &[grumpkin::Fq],
    starting_index: usize,
    context: &mut GeneratorContext<GrumpkinConfig>,
) -> <GrumpkinConfig as CurveConfig>::BaseField {
    let res: Affine<GrumpkinConfig> = (length_generator(starting_index)
        * <GrumpkinConfig as CurveConfig>::ScalarField::from(inputs.len() as u64))
    .into_affine();
    //Note output is x in projective coordinates not affine.
    (res + commit_native(inputs, context)).into_affine().x
}

//Note: this can be abstracted to a lazy_static!()
// length_generator: (16063406592428334581056774180896419344331184927683265052314702073066435943873, 17916849626726460830866689840922781751319971491928759743495979859497111217886)
// length_generator with index 5:  (5536982424527559415100431280492329513183658401277695557077208029565215414989, 11331795194798001435190325606804339647380614317757253421377381911305808989494)
fn length_generator<E: SWCurveConfig>(starting_index: usize) -> Affine<E> {
    derive_generators::<E>("pedersen_hash_length".as_bytes(), 1, starting_index)[0]
}

pub(crate) fn hash_buffer<E: SWCurveConfig>(
    input: &[u8],
    context: &mut GeneratorContext<E>,
) -> E::BaseField {
    todo!()
}

#[cfg(test)]
pub(crate) mod test {
    use crate::crypto::generator::GENERATOR_CONTEXT;

    use super::*;

    use ark_ff::MontFp;
    use ark_std::One;
    use grumpkin::Fq;

    //reference: https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/crypto/pedersen_hash/pedersen.test.cpp
    /*
    res = length_generator.operate_with_self(inputs.len()).operate_with(commit_1).x
    res = length_generator.operate_with_self(2u64).operate_with(commit_1).x

    sage:
        r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
        a = 0
        b = -17
        Fr = GF(r)
        grumpkin = EllipticCurve(Fr, [a,b])
        grumpkin.set_order(p)
        length_generator = (
            16063406592428334581056774180896419344331184927683265052314702073066435943873,
            17916849626726460830866689840922781751319971491928759743495979859497111217886
        )
        commit_1 = (
            231570567088489780672426506353362499554225005377301234298356723277158049403,
            15307670091902218669505377418137932514463250251528740589240008994863263703888
        )
        res = 2 * length_generator + commit_1
        #res :
        #(
        #   16687715145901069277107513335124964838889590446104409078894381382220183021285
        #   : 6241750969117076836813882815945137218857312263712512894890931111676305664938
        #   : 1
        #)
        #res.x to projective
        # 16687715145901069277107513335124964838889590446104409078894381382220183021285

    */
    #[test]
    fn hash_one() {
        let res = hash(
            &[Fq::one(), Fq::one()],
            &mut GENERATOR_CONTEXT.lock().unwrap(),
        );

        assert_eq!(
            res,
            MontFp!(
                "16687715145901069277107513335124964838889590446104409078894381382220183021285"
            )
        );
    }

    /*
    res = length_generator_5.operate_with_self(inputs.len()).operate_with(commit_1).x
    res = length_generator_5.operate_with_self(2u64).operate_with(commit_1).x

    sage:
        r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
        p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
        a = 0
        b = -17
        Fr = GF(r)
        grumpkin = EllipticCurve(Fr, [a,b])
        grumpkin.set_order(p)
        length_generator_5 = (
            5536982424527559415100431280492329513183658401277695557077208029565215414989,
            11331795194798001435190325606804339647380614317757253421377381911305808989494
        )
        commit_1 = (
            231570567088489780672426506353362499554225005377301234298356723277158049403,
            15307670091902218669505377418137932514463250251528740589240008994863263703888
        )
        res = 2 * length_generator + commit_1
        #res :
        #(
        #   3968502498651788738938452400218391002721537179087268284303065092205739166561
        #   : 8891768054196819470001457132198459539016113745327151336922089362609089470852
        #   : 1
        #)
        #res.x to projective
        # 3968502498651788738938452400218391002721537179087268284303065092205739166561
    */
    #[test]
    fn hash_one_with_index() {
        let res = hash_with_index(
            &[Fq::one(), Fq::one()],
            5,
            &mut GENERATOR_CONTEXT.lock().unwrap(),
        );
        assert_eq!(
            res,
            //MontFp!("3382712453967845713887399370753050874236856032939624587459622627919799159283")
            MontFp!("3968502498651788738938452400218391002721537179087268284303065092205739166561")
        );
    }
}
