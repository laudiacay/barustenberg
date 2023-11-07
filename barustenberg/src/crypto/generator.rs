use grumpkin::GrumpkinConfig;
use lazy_static::lazy_static;

use std::{collections::HashMap, fmt::Debug, sync::Mutex};

use ark_ec::{
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr,
};

use crate::ecc::groups::group::derive_generators;

pub(crate) const DEFAULT_NUM_GENERATORS: usize = 8;
pub(crate) const DEFAULT_DOMAIN_SEPARATOR: &str = "DEFAULT_DOMAIN_SEPARATOR";

/*
List of all precomputed generators in affine coordinates

All_Generators:
[
    (
        18427940726016234078985946418448280648870225692973225849694456867521160726934,
         21532357112255058987590902028734969864671062849942210368353786847928073297018
    ),
    (
        391479787776068058408721993510469975463547477513640094152105077479335183379,
        11137044286323765227152527641563178484030256868339213989437323640137135753514
    ),
    (
        14442619639594329090948287275462914401869112721451669244822744186993058642713,
        19060955993989836072679297498661860739767413747342022719099475476111802235684
    ),
    (
        93575404162741844566953656450595175402904060182800507239163355611016993528,
        16491093566555695425670083230468735512408272342024790546008254381077191753569
    ),
    (
        5506724260173198264586219989887984482434021843640055250566834480900107264426,
        13990390870884641278449924068113696783733397209027204117951316692612190950881
    ),
    (
        10533749543385051784888046135834657407284167410770423050691213988583851345747,
        19708387793592422627166852317306604475206645861188213514486090887853283716774
    ),
    (
        7533880232966985904222877823792629149734462662193749537693385897647000764500,
        16734016857686057769745723256575698627097333031607825670976385101205229633402
    ),
    (
        1523821630696474771658301553673949903269030288485867160765272196830332550594,
        15388812865175496227267879434763016586289980098899160549458825066780507118911
    )
]

*/

//Ref that can be imported to access pre-computed generators
lazy_static! {
    pub(crate) static ref GENERATOR_CONTEXT: Mutex<GeneratorContext<GrumpkinConfig>> =
        Mutex::new(GeneratorContext::default());
}

#[derive(Debug, Clone)]
pub(crate) struct GeneratorList<E: SWCurveConfig>(Vec<Affine<E>>);

// In barustenberg there exists a shared ladder storing cached precomputed values.
#[derive(Clone, Debug)]
pub(crate) struct GeneratorData<E: SWCurveConfig> {
    pub(crate) precomputed_generators: [Affine<E>; DEFAULT_NUM_GENERATORS],
    pub(crate) generator_map: HashMap<String, GeneratorList<E>>,
}

impl<E: SWCurveConfig> Default for GeneratorData<E> {
    fn default() -> Self {
        Self {
            precomputed_generators: Self::make_precomputed_generators(),
            generator_map: HashMap::new(),
        }
    }
}

impl<E: SWCurveConfig> GeneratorData<E> {
    fn make_precomputed_generators() -> [Affine<E>; DEFAULT_NUM_GENERATORS] {
        let mut output: [Affine<E>; DEFAULT_NUM_GENERATORS] =
            [Affine::zero(); DEFAULT_NUM_GENERATORS];
        let res: Vec<Affine<E>> = derive_generators(
            DEFAULT_DOMAIN_SEPARATOR.as_bytes(),
            DEFAULT_NUM_GENERATORS,
            0,
        );
        output.copy_from_slice(&res[..DEFAULT_NUM_GENERATORS]);
        output
    }

    //NOTE: can add default arguments by wrapping function parameters with options
    pub(crate) fn get(
        &mut self,
        num_generators: usize,
        generator_offset: usize,
        domain_separator: &str,
    ) -> Vec<Affine<E>> {
        let is_default_domain = domain_separator == DEFAULT_DOMAIN_SEPARATOR;
        if is_default_domain && (num_generators + generator_offset) < DEFAULT_NUM_GENERATORS {
            return self.precomputed_generators.to_vec();
        }

        // Case 2: we want default generators, but more than we precomputed at compile time. If we have not yet copied
        // the default generators into the map, do so.
        if is_default_domain && !self.generator_map.is_empty() {
            let _ = self
                .generator_map
                .insert(
                    DEFAULT_DOMAIN_SEPARATOR.to_string(),
                    GeneratorList(self.precomputed_generators.to_vec()),
                )
                .unwrap();
        }

        //TODO: open to suggestions for this
        let mut generators = self
            .generator_map
            .get(DEFAULT_DOMAIN_SEPARATOR)
            .unwrap()
            .0
            .clone();

        if num_generators + generator_offset > generators.len() {
            let num_extra_generators = num_generators + generator_offset - generators.len();
            let extended_generators = derive_generators(
                domain_separator.as_bytes(),
                num_extra_generators,
                generators.len(),
            );

            generators.extend_from_slice(&extended_generators);
        }

        generators
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GeneratorContext<E: SWCurveConfig> {
    pub(crate) offset: usize,
    pub(crate) domain_separator: &'static str,
    pub(crate) generators: GeneratorData<E>,
}

impl<E: SWCurveConfig> Default for GeneratorContext<E> {
    fn default() -> Self {
        Self {
            offset: 0,
            domain_separator: DEFAULT_DOMAIN_SEPARATOR,
            generators: GeneratorData::default(),
        }
    }
}
