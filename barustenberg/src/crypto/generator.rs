/*
#[derive(Default, PartialEq, Eq, PartialOrd, Debug)]
pub(crate) struct GeneratorIndex {
    pub(crate) index: usize,
    pub(crate) sub_index: usize,
}

//TODO: Check if this is necessary I overloaded the operator following:
// https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/crypto/generators/generator_data.hpp#L16
impl Ord for GeneratorIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.index, &self.sub_index).cmp(&(other.index, &other.sub_index))
    }
}

// generator indexes
const DEFAULT_GEN_1: GeneratorIndex = GeneratorIndex { index: 0, sub_index: 0 };
const DEFAULT_GEN_2: GeneratorIndex = GeneratorIndex { index: 0, sub_index: 1 };
const DEFAULT_GEN_3: GeneratorIndex = GeneratorIndex { index: 0, sub_index: 2 };
const DEFAULT_GEN_4: GeneratorIndex = GeneratorIndex { index: 0, sub_index: 3 };

#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct FixedBaseLadder {
    pub(crate) one: Affine,
    pub(crate) three: Affine,
    pub(crate) q_x_1: Fq,
    pub(crate) q_x_2: Fq,
    pub(crate) q_y_1: Fq,
    pub(crate) q_y_2: Fq,
}

const BIT_LENGTH: usize = 256;
const QUAD_LENGTH: usize = BIT_LENGTH / 2 + 1;
const AUX_LENGTH: usize = 2;

type Ladder = [FixedBaseLadder; QUAD_LENGTH + AUX_LENGTH];

#[derive(Debug)]
pub(crate) struct GeneratorData {
    pub(crate) generator: Affine,
    pub(crate) aux_generator: Affine,
    pub(crate) skew_generator: Affine,
    pub(crate) ladder: Ladder,
}

/*
    pub(crate) const fn get_ladder_offset(&self, num_bits: usize, offset: usize) -> &FixedBaseLadder {
        get_ladder_internal(&self.ladder, num_bits, 0)
    }

    pub(crate) const fn get_hash_ladder(&self, num_bits: usize) -> &FixedBaseLadder {
        get_ladder_internal(&self.ladder, num_bits, AUX_LENGTH)
    }
*/

/// The number of unique base points with default main index with precomputed ladders
const NUM_DEFAULT_GENERATORS: usize = 200;

/// Contains number of hash indices all of which support a fixed number of generators per index.
#[derive(Default, Debug)]
pub(crate) struct HashIndexParams {
    pub(crate) num_indices: usize,
    pub(crate) num_generators_per_index: usize,
}

impl HashIndexParams {
    pub(crate) const fn total_generators(&self) -> usize {
       self.num_indices * self.num_generators_per_index
    }
}

const LOW: HashIndexParams = HashIndexParams { num_indices: 32, num_generators_per_index: 8 };
const MID: HashIndexParams = HashIndexParams { num_indices: 8, num_generators_per_index: 16 };
const HIGH: HashIndexParams = HashIndexParams { num_indices: 4, num_generators_per_index: 48 };

const NUM_HASH_INDICES: usize = LOW.num_indices + MID.num_indices + HIGH.num_indices;
const NUM_INDEXED_GENERATORS: usize = LOW.total_generators() + MID.total_generators() + HIGH.total_generators();
const SIZE_OF_GENERATOR_DATA_ARRAY: usize = NUM_DEFAULT_GENERATORS + NUM_INDEXED_GENERATORS;
const NUM_GENERATOR_TYPES: usize = 3;
const NUM_GENERATORS: usize = SIZE_OF_GENERATOR_DATA_ARRAY * NUM_GENERATOR_TYPES;


//We once time initialize this table and use it for all hashing. I think given the circumstances this is perfect for a lazy_static but if we want to just init this in the main runner that works as well.
lazy_static!(
    pub(crate) static ref GENERATORS: PedersonGenerators = PedersonGenerators::new();
);
// In barustenberg there exists a shared ladder storing cached precomputed values.
pub(crate) struct PedersonGenerators {
    generator_data: Vec<GeneratorData>,
    g1_ladder: Ladder,
}

impl PedersonGenerators {
    /**
     * Precompute ladders and hash ladders
     *
     * `ladders` contains precomputed multiples of a base point
     *
     * Each entry in `ladders` is a `fixed_base_ladder` struct, which contains a pair of points,
     * `one` and `three`
     *
     * e.g. a size-4 `ladder` over a base point `P`, will have the following structure:
     *
     *    ladder[3].one = [P]
     *    ladder[3].three = 3[P]
     *    ladder[2].one = 4[P]
     *    ladder[2].three = 12[P]
     *    ladder[1].one = 16[P]
     *    ladder[1].three = 3*16[P]
     *    ladder[0].one = 64[P] + [P]
     *    ladder[0].three = 3*64[P]
     *
     * i.e. for a ladder size of `n`, we have the following:
     *
     *                        n - 1 - i
     *    ladder[i].one   = (4           ).[P]
     *                          n - 1 - i
     *    ladder[i].three = (3*4           ).[P]
     *
     * When a fixed-base scalar multiplier is decomposed into a size-2 WNAF, each ladder entry represents
     * the positive half of a WNAF table
     *
     * `hash_ladders` are stitched together from two `ladders` objects to preserve the uniqueness of a pedersen
     *hash. If a pedersen hash input is a 256-bit scalar, using a single generator point would mean that multiple
    *inputs would hash to the same output.
    *
    * e.g. if the grumpkin curve order is `n`, then hash(x) = hash(x + n) if we use a single generator
    *
    * For this reason, a hash ladder is built in a way that enables hashing the 252 higher bits of a 256 bit scalar
    * according to one generator and the four lower bits according to a second.
    *
    * Specifically,
    *
    *  1. For j=0,...,126, hash_ladders[i][j]=ladders[i][j] (i.e. generator  i)
    *  2. For j=127,128  hash_ladders[i][j]=aux_ladders[i][j] (i.e. auxiliary generator i)
    *
    * This is sufficient to create an injective hash for 256 bit strings
    * The reason we need 127 elements to hash 252 bits, or equivalently 126 quads, is that the first element of the
    *ladder is used simply to add the  "normalization factor" 4^{127}*[P] (so ladder[0].three is never used); this
    *addition makes all resultant scalars positive. When wanting to hash e.g. 254 instead of 256 bits, we will
    *start the ladder one step forward - this happends in `get_ladder_internal`
    **/
    fn new() -> Self {
        let (generators, aux_generators, skew_generators) = derive_generators::<NUM_GENERATORS>();
        let generator_data = (0..(NUM_DEFAULT_GENERATORS + SIZE_OF_GENERATOR_DATA_ARRAY)).into_iter().map(|i| {
            let mut ladder = Self::compute_fixed_base_ladder::<QUAD_LENGTH>(generators[i]);
            let aux_ladder = Self::compute_fixed_base_ladder::<AUX_LENGTH>(aux_generators[i]);
            for j in 0..AUX_LENGTH {
                ladder[j + QUAD_LENGTH] = aux_ladder[j];
            }
            GeneratorData {
                generator: generators[i],
                aux_generator: aux_generators[i],
                skew_generator: skew_generators[i],
                ladder,
            }
        }).collect();

        //TODO: change this to be Affine::one()
        let g1_ladder = Self::compute_fixed_base_ladder::<QUAD_LENGTH>(Affine::zero());
        PedersonGenerators { generator_data, g1_ladder }
    }

/**
 * @brief Returns a reference to the generator data for the specified generator index.
 * The generator index is composed of an index and sub-index. The index specifies
 * which hash index the generator belongs to, and the sub-index specifies the
 * position of the generator within the hash index.
 *
 * The generator data is stored in a global array of generator_data objects, which
 * is initialized lazily when the function is called for the first time. The global
 * array includes both default generators and user-defined generators.
 *
 * If the specified index is 0, the sub-index is used to look up the corresponding
 * default generator in the global array. Otherwise, the global index of the generator
 * is calculated based on the index and sub-index, and used to look up the corresponding
 * user-defined generator in the global array.
 *
 * The function throws an exception if the specified index is invalid.
 *
 * @param index The generator index, consisting of an index and sub-index.
 * @return A reference to the generator data for the specified generator index.
 * @throws An exception if the specified index is invalid.
 *
 * @note TODO: Write a generator indexing example
 */
    pub(crate) fn get_generator_data(&self, index: GeneratorIndex) -> &GeneratorData {
        if index.index == 0 {
            assert!(index.sub_index < NUM_DEFAULT_GENERATORS);
            return &self.generator_data[index.sub_index];
        }
        assert!(index.index <= NUM_HASH_INDICES);
        let mut global_index_offset = 0;
        if 0 < index.index && index.index <= LOW.num_indices {
            assert!(index.sub_index < LOW.num_generators_per_index);
            let local_index_offset = 0;
            let generator_count_offset = 0;
            global_index_offset = generator_count_offset + (index.index - local_index_offset - 1) * LOW.num_generators_per_index;
        } else if index.index <= (LOW.num_indices + MID.num_indices) {
            // Calculate the global index of the generator for the MID hash index
            assert!(index.sub_index < MID.num_generators_per_index);
            let local_index_offset = LOW.num_indices;
            let generator_count_offset = LOW.total_generators();
            global_index_offset = generator_count_offset + (index.index - local_index_offset - 1) * MID.num_generators_per_index;

        } else if index.index <= (LOW.num_indices + MID.num_indices + HIGH.num_indices) {
            // Calculate the global index of the generator for the HIGH hash index
            let local_index_offset = LOW.num_indices + MID.num_indices;
            let generator_count_offset = LOW.total_generators() + MID.total_generators();
            assert!(index.sub_index < HIGH.num_generators_per_index);
            global_index_offset = generator_count_offset + (index.index - local_index_offset - 1) * HIGH.num_generators_per_index;
        } else {
            // Throw an exception for invalid index values
            panic!("invalid hash index {:?}", index.index);
        }

        // Return a reference to the user-defined generator with the specified index and sub-index
        &self.generator_data[NUM_DEFAULT_GENERATORS + global_index_offset + index.sub_index]
    }


    /*
    pub(crate) const fn get_g1_ladder(&self, num_bits: usize) -> &FixedBaseLadder {
        get_ladder_internal(&self.g1_ladder, num_bits, 0)
    }
    */


    fn compute_fixed_base_ladder<const LADDER_LENGTH: usize>(generator: Affine) -> Ladder {
        assert!(LADDER_LENGTH < QUAD_LENGTH + AUX_LENGTH);
        let mut ladder_temp = Vec::new();
        let mut ladder = [FixedBaseLadder::default(); QUAD_LENGTH + AUX_LENGTH];
        //Precisely this should G1Affine but... ark-grumpkin doesn't implement that lol
        //TODO: probs rename to accumulator to remove redundant allocation.
        let mut accumulator = generator.into_group();
        for i in 0..LADDER_LENGTH {
            ladder_temp[i] = accumulator;
            accumulator.double_in_place();
            ladder_temp[LADDER_LENGTH + i] = ladder_temp[i] + accumulator;
            accumulator.double_in_place();
        }

        //TODO: figure out better way to accomplish this indexing with iterators
        for i in 0..LADDER_LENGTH {
            ladder[LADDER_LENGTH - 1 - i].one.x = ladder_temp[i].x;
            ladder[LADDER_LENGTH - 1 - i].one.x = ladder_temp[i].y;
            ladder[LADDER_LENGTH - 1 - i].three.x = ladder_temp[LADDER_LENGTH + i].x;
            ladder[LADDER_LENGTH - 1 - i].three.y = ladder_temp[LADDER_LENGTH + i].y;
        }

        let eight_inverse = Fq::from(8).inverse().unwrap();
        let mut y_denominators = Vec::new();
        for i in 0..LADDER_LENGTH {
            // TODO: For the love of god see how we can burn these unwraps in a Joan of Arc kind of pyre. Can we start a roman catholic, spanish inquisition style witch hunt against the arkworks code base.
            // They'd never expect it... cause no one expects the spanish inquisition...
            // Link: https://www.youtube.com/watch?v=T2ncJ6ciGyM
            let x_beta = ladder[i].one.x().unwrap();
            let x_gamma = ladder[i].three.x().unwrap();
            let y_beta = ladder[i].one.x().unwrap();
            let y_gamma = ladder[i].three.x().unwrap();
            //TODO: simplify this
            let mut x_beta_times_nine = x_beta + x_beta;
            x_beta_times_nine = x_beta_times_nine + x_beta_times_nine;
            x_beta_times_nine = x_beta_times_nine + x_beta_times_nine;
            x_beta_times_nine = x_beta_times_nine + x_beta;

            let x_alpha_1 = (x_gamma - x_beta) * eight_inverse;
            let x_alpha_2 = (x_beta_times_nine - x_gamma) * eight_inverse;

            let t0 = x_beta - x_gamma;
            y_denominators[i] = (t0 + t0) + t0;

            let y_alpha_1 = ((y_beta + y_beta) + y_beta) - y_gamma;
            let mut t1 = x_gamma * y_beta;
            t1 = (t1 + t1) + t1;
            let y_alpha_2 = (x_beta * y_gamma) - t1;
            ladder[i].q_x_1 = x_alpha_1;
            ladder[i].q_x_2 = x_alpha_2;
            ladder[i].q_y_1 = y_alpha_1;
            ladder[i].q_y_2 = y_alpha_2;
        }
        batch_inversion(&mut y_denominators);
        for i in 0..LADDER_LENGTH {
            ladder[i].q_y_1 *= y_denominators[i];
            ladder[i].q_y_2 *= y_denominators[i];
        }
        ladder
    }
}

#[inline(always)]
fn derive_grumpkin_generators<const N: usize>() -> Vec<Affine> {
    let mut generators = Vec::new();
    let mut count = 0;
    let mut seed = 0u32;
    while count < N {
        seed += 1;
        //TODO: This should use hash to curve but thats tricky and needs to be thought about more given using arkworks.
        let candidate = Affine::from_random_bytes(&seed.to_le_bytes()).unwrap();
        if candidate.is_on_curve() && !candidate.is_zero() {
            generators[count] = candidate;
            count += 1;
        }
    }
    generators
}

//TODO: simplify this once tests work
#[inline(always)]
fn derive_generators<const N: usize>() -> (Vec<Affine>, Vec<Affine>, Vec<Affine>) {
    assert!(N % NUM_GENERATOR_TYPES == 0);
    let mut generators = Vec::new();
    let mut aux_generators = Vec::new();
    let mut skew_generators = Vec::new();
    let res = derive_grumpkin_generators::<N>();
    res.chunks(3).for_each(|g| {
        generators.push(g[0]);
        aux_generators.push(g[1]);
        skew_generators.push(g[2]);
    });
(generators, aux_generators, skew_generators)
}


const fn get_ladder_offset(num_bits: usize, offset: usize) -> usize {
    let mut n: usize;
    if num_bits == 0 {
        n = 0;
    } else {
        n = (num_bits - 1) >> 1;
        if ((n << 1) + 1) < num_bits {
            n += 1;
        }
    }
    //TODO: remove clone
    QUAD_LENGTH + offset - n - 1
}

const WNAF_MASK: u64 = 0x7fffffffu64;

//USE ARKWORKS!!!!
pub(crate) fn fixed_base_scalar_mul<const NUM_BITS: u64>(f: Fr, index: usize) -> Affine {
    let gen_data = GENERATORS.get_generator_data(GeneratorIndex { index, sub_index: 0 });
    assert!(f != Fr::zero());

    let num_quads_base = (NUM_BITS - 1) >> 1;
    let num_quads = if (num_quads_base << 1) + 1 < NUM_BITS { (num_quads_base + 1) as usize } else { num_quads_base as usize };
    let num_wnaf_bits = (num_quads << 1) + 1;

    let offset = get_ladder_offset(NUM_BITS as usize, 0);

    let mut wnaf_entries = vec![0u64; num_quads + 2];
    //TODO: maybe have this be u64
    let skew = 0u64;
    fixed_wnaf::<GrumpkinConfig, 1, 2>(num_wnaf_bits as u64, f, &mut wnaf_entries, &skew, 0);

    let mut accumulator = Affine::zero();
    if skew == 1 {
        accumulator = (accumulator - gen_data.generator).into();
    }

    for i in 0..num_quads {
        let entry = wnaf_entries[i + 1];
        let point_to_add = if (entry & WNAF_MASK) == 1 { gen_data.ladder[offset + i + 1].three} else { gen_data.ladder[offset + i + 1].one };
        let predicate = (entry >> 31) & 1;
        //TODO: this mixed addition shit
    }

    accumulator
}



#[cfg(test)]
mod test {
    use ark_ec::{short_weierstrass::Affine, AffineRepr, CurveGroup};

    use super::*;

    #[test]
    fn hash_ladder_structure() {
        let index = GeneratorIndex { index: 2, sub_index: 0};
        let gen_data = GENERATORS.get_generator_data(index);
        let p = gen_data.generator;
        let q = gen_data.aux_generator;
        /*
         * Check if the hash ladder is structured in the following way:
         * +-----+------------+----------------+
         * | idx | one        | three          |
         * +-----+------------+----------------+
         * | 0   | 4^{n-2}[P] | (3*4^{n-2})[P] |
         * | 1   | 4^{n-3}[P] | (3*4^{n-3})[P] |
         * | 2   | 4^{n-4}[P] | (3*4^{n-4})[P] |
         * | .   | .          | .              |
         * | .   | .          | .              |
         * | .   | .          | .              |
         * | 124 | 4[P]       | (3*4)[P]       |
         * | 125 | 1[P]       | (3*1)[P]       |
         * +-----+------------+----------------+
         * | 126 | 4[Q]       | (3*4)[Q]       |
         * | 127 | 1[Q]       | (3*1)[Q]       |
         * +-----+------------+----------------+
         *
         * Here num_quads is n = 127.
         */
        let num_quads = 127;
        let offset = get_ladder_offset(254, AUX_LENGTH);

        //Check auxiliary generator powers
        let mut acc_q = q.into_group();
        for i in ((num_quads - 2)..=num_quads).rev() {
            let local_acc_q = acc_q;
            assert_eq!(acc_q, gen_data.ladder[offset + i].one);
            //LOL fuck arkworks with the bullshit that you can't double an affine element in place cause its not implemented for affine.
            acc_q.double_in_place();
            assert_eq!(acc_q + local_acc_q, gen_data.ladder[offset + i].three);
            acc_q.double_in_place();
        }

        //Check normal generator powers
        let mut acc_p = p.into_group();
        for i in ((num_quads - 2)..=num_quads).rev() {
            let local_acc_p = acc_p;
            assert_eq!(acc_p, gen_data.ladder[offset + i].one);
            //LOL fuck arkworks with the bullshit that you can't double an affine element in place cause its not implemented for affine.
            acc_p.double_in_place();
            assert_eq!(acc_p + local_acc_p, gen_data.ladder[offset + i].three);
            acc_p.double_in_place();
        }

        let scalar = Affine::zero();

        assert_eq!(gen_data.ladder[0].one.into_group(), scalar);
    }
}
*/

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
