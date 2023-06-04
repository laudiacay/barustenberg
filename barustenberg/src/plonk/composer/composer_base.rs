use std::{collections::HashMap, sync::Arc};

use ark_ec::AffineRepr;
use rand::RngCore;
use std::default::Default;

use crate::{
    plonk::proof_system::{proving_key::ProvingKey, verification_key::VerificationKey},
    srs::reference_string::{
        file_reference_string::FileReferenceStringFactory, BaseReferenceStringFactory,
        ReferenceStringFactory,
    },
};

use ark_ff::{FftField, Field};

pub(crate) const DUMMY_TAG: u32 = 0;
pub(crate) const REAL_VARIABLE: u32 = u32::MAX - 1;
pub(crate) const FIRST_VARIABLE_IN_CLASS: u32 = u32::MAX - 2;
pub(crate) const NUM_RESERVED_GATES: usize = 4;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[allow(clippy::enum_clike_unportable_variant)]
// Note that this will fail to compile on 32-bit systems
pub(crate) enum WireType {
    Left = 0,
    Right = 1 << 30,
    Output = 1 << 31,
    Fourth = 0xc0000000,
}
pub(crate) enum ComposerType {
    Standard,
    Turbo,
    Plookup,
    StandardHonk,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) struct CycleNode {
    pub(crate) gate_index: u32,
    pub(crate) wire_type: WireType,
}

pub(crate) struct SelectorProperties {
    pub(crate) name: String,
    pub(crate) requires_lagrange_base_polynomial: bool,
}

impl CycleNode {
    pub(crate) fn new(gate_index: u32, wire_type: WireType) -> Self {
        Self {
            gate_index,
            wire_type,
        }
    }
}

pub(crate) struct ComposerBase<'a, F: Field + FftField, G1Affine: AffineRepr, G2Affine: AffineRepr>
{
    pub(crate) num_gates: usize,
    crs_factory: Arc<dyn ReferenceStringFactory<G1Affine, G2Affine>>,
    num_selectors: usize,
    selectors: Vec<Vec<F>>,
    selector_properties: Vec<SelectorProperties>,
    rand_engine: Option<Box<dyn RngCore>>,
    circuit_proving_key: Option<Arc<ProvingKey<'a, F, G1Affine>>>,
    circuit_verification_key: Option<Arc<VerificationKey<'a, F>>>,
    w_l: Vec<u32>,
    w_r: Vec<u32>,
    w_o: Vec<u32>,
    w_4: Vec<u32>,
    failed: bool,
    _err: Option<String>,
    zero_idx: u32,
    public_inputs: Vec<u32>,
    variables: Vec<F>,
    /// index of next variable in equivalence class (=REAL_VARIABLE if you're last)
    next_var_index: Vec<u32>,
    /// index of  previous variable in equivalence class (=FIRST if you're in a cycle alone)
    prev_var_index: Vec<u32>,
    /// indices of corresponding real variables
    real_variable_index: Vec<u32>,
    real_variable_tags: Vec<u32>,
    current_tag: u32,
    /// The permutation on variable tags. See
    /// https://hackernoon.com/plookup-an-algorithm-widely-used-in-zkevm-ymw37qu
    /// DOCTODO: Check this link is sufficient
    tau: HashMap<u32, u32>,
    wire_copy_cycles: Vec<Vec<CycleNode>>,
    computed_witness: bool,
}

impl<'a, F: Field + FftField, G1Affine: AffineRepr, G2Affine: AffineRepr>
    ComposerBase<'a, F, G1Affine, G2Affine>
{
    pub(crate) fn new(
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
    ) -> Self {
        let crs_factory = Arc::new(FileReferenceStringFactory::new(
            "../srs_db/ignition".to_string(),
        ));
        Self::with_crs_factory(crs_factory, num_selectors, size_hint, selector_properties)
    }

    pub(crate) fn default() -> Self {
        Self {
            num_gates: 0,
            crs_factory: Arc::new(BaseReferenceStringFactory::<G1Affine, G2Affine>::default()),
            num_selectors: 0,
            selectors: Default::default(),
            selector_properties: Default::default(),
            rand_engine: Default::default(),
            circuit_proving_key: Default::default(),
            circuit_verification_key: Default::default(),
            w_l: vec![],
            w_r: vec![],
            w_o: vec![],
            w_4: vec![],
            public_inputs: vec![],
            variables: Default::default(),
            next_var_index: Default::default(),
            prev_var_index: Default::default(),
            real_variable_index: Default::default(),
            real_variable_tags: Default::default(),
            current_tag: DUMMY_TAG,
            tau: Default::default(),
            wire_copy_cycles: Default::default(),
            computed_witness: false,
            failed: Default::default(),
            _err: Default::default(),
            zero_idx: Default::default(),
        }
    }

    pub(crate) fn with_crs_factory(
        crs_factory: Arc<dyn ReferenceStringFactory<G1Affine, G2Affine>>,
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
    ) -> Self {
        let mut selfie = Self::default();
        selfie.selectors = vec![Vec::with_capacity(size_hint); num_selectors];
        selfie.rand_engine = None;
        selfie.circuit_proving_key = None;
        selfie.circuit_verification_key = None;
        selfie.num_selectors = num_selectors;
        selfie.selector_properties = selector_properties;
        selfie.crs_factory = crs_factory;
        selfie.num_gates = 0;
        selfie
    }
    pub(crate) fn with_keys(
        p_key: Arc<ProvingKey<'a, F, G1Affine>>,
        v_key: Arc<VerificationKey<'a, F>>,
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
    ) -> Self {
        let mut selfie = Self::default();
        selfie.selectors = vec![Vec::with_capacity(size_hint); num_selectors];
        selfie.rand_engine = None;
        selfie.circuit_proving_key = Some(p_key);
        selfie.circuit_verification_key = Some(v_key);
        selfie.num_selectors = num_selectors;
        selfie.selector_properties = selector_properties;
        selfie.num_gates = 0;
        selfie.crs_factory = Arc::new(FileReferenceStringFactory::new(
            "../srs_db/ignition".to_string(),
        ));
        selfie
    }
    pub(crate) fn get_first_variable_in_class(&self, index: usize) -> usize {
        let mut idx = index as u32;
        while self.prev_var_index[idx as usize] != FIRST_VARIABLE_IN_CLASS {
            idx = self.prev_var_index[idx as usize];
        }
        idx as usize
    }
    fn update_real_variable_indices(&mut self, index: u32, new_real_index: u32) {
        let mut cur_index = index;
        loop {
            self.real_variable_index[cur_index as usize] = new_real_index;
            cur_index = self.next_var_index[cur_index as usize];
            if cur_index == REAL_VARIABLE {
                break;
            }
        }
    }

    /// Get the value of the variable v_{index}.
    /// N.B. We should probably inline this.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the variable.
    ///
    /// # Returns
    ///
    /// * The value of the variable.
    #[inline]
    fn get_variable(&self, index: u32) -> F {
        assert!(self.variables.len() > index as usize);
        self.variables[self.real_variable_index[index as usize] as usize]
    }
    /// Get a reference to the variable v_{index}.
    ///
    /// We need this function for check_circuit functions.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the variable.
    ///
    /// # Returns
    ///
    /// * The value of the variable.
    #[inline]
    fn get_variable_reference(&self, index: u32) -> &F {
        assert!(self.variables.len() > index as usize);
        &self.variables[self.real_variable_index[index as usize] as usize]
    }

    fn get_public_input(&self, index: u32) -> F {
        self.get_variable(self.public_inputs[index as usize])
    }

    fn get_public_inputs(&self) -> Vec<F> {
        let mut result = Vec::new();
        for i in 0..self.get_num_public_inputs() {
            result.push(self.get_public_input(i.try_into().unwrap()));
        }
        result
    }
    /// Add a variable to variables
    ///
    /// # Arguments
    ///
    /// * `in_value` - The value of the variable
    ///
    /// # Returns
    ///
    /// * The index of the new variable in the variables vector
    fn add_variable(&mut self, in_value: F) -> u32 {
        self.variables.push(in_value);

        // By default, we assume each new variable belongs in its own copy-cycle. These defaults can be modified later
        // by `assert_equal`.
        let index = self.variables.len() as u32 - 1;
        self.real_variable_index.push(index);
        self.next_var_index.push(REAL_VARIABLE);
        self.prev_var_index.push(FIRST_VARIABLE_IN_CLASS);
        self.real_variable_tags.push(DUMMY_TAG);
        self.wire_copy_cycles.push(Vec::new()); // Note: this doesn't necessarily need to be initialised here. In fact, the
                                                // number of wire_copy_cycles often won't match the number of variables; its
                                                // non-zero entries will be a smaller vector of size equal to the number of
                                                // "real variables" (i.e. unique indices in the `real_variable_index` vector).
                                                // `wire_copy_cycles` could instead be instantiated during
                                                // compute_wire_copy_cycles(), although that would require a loop to identify
                                                // the number of unique "real variables".
        index
    }
    /// Add a public variable to variables
    ///
    /// The only difference between this and add_variable is that here it is
    /// also added to the public_inputs vector
    ///
    /// # Arguments
    ///
    /// * `in_value` - The value of the variable
    ///
    /// # Returns
    ///
    /// * The index of the new variable in the variables vector
    fn add_public_variable(&mut self, in_value: F) -> u32 {
        let index = self.add_variable(in_value);
        self.public_inputs.push(index);
        index
    }

    /// Make a witness variable public.
    ///
    /// # Arguments
    ///
    /// * `witness_index` - The index of the witness.
    fn set_public_input(&mut self, witness_index: u32) {
        let does_not_exist = self
            .public_inputs
            .iter()
            .all(|&input| input != witness_index);
        if does_not_exist {
            self.public_inputs.push(witness_index);
        }
        assert!(
            does_not_exist,
            "Attempted to set a public input that is already public!"
        );
    }

    // fn assert_equal(&mut self, a_idx: u32, b_idx: u32, msg: Option<&str>);

    // Add the implementation for `compute_wire_copy_cycles` and `compute_sigma_permutations` when needed.
    // These methods are generic and may require additional code and context.

    fn get_circuit_subgroup_size(&self, num_gates: usize) -> usize {
        let log2_n = num_gates.next_power_of_two().trailing_zeros() as usize;
        1 << log2_n
    }
    fn get_num_public_inputs(&self) -> usize {
        self.public_inputs.len()
    }

    fn assert_valid_variables(&self, variable_indices: &[u32]) {
        for &variable_index in variable_indices {
            assert!(self.is_valid_variable(variable_index));
        }
    }

    fn is_valid_variable(&self, variable_index: u32) -> bool {
        (self.variables.len() as u32) > variable_index
    }
}
// /**
//  * Composer Example: Pythagorean triples.
//  *
//  * (x_1 * x_1) + (x_2 * x_2) == (x_3 * x_3)
//  *
//  *************************************************************************************************************
//  *
//  * Notation as per the 'Constraint Systems' section of the Plonk paper:
//  *       ______________________
//  *      |                      |
//  *      |              a_1 = 1 | c_1 = 4
//  *      |  w_1 = x_{a_1} = x_1 | w_9 = x_{c_1} = x_4
//  *  x_1 |                      * ---------------------- x_4
//  *      |              b_1 = 1 | Gate 1                   |
//  *      |  w_5 = x_{b_1} = x_1 |                  a_4 = 4 | c_4 = 7
//  *      |______________________|      w_4 = x_{a_4} = x_4 | w_12 = x_{c_4} = x_7
//  *                                                        + ------------------------ x_7
//  *                                                b_4 = 5 | Gate 4                    =
//  *       ______________________       w_8 = x_{b_4} = x_5 |                           =
//  *      |                      |                          |                           =
//  *      |              a_2 = 2 | c_2 = 5                  |                           =
//  *      |  w_2 = x_{a_2} = x_2 | w_10 = x_{c_2} = x_5     |                           =
//  *  x_2 |                      * ---------------------- x_5                           =
//  *      |              b_2 = 2 | Gate 2                                               =
//  *      |  w_6 = x_{b_2} = x_2 |                                      These `=`s      =
//  *      |______________________|                                      symbolise a     =
//  *                                                                    copy-constraint =
//  *                                                                                    =
//  *       ______________________                                                       =
//  *      |                      |                                                      =
//  *      |              a_3 = 3 | c_3 = 6                                              =
//  *      |  w_3 = x_{a_3} = x_3 | w_11 = x_{c_3} = x_6                                 =
//  *  x_3 |                      * --------------------------------------------------- x_6
//  *      |              b_3 = 3 | Gate 3                                               ^
//  *      |  w_7 = x_{b_3} = x_3 |                           Suppose x_6 is the only____|
//  *      |______________________|                           public input.
//  *
//  * - 4 gates.
//  * - 7 "variables" or "witnesses", denoted by the x's, whose indices are pointed-to by the values of the a,b,c's.
//  *   #gates <= #variables <= 2 * #gates, always (see plonk paper).
//  * - 12 "wires" (inputs / outputs to gates) (= 3 * #gates; for a fan-in-2 gate), denoted by the w's.
//  *   Each wire takes the value of a variable (# wires >= # variables).
//  *
//  * a_1 = b_1 = 1
//  * a_2 = b_2 = 2
//  * a_3 = b_3 = 3
//  * a_4 = c_1 = 4
//  * b_4 = c_2 = 5
//  * c_3 =       6
//  * c_4 =       7
//  *   ^     ^   ^
//  *   |     |   |____ indices of the x's (variables (witnesses))
//  *   |_____|________ indices of the gates
//  *
//  * So x_{a_1} = x_1, etc.
//  *
//  * Then we have "wire values":
//  * w_1  = x_{a_1} = x_1
//  * w_2  = x_{a_2} = x_2
//  * w_3  = x_{a_3} = x_3
//  * w_4  = x_{a_4} = x_4
//  *
//  * w_5  = x_{b_1} = x_1
//  * w_6  = x_{b_2} = x_2
//  * w_7  = x_{b_3} = x_3
//  * w_8  = x_{b_4} = x_5
//  *
//  * w_9  = x_{c_1} = x_4
//  * w_10 = x_{c_2} = x_5
//  * w_11 = x_{c_3} = x_6
//  * w_12 = x_{c_4} = x_7
//  *
//  ****************************************************************************************************************
//  *
//  * Notation as per this codebase is different from the Plonk paper:
//  * This example is reproduced exactly in the stdlib field test `test_field_pythagorean`.
//  *
//  * variables[0] = 0 for all circuits <-- this gate is not shown in this diagram.
//  *                   ______________________
//  *                  |                      |
//  *                  |                      |
//  *                  |           w_l[1] = 1 | w_o[1] = 4
//  *     variables[1] |                      * ------------------- variables[4]
//  *                  |           w_r[1] = 1 | Gate 1                   |
//  *                  |                      |                          |
//  *                  |______________________|               w_l[4] = 4 | w_o[4] = 7
//  *                                                                    + --------------------- variables[7]
//  *                                                         w_r[4] = 5 | Gate 4                    =
//  *                   ______________________                           |                           =
//  *                  |                      |                          |                           =
//  *                  |                      |                          |                           =
//  *                  |           w_l[2] = 2 | w_o[2] = 5               |                           =
//  *     variables[2] |                      * ------------------- variables[5]                     =
//  *                  |           w_r[2] = 2 | Gate 2                                               =
//  *                  |                      |                                      These `=`s      =
//  *                  |______________________|                                      symbolise a     =
//  *                                                                                copy-constraint =
//  *                                                                                                =
//  *                   ______________________                                                       =
//  *                  |                      |                                                      =
//  *                  |                      |                                                      =
//  *                  |           w_l[3] = 3 | w_o[3] = 6                                           =
//  *     variables[3] |                      * ------------------------------------------------variables[6]
//  *                  |           w_r[3] = 3 | Gate 3                                               ^
//  *                  |                      |                           Suppose this is the only___|
//  *                  |______________________|                           public input.
//  *
//  * - 5 gates (4 gates plus the 'zero' gate).
//  * - 7 "variables" or "witnesses", stored in the `variables` vector.
//  *   #gates <= #variables <= 2 * #gates, always (see plonk paper).
//  * - 12 "wires" (inputs / outputs to gates) (= 3 * #gates; for a fan-in-2 gate), denoted by the w's.
//  *   Each wire takes the value of a variable (# wires >= # variables).
//  *
//  * ComposerBase naming conventions:
//  *   - n = 5 gates (4 gates plus the 'zero' gate).
//  *   - variables <-- A.k.a. "witnesses". Indices of this variables vector are referred to as `witness_indices`.
//  * Example of varibales in this example (a 3,4,5 triangle):
//  *   - variables      = [  0,   3,   4,   5,   9,  16,  25,  25]
//  *   - public_inputs  = [6] <-- points to variables[6].
//  *
//  * These `w`'s are called "wires". In fact, they're witness_indices; pointing to indices in the `variables` vector.
//  *   - w_l = [ 0, 1, 2, 3, 4]
//  *   - w_r = [ 0, 1, 2, 3, 5]
//  *   - w_o = [ 0, 4, 5, 6, 7]
//  *             ^ The 0th wires are 0, for the default gate which instantiates the first witness as equal to 0.
//  *   - w_4 = [ 0, 0, 0, 0, 0] <-- not used in this example.
//  *   - selectors = [
//  *                   q_m: [ 0, 1, 1, 1, 0],
//  *                   q_c: [ 0, 0, 0, 0, 0],
//  *                   q_1: [ 1, 0, 0, 0, 1],
//  *                   q_2: [ 0, 0, 0, 0, 1],
//  *                   q_3: [ 0,-1,-1,-1,-1],
//  *                   q_4: [ 0, 0, 0, 0, 0], <-- not used in this example; doesn't exist in Standard PlonK.
//  *                 ]
//  *
//  * These vectors imply copy-cycles between variables. ("copy-cycle" meaning "a set of variables which must always be
//  * equal"). The indices of these vectors correspond to those of the `variables` vector. Each index contains
//  * information about the corresponding variable.
//  *   - next_var_index = [ -1,  -1,  -1,  -1,  -1,  -1,  -1,   6]
//  *   - prev_var_index = [ -2,  -2,  -2,  -2,  -2,  -2,   7,  -2]
//  *   - real_var_index = [  0,   1,   2,   3,   4,   5,   6,   6] <-- Notice this repeated 6.
//  *
//  *   `-1` = "The variable at this index is considered the last in its cycle (no next variable exists)"
//  *          Note: we (arbitrarily) consider the "last" variable in a cycle to be the true representative of its
//  *          cycle, and so dub it the "real" variable of the cycle.
//  *   `-2` = "The variable at this index is considered the first in its cycle (no previous variable exists)"
//  *   Any other number denotes the index of another variable in the cycle = "The variable at this index is equal to
//  *   the variable at this other index".
//  *
//  * By default, when a variable is added to the composer, we assume the variable is in a copy-cycle of its own. So
//  * we set `next_var_index = -1`, `prev_var_index = -2`, `real_var_index` = the index of the variable in `variables`.
//  * You can see in our example that all but the last two indices of each *_index vector contain the default values.
//  * In our example, we have `variables[6].assert_equal(variables[7])`. The `assert_equal` function modifies the above
//  * vectors' entries for variables 6 & 7 to imply a copy-cycle between them. Arbitrarily, variables[7] is deemed the
//  * "first" in the cycle and variables[6] is considered the last (and hence the "real" variable which represents the
//  * cycle).
//  *
//  * By the time we get to computing wire copy-cycles, we need to allow for public_inputs, which in the plonk protocol
//  * are positioned to be the first witness values. `variables` doesn't include these public inputs (they're stored
//  * separately). In our example, we only have one public input, equal to `variables[6]`. We create a new "gate" for
//  * this 'public inputs' version of `variables[6]`, and push it to the front of our gates. (i.e. The first
//  * gate_index-es become gates for the public inputs, and all our `variables` occupy gates with gate_index-es shifted
//  * up by the number of public inputs (by 1 in this example)):
//  *   - wire_copy_cycles = [
//  *         // The i-th index of `wire_copy_cycles` details the set of wires which all equal
//  *         // variables[real_var_index[i]]. (I.e. equal to the i-th "real" variable):
//  *         [
//  *             { gate_index: 1, left   }, // w_l[1-#pub] = w_l[0] -> variables[0] = 0 <-- tag = 1 (id_mapping)
//  *             { gate_index: 1, right  }, // w_r[1-#pub] = w_r[0] -> variables[0] = 0
//  *             { gate_index: 1, output }, // w_o[1-#pub] = w_o[0] -> variables[0] = 0
//  *             { gate_index: 1, 4th },    // w_4[1-#pub] = w_4[0] -> variables[0] = 0
//  *             { gate_index: 2, 4th },    // w_4[2-#pub] = w_4[1] -> variables[0] = 0
//  *             { gate_index: 3, 4th },    // w_4[3-#pub] = w_4[2] -> variables[0] = 0
//  *             { gate_index: 4, 4th },    // w_4[4-#pub] = w_4[3] -> variables[0] = 0
//  *             { gate_index: 5, 4th },    // w_4[5-#pub] = w_4[4] -> variables[0] = 0
//  *         ],
//  *         [
//  *             { gate_index: 2, left   }, // w_l[2-#pub] = w_l[1] -> variables[1] = 3 <-- tag = 1
//  *             { gate_index: 2, right  }, // w_r[2-#pub] = w_r[1] -> variables[1] = 3
//  *         ],
//  *         [
//  *             { gate_index: 3, left   }, // w_l[3-#pub] = w_l[2] -> variables[2] = 4 <-- tag = 1
//  *             { gate_index: 3, right  }, // w_r[3-#pub] = w_r[2] -> variables[2] = 4
//  *         ],
//  *         [
//  *             { gate_index: 4, left   }, // w_l[4-#pub] = w_l[3] -> variables[3] = 5 <-- tag = 1
//  *             { gate_index: 4, right  }, // w_r[4-#pub] = w_r[3] -> variables[3] = 5
//  *         ],
//  *         [
//  *             { gate_index: 2, output }, // w_o[2-#pub] = w_o[1] -> variables[4] = 9 <-- tag = 1
//  *             { gate_index: 5, left   }, // w_l[5-#pub] = w_l[4] -> variables[4] = 9
//  *         ],
//  *         [
//  *             { gate_index: 3, output }, // w_o[3-#pub] = w_o[2] -> variables[5] = 16 <-- tag = 1
//  *             { gate_index: 5, right  }, // w_r[5-#pub] = w_r[4] -> variables[5] = 16
//  *         ],
//  *         [
//  *             { gate_index: 0, left   }, // public_inputs[0] -> w_l[0] -> variables[6] = 25 <-- tag = 1
//  *             { gate_index: 0, right  }, // public_inputs[0] -> w_r[0] -> variables[6] = 25
//  *             { gate_index: 4, output }, // w_o[4-#pub] = w_o[3] -> variables[6] = 25
//  *             { gate_index: 5, output }, // w_o[5-#pub] = w_o[4] -> variables[7] == variables[6] = 25
//  *         ],
//  *     ]
//  *
//  *
//  * Converting the wire_copy_cycles' objects into coordinates [row #, column #] (l=0, r=1, o=2, 4th = 3), and showing
//  * how the cycles permute with arrows:
//  * Note: the mappings (permutations) shown here (by arrows) are exactly those expressed by `sigma_mappings`.
//  * Note: `-*>` denotes when a sigma_mappings entry has `is_tag=true`.
//  * Note: [[ , ]] denotes an entry in sigma_mappings which has been modified due to is_tag=true or
//  *       is_public_input=true.
//  * Note: `-pub->` denotes a sigma_mappings entry from a left-wire which is a public input.
//  *
//  * Eg: [i, j] -> [k, l] means sigma_mappings[j][i]
//  *    has: subgroup_index = k, column_index = l, is_false = true and is_public_input = false
//  * Eg: [i, j] -*> [[k, l]]    means sigma_mappings[j][i]
//  *    has: subgroup_index = k, column_index = l, is_tag = true
//  * Eg: [i, j] -pub-> [[k, l]] means sigma_mappings[j][i]
//  *    has: subgroup_index = k, column_index = l, is_public_input = true
//  *
//  *     [
//  *         [1, 0] -> [1, 1] -> [1, 2] -> [1, 3] -> [2, 3] -> [3, 3] -> [4, 3] -> [5, 3] -*> [[0, 0]],
//  *         [2, 0] -> [2, 1] -*> [[0, 0]],
//  *         [3, 0] -> [3, 1] -*> [[0, 0]],
//  *         [4, 0] -> [4, 1] -*> [[0, 0]],
//  *         [2, 2] -> [5, 0] -*> [[0, 2]], <-- the column # (2) is ignored if is_tag=true.
//  *         [3, 2] -> [5, 1] -*> [[0, 2]], <-- the column # (2) is ignored if is_tag=true.
//  *         [0, 0] -----> [0, 1] -> [4, 2] -> [5, 2] -*> [[0, 0]],
//  *                -pub->[[0, 0]]
//  *         //         self^  ^ignored when is_public_input=true
//  *     ]   ^^^^^^
//  *         These are tagged with is_tag=true in `id_mappings`...
//  *
//  * Notice: the 0th row (gate) is the one public input of our example circuit. Two wires point to this public input:
//  * w_l[0] and w_r[0]. The reason _two_ wires do (and not just w_l[0]) is (I think) because of what we see above in
//  * the sigma_mappings data.
//  * - The [0,0] (w_l[0]) entry of sigma_mappings maps to [0,0], with is_public_input=true set. This is used by
//  * permutation.hpp to ensure the correct zeta_0 term is created for the ∆_PI term of the separate
//  * "plonk public inputs" paper.
//  * - The [0,1] (w_r[0]) entry of sigma_mappings maps to the next wire in the cycle ([4,2]=w_o[4-1]=w_o[3]). This is
//  * used to create the correct value for the sigma polynomial at S_σ_2(0).
//  *
//  * `id_mappings` maps every [row #, column #] to itself, except where is_tag=true where:
//  *
//  *  [1, 0] -*> [[0, 0]],
//  *  [2, 0] -*> [[0, 0]],
//  *  [3, 0] -*> [[0, 0]],
//  *  [4, 0] -*> [[0, 0]],
//  *  [2, 2] -*> [[0, 2]],
//  *  [3, 2] -*> [[0, 2]],
//  *  [0, 0] -*> [[0, 0]],
//  *                  ^this column data is ignored by permutation.hpp when is_tag=true
//  *
//  *
//  *
//  * The (subgroup.size() * program_width) elements of sigma_mappings are of the form:
//  * {
//  *     subgroup_index: j, // iterates over all rows in the subgroup
//  *     column_index: i, // l,r,o,4
//  *     is_public_input: false,
//  *     is_tag: false,
//  * }
//  *   - sigma_mappings = [
//  *         // The i-th index of sigma_mappings is the "column" index (l,r,o,4).
//  *         [
//  *             // The j-th index of sigma_mappings[i] is the subgroup_index or "row"
//  *             {
//  *                 subgroup_index: j,
//  *                 column_index: i,
//  *                 is_public_input: false,
//  *                 is_tag: false,
//  *             },
//  *         ],
//  *     ];
//  *
//  */
