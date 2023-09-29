use core::num;
use std::{
    borrow::Borrow,
    collections::HashMap,
    sync::{Arc, RwLock}, borrow::Borrow,
};

use ark_bn254::Fr;
use ark_ff::Zero;
use rand::RngCore;

use anyhow::Result;

use crate::{
    plonk::proof_system::{
        proving_key::ProvingKey,
        types::{polynomial_manifest::PolynomialSource, PolynomialManifest},
        utils::permutation::{PermutationMapping, PermutationSubgroupElement},
        verification_key::VerificationKey,
    },
    polynomials::Polynomial,
    srs::reference_string::ReferenceStringFactory,
    transcript::Manifest,
};

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

#[derive(Debug, Default, Clone, Copy)]
pub(crate) enum ComposerType {
    #[default]
    Standard,
    _Turbo,
    _Plookup,
    _StandardHonk,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) struct CycleNode {
    pub(crate) gate_index: u32,
    pub(crate) wire_type: WireType,
}

impl CycleNode {
    pub(crate) fn new(gate_index: u32, wire_type: WireType) -> Self {
        Self {
            gate_index,
            wire_type,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SelectorProperties {
    pub(crate) name: String,
    pub(crate) requires_lagrange_base_polynomial: bool,
}

#[derive(Default)]
pub(crate) struct ComposerBaseData<RSF: ReferenceStringFactory> {
    pub(crate) num_gates: usize,
    pub(crate) crs_factory: Arc<RSF>,
    pub(crate) num_selectors: usize,
    pub(crate) selectors: Vec<Vec<Fr>>,
    pub(crate) selector_properties: Vec<SelectorProperties>,
    pub(crate) rand_engine: Option<Box<dyn RngCore>>,
    pub(crate) circuit_proving_key: Option<Arc<RwLock<ProvingKey<Fr>>>>,
    pub(crate) circuit_verification_key: Option<Arc<RwLock<VerificationKey<Fr>>>>,
    pub(crate) w_l: Vec<u32>,
    pub(crate) w_r: Vec<u32>,
    pub(crate) w_o: Vec<u32>,
    pub(crate) w_4: Vec<u32>,
    pub(crate) failed: bool,
    pub(crate) _err: Option<String>,
    pub(crate) zero_idx: u32,
    pub(crate) public_inputs: Vec<u32>,
    pub(crate) variables: Vec<Fr>,
    /// index of next variable in equivalence class (=REAL_VARIABLE if you're last)
    pub(crate) next_var_index: Vec<u32>,
    /// index of  previous variable in equivalence class (=FIRST if you're in a cycle alone)
    pub(crate) prev_var_index: Vec<u32>,
    /// indices of corresponding real variables
    pub(crate) real_variable_index: Vec<u32>,
    pub(crate) real_variable_tags: Vec<u32>,
    pub(crate) current_tag: u32,
    /// The permutation on variable tags. See
    /// https://hackernoon.com/plookup-an-algorithm-widely-used-in-zkevm-ymw37qu
    /// DOCTODO: Check this link is sufficient
    pub(crate) tau: HashMap<u32, u32>,
    pub(crate) wire_copy_cycles: Vec<Vec<CycleNode>>,
    pub(crate) computed_witness: bool,
}

pub(crate) trait ComposerBase {
    type RSF: ReferenceStringFactory;

    const G1_SIZE: usize = 64;
    const FR_SIZE: usize = 32;

    fn create_manifest(&self, num_public_inputs: usize) -> Manifest;

    fn with_crs_factory(
        crs_factory: Arc<Self::RSF>,
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
    ) -> Self;

    fn with_keys(
        p_key: Arc<RwLock<ProvingKey<Fr>>>,
        v_key: Arc<RwLock<VerificationKey<Fr>>>,
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
        crs_factory: Arc<Self::RSF>,
    ) -> Self;

    /// should be inlined in implementations
    fn composer_base_data(&self) -> Arc<RwLock<ComposerBaseData<Self::RSF>>>;

    fn get_first_variable_in_class(&self, index: usize) -> usize {
        let mut idx = index as u32;
        let cbd = self.composer_base_data();
        let cbd = (*cbd).read().unwrap();
        while cbd.prev_var_index[idx as usize] != FIRST_VARIABLE_IN_CLASS {
            idx = cbd.prev_var_index[idx as usize];
        }
        idx as usize
    }
    fn update_real_variable_indices(&mut self, index: u32, new_real_index: u32) {
        let mut cur_index = index;
        loop {
            let cbd = self.composer_base_data();
            let mut cbd = (*cbd).write().unwrap();
            cbd.real_variable_index[cur_index as usize] = new_real_index;

            cbd.real_variable_index[cur_index as usize] = new_real_index;
            cur_index = cbd.next_var_index[cur_index as usize];
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
    fn get_variable(&self, index: u32) -> Fr {
        let cbd = self.composer_base_data();
        let cbd = (*cbd).read().unwrap();
        assert!(cbd.variables.len() > index as usize);
        cbd.variables[cbd.real_variable_index[index as usize] as usize]
    }

    fn get_public_input(&self, index: u32) -> Fr {
        let cbd = self.composer_base_data();
        let cbd = (*cbd).read().unwrap();
        self.get_variable(cbd.public_inputs[index as usize])
    }

    fn get_public_inputs(&self) -> Vec<Fr> {
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
    fn add_variable(&mut self, in_value: Fr) -> u32 {
        let cbd = self.composer_base_data();
        let mut cbd = (*cbd).write().unwrap();

        cbd.variables.push(in_value);

        // By default, we assume each new variable belongs in its own copy-cycle. These defaults can be modified later
        // by `assert_equal`.
        let index = cbd.variables.len() as u32 - 1;
        cbd.real_variable_index.push(index);
        cbd.next_var_index.push(REAL_VARIABLE);
        cbd.prev_var_index.push(FIRST_VARIABLE_IN_CLASS);
        cbd.real_variable_tags.push(DUMMY_TAG);
        cbd.wire_copy_cycles.push(Vec::new()); // Note: this doesn't necessarily need to be initialised here. In fact, the
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
    fn add_public_variable(&mut self, in_value: Fr) -> u32 {
        let index = self.add_variable(in_value);
        (*self.composer_base_data())
            .write()
            .unwrap()
            .public_inputs
            .push(index);
        index
    }

    /// Make a witness variable public.
    ///
    /// # Arguments
    ///
    /// * `witness_index` - The index of the witness.
    fn set_public_input(&mut self, witness_index: u32) {
        let cbd = self.composer_base_data();
        let mut cbd = (*cbd).write().unwrap();
        let does_not_exist = cbd
            .public_inputs
            .iter()
            .all(|&input| input != witness_index);
        if does_not_exist {
            cbd.public_inputs.push(witness_index);
        }
        assert!(
            does_not_exist,
            "Attempted to set a public input that is already public!"
        );
    }

    fn assert_equal(&mut self, a_idx: u32, b_idx: u32, msg: String) {
        let cbd = self.composer_base_data();
        let mut cbd = (*cbd).write().unwrap();
        self.assert_valid_variables(&[a_idx, b_idx]);
        let values_equal = self.get_variable(a_idx) == self.get_variable(b_idx);
        if !values_equal && !self.failed() {
            self.failure(msg.clone());
        }
        let a_real_idx = cbd.real_variable_index[a_idx as usize];
        let b_real_idx = cbd.real_variable_index[b_idx as usize];
        if a_real_idx == b_real_idx {
            return;
        }
        let b_start_idx = self.get_first_variable_in_class(b_idx as usize);
        self.update_real_variable_indices(b_start_idx as u32, a_real_idx);
        let a_start_idx = self.get_first_variable_in_class(a_idx as usize);
        cbd.next_var_index[b_real_idx as usize] = a_start_idx as u32;
        cbd.prev_var_index[a_start_idx] = b_real_idx;
        let no_tag_clash = cbd.real_variable_tags[a_real_idx as usize] == DUMMY_TAG
            || cbd.real_variable_tags[b_real_idx as usize] == DUMMY_TAG
            || cbd.real_variable_tags[a_real_idx as usize]
                == cbd.real_variable_tags[b_real_idx as usize];
        if !no_tag_clash && !self.failed() {
            self.failure(msg);
        }
        if cbd.real_variable_tags[a_real_idx as usize] == DUMMY_TAG {
            cbd.real_variable_tags[a_real_idx as usize] =
                cbd.real_variable_tags[b_real_idx as usize];
        }
    }

    /// Compute wire copy cycles
    ///
    /// First set all wire_copy_cycles corresponding to public_inputs to point to themselves.
    /// Then go through all witnesses in w_l, w_r, w_o and w_4 (if program width is > 3) and
    /// add them to cycles of their real indexes.
    ///
    /// # Arguments
    ///
    /// * `program_width` - Program width
    fn compute_wire_copy_cycles(&mut self, program_width: usize) {
        // NOTE: for additional Flavors such as Ultra and Goblin plonk we need to define additional offsets.
        // See: https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/proof_system/composer/permutation_lib.hpp#L88
        //      https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/proof_system/composer/permutation_lib.hpp#L99

        let cbd = self.composer_base_data().clone();
        let mut cbd = (*cbd).write().unwrap();
        // We use the permutation argument to enforce the public input variables to be equal to values provided by the
        // verifier. The convension we use is to place the public input values as the first rows of witness vectors.
        // More specifically, we set the LEFT and RIGHT wires to be the public inputs and set the other elements of the row
        // to 0. All selectors are zero at these rows, so they are fully unconstrained. The "real" gates that follow can use
        // references to these variables.
        //
        // The copy cycle for the i-th public variable looks like
        //   (i) -> (n+i) -> (i') -> ... -> (i'')
        // (Using the convention that W^L_i = W_i and W^R_i = W_{n+i}, W^O_i = W_{2n+i})
        //
        // This loop initializes the i-th cycle with (i) -> (n+i), meaning that we always expect W^L_i = W^R_i,
        // for all i s.t. row i defines a public input.
        for i in 0..cbd.public_inputs.len() {
            let public_input_index =
                cbd.real_variable_index[cbd.public_inputs[i] as usize] as usize;
            //NOTE: for goblin plonk the gate_index of each cycle node is i + public_offsets
            let cycle = &mut cbd.wire_copy_cycles[public_input_index];
            cycle.push(CycleNode {
                gate_index: i as u32,
                wire_type: WireType::Left,
            });
            cycle.push(CycleNode {
                gate_index: i as u32,
                wire_type: WireType::Right,
            });
        }

        let num_public_inputs = self.get_public_inputs().len() as u32;
        // Iterate over all variables of the "real" gates, and add a corresponding node to the cycle for that variable
        for i in 0..cbd.num_gates {
            let w_1_index = cbd.real_variable_index[cbd.w_l[i] as usize] as usize;
            let w_2_index = cbd.real_variable_index[cbd.w_r[i] as usize] as usize;
            let w_3_index = cbd.real_variable_index[cbd.w_o[i] as usize] as usize;

            cbd.wire_copy_cycles[w_1_index].push(CycleNode {
                gate_index: i as u32 + num_public_inputs,
                wire_type: WireType::Left,
            });
            cbd.wire_copy_cycles[w_2_index].push(CycleNode {
                gate_index: i as u32 + num_public_inputs,
                wire_type: WireType::Right,
            });
            cbd.wire_copy_cycles[w_3_index].push(CycleNode {
                gate_index: i as u32 + num_public_inputs,
                wire_type: WireType::Output,
            });

            if program_width > 3 {
                let w_4_index = cbd.real_variable_index[cbd.w_4[i] as usize] as usize;
                cbd.wire_copy_cycles[w_4_index].push(CycleNode {
                    gate_index: i as u32 + num_public_inputs,
                    wire_type: WireType::Fourth,
                });
            }
        }
    }

    /// Compute sigma and id permutation polynomials in lagrange base.
    ///
    /// # Arguments
    ///
    /// * `key` - Proving key.
    /// * `with_tags` - Closely linked with `id_poly`: id_poly is a flag that describes whether we're using
    ///                 Vitalik's trick of using trivial identity permutation polynomials (id_poly = false). OR whether the identity
    ///                 permutation polynomials are circuit-specific and stored in the proving/verification key (id_poly = true).
    fn compute_sigma_permutations(
        &mut self,
        proving_key: Arc<RwLock<ProvingKey<Fr>>>,
        program_width: usize,
        with_tags: bool,
    ) -> PermutationMapping {
        // // Compute wire copy cycles for public and private variables
        self.compute_wire_copy_cycles(program_width);
        let mut mapping = PermutationMapping::default();
        let proving_key = proving_key.read().unwrap();
        let cbd = self.composer_base_data().clone();
        let cbd = (*cbd).read().unwrap();

        // Instantiate the sigma and id mappings by reserving enough space and pushing 'default' permutation subgroup
        // elements that point to themselves.
        for i in 0..program_width {
            mapping.sigmas[i].reserve(proving_key.circuit_size);
            if with_tags {
                mapping.ids[i].reserve(proving_key.circuit_size);
            }
            for j in 0..proving_key.circuit_size {
                mapping.sigmas[i].push(PermutationSubgroupElement {
                    row_index: j as u32,
                    column_index: i as u8,
                    is_public_input: false,
                    is_tag: false,
                });
                if with_tags {
                    mapping.ids[i].push(PermutationSubgroupElement {
                        row_index: j as u32,
                        column_index: i as u8,
                        is_public_input: false,
                        is_tag: false,
                    });
                }
            }
        }

        // Represents the index of a variable in circuit_constructor.variables (needed only for generalized)
        let real_variable_tags = &cbd.real_variable_tags;
        for (i, cycle) in cbd.wire_copy_cycles.iter().enumerate() {
            for node_idx in 0..cycle.len() {
                // Get the indices of the current node and next node in the cycle
                let current_cycle_node = cycle[node_idx];
                // If current node is the last one in the cycle, then the next one is the first one
                let next_cycle_node_index = if node_idx == cycle.len() - 1 {
                    0
                } else {
                    node_idx + 1
                };
                let next_cycle_node = cycle[next_cycle_node_index];
                let current_row = current_cycle_node.gate_index as usize;
                let next_row = next_cycle_node.gate_index;

                let current_column = current_cycle_node.wire_type as usize;
                let next_column = next_cycle_node.wire_type;
                // Point current node to the next node
                //TODO: Do we need to convert these using the following:
                // let row =  current_row & 0xffffff;
                //  let column = current_column >> 30;
                mapping.sigmas[current_column][current_row] = PermutationSubgroupElement {
                    row_index: next_row,
                    column_index: next_column as u8,
                    is_public_input: false,
                    is_tag: false,
                };

                if with_tags {
                    let first_node = node_idx == 0;
                    let last_node = next_cycle_node_index == 0;

                    if first_node {
                        mapping.ids[current_column][current_row].is_tag = true;
                        mapping.ids[current_column][current_row].row_index = real_variable_tags[i];
                    }
                    if last_node {
                        mapping.sigmas[current_column][current_row].is_tag = true;

                        // TODO(Zac): yikes, std::maps (tau) are expensive. Can we find a way to get rid of this?
                        // YES THIS IS AN UNWRAP! FUCK IT WE BOOL!
                        mapping.sigmas[current_column][current_row].row_index =
                            *cbd.tau.get(&real_variable_tags[i]).unwrap();
                    }
                }
            }
        }

        // Add information about public inputs to the computation
        // The public inputs are placed at the top of the execution trace, potentially offset by a zero row.
        // NOTE: for other flavors of plonk such as goblin plonk the index of public inputs needs to be offset.
        // See: https://github.com/AztecProtocol/barretenberg/blob/master/cpp/src/barretenberg/proof_system/composer/permutation_lib.hpp#L252
        for i in 0..cbd.public_inputs.len() {
            mapping.sigmas[0][i].row_index = i as u32;
            mapping.sigmas[0][i].column_index = 0;
            mapping.sigmas[0][i].is_public_input = true;
            if mapping.sigmas[0][i].is_tag {
                println!("MAPPING IS BOTH A TAG AND A PUBLIC INPUT");
            }
        }
        mapping
    }

    fn compute_witness_base(&mut self, program_width: usize, minimum_circuit_size: Option<usize>) {
        let minimum_circuit_size = minimum_circuit_size.unwrap_or(0);

        let cbd = self.composer_base_data().clone();
        let mut cbd = (*cbd).write().unwrap();

        if cbd.computed_witness {
            return;
        }

        let total_num_gates = cbd.num_gates + cbd.public_inputs.len();
        let total_num_gates = total_num_gates.max(minimum_circuit_size);
        let subgroup_size = self.get_circuit_subgroup_size(total_num_gates + NUM_RESERVED_GATES);

        let zero_idx = cbd.zero_idx;

        for _ in total_num_gates..subgroup_size {
            cbd.w_l.push(zero_idx);
            cbd.w_r.push(zero_idx);
            cbd.w_o.push(zero_idx);
        }
        if program_width > 3 {
            for _ in total_num_gates..subgroup_size {
                cbd.w_4.push(zero_idx);
            }
        }

        let mut w_1_lagrange = Polynomial::new(subgroup_size);
        let mut w_2_lagrange = Polynomial::new(subgroup_size);
        let mut w_3_lagrange = Polynomial::new(subgroup_size);
        let mut w_4_lagrange = if program_width > 3 {
            Polynomial::new(subgroup_size)
        } else {
            Polynomial::new(0)
        };

        for i in 0..cbd.public_inputs.len() {
            w_1_lagrange[i] = self.get_variable(cbd.public_inputs[i]);
            w_2_lagrange[i] = self.get_variable(cbd.public_inputs[i]);
            w_3_lagrange[i] = Fr::zero();
            if program_width > 3 {
                w_4_lagrange[i] = Fr::zero();
            }
        }

        for i in cbd.public_inputs.len()..subgroup_size {
            w_1_lagrange[i] = self.get_variable(cbd.w_l[i - cbd.public_inputs.len()]);
            w_2_lagrange[i] = self.get_variable(cbd.w_r[i - cbd.public_inputs.len()]);
            w_3_lagrange[i] = self.get_variable(cbd.w_o[i - cbd.public_inputs.len()]);
            if program_width > 3 {
                w_4_lagrange[i] = self.get_variable(cbd.w_4[i - cbd.public_inputs.len()]);
            }
        }

        let cpk = cbd.circuit_proving_key.clone().unwrap();
        let mut pkey = cpk.as_ref().write().unwrap();

        pkey.polynomial_store
            .insert(&"w_1_lagrange".to_string(), w_1_lagrange);
        pkey.polynomial_store
            .insert(&"w_2_lagrange".to_string(), w_2_lagrange);
        pkey.polynomial_store
            .insert(&"w_3_lagrange".to_string(), w_3_lagrange);
        if program_width > 3 {
            pkey.polynomial_store
                .insert(&"w_4_lagrange".to_string(), w_4_lagrange);
        }

        cbd.computed_witness = true;
    }

    fn get_circuit_subgroup_size(&self, num_gates: usize) -> usize {
        let log2_n = num_gates.next_power_of_two().trailing_zeros() as usize;
        1 << log2_n
    }

    fn get_num_public_inputs(&self) -> usize {
        let cbd = self.composer_base_data();
        let cbd = cbd.read().unwrap();
        cbd.public_inputs.len()
    }

    fn assert_valid_variables(&self, variable_indices: &[u32]) {
        for &variable_index in variable_indices {
            assert!(self.is_valid_variable(variable_index));
        }
    }

    fn is_valid_variable(&self, variable_index: u32) -> bool {
        let cbd = self.composer_base_data();
        let cbd = cbd.read().unwrap();
        (cbd.variables.len() as u32) > variable_index
    }

    fn set_err(&mut self, err: String) {
        let cbd = self.composer_base_data();
        let mut cbd = (*cbd).write().unwrap();
        cbd._err = Some(err);
    }

    fn failure(&mut self, err: String) {
        let cbd = self.composer_base_data();
        let mut cbd = (*cbd).write().unwrap();
        cbd.failed = true;
        self.set_err(err)
    }

    fn failed(&self) -> bool {
        let cbd = self.composer_base_data();
        let cbd = cbd.read().unwrap();
        cbd.failed
    }

    /**
     * Compute proving key base.
     *
     * 1. Load crs.
     * 2. Initialize this.circuit_proving_key.
     * 3. Create constraint selector polynomials from each of this composer's `selectors` vectors and add them to the
     * proving key.
     *
     * N.B. Need to add the fix for coefficients
     *
     * @param minimum_circuit_size Used as the total number of gates when larger than n + count of public inputs.
     * @param num_reserved_gates The number of reserved gates.
     * @return Pointer to the initialized proving key updated with selector polynomials.
     * */
    fn compute_proving_key_base(
        &mut self,
        composer_type: ComposerType,
        minimum_circuit_size: usize,
        num_reserved_gates: usize,
    ) -> Arc<RwLock<ProvingKey<Fr>>> {
        let cbd = self.composer_base_data().clone();
        let mut cbd = (*cbd).write().unwrap();
        let num_filled_gates = cbd.num_gates + cbd.public_inputs.len();
        let total_num_gates = if minimum_circuit_size > num_filled_gates {
            minimum_circuit_size
        } else {
            num_filled_gates
        };
        let subgroup_size = self.get_circuit_subgroup_size(total_num_gates + num_reserved_gates); // next power of 2

        // In the case of standard plonk, if 4 roots are cut out of the vanishing polynomial,
        // then the degree of the quotient polynomial t(X) is 3n. This implies that the degree
        // of the constituent t_{high} of t(X) must be n (as against (n - 1) for other composer types).
        // Thus, to commit to t_{high}, we need the crs size to be (n + 1) for standard plonk.
        //
        // For more explanation about the degree of t(X), see
        // ./src/barretenberg/plonk/proof_system/prover/prover.cpp/ProverBase::compute_quotient_commitments
        //
        let crs = cbd
            .crs_factory
            .get_prover_crs(subgroup_size + 1)
            .unwrap()
            .clone()
            .unwrap();
        // initialize proving key
        cbd.circuit_proving_key = Some(Arc::new(RwLock::new(ProvingKey::new(
            subgroup_size,
            cbd.public_inputs.len(),
            crs,
            composer_type,
        ))));

        let n_selectors = cbd.num_selectors;
        let n_public_inputs = cbd.public_inputs.len();
        let n_gates = cbd.num_gates;
        for i in 0..n_selectors {
            let properties = cbd.selector_properties[i].clone();
            let selector_values = &mut cbd.selectors[i];
            assert_eq!(n_gates, selector_values.len());
            // Fill unfilled gates' selector values with zeroes (stopping 1 short; the last value will be nonzero).
            for _j in num_filled_gates..(subgroup_size - 1) {
                selector_values.push(Fr::zero());
            }
            // Add a nonzero value at the end of each selector vector. This ensures that, if the selector would otherwise
            // have been 'empty':
            //    1) that its commitment won't be the point at infinity. We avoid the point at
            //    infinity in the native verifier because this is an edge case in the recursive verifier circuit, and we
            //    want the logic to be consistent between both verifiers.
            //    2) that its commitment won't be equal to any other selectors' commitments (which would break biggroup
            //    operations when verifying snarks within a circuit, since doubling is not directly supported). This in turn
            //    ensures that when we commit to a selector, we will never get the point at infinity.
            //
            // Note: Setting the selector to nonzero would ordinarily make the proof fail if we did not have a satisfying
            // constraint. This is not the case for the last selector position, as it is never checked in the proving
            // system; observe that we cut out 4 roots and only use 3 for zero knowledge. The last root, corresponds to this
            // position.
            selector_values.push(Fr::from((i + 1) as u32));
            // Compute lagrange form of selector polynomial

            let mut selector_poly_lagrange = Polynomial::new(subgroup_size);
            for k in 0..n_public_inputs {
                selector_poly_lagrange[k] = Fr::zero();
            }
            for k in n_public_inputs..subgroup_size {
                selector_poly_lagrange[k] = selector_values[k - n_public_inputs];
            }
            // Compute monomial form of selector polynomial

            let mut selector_poly: Polynomial<Fr> = Polynomial::new(subgroup_size);

            let pkey = cbd.circuit_proving_key.clone().unwrap();
            let mut pkey = (*pkey).write().unwrap();

            pkey.small_domain.ifft(
                &mut selector_poly_lagrange.coefficients[..],
                &mut selector_poly.coefficients[..],
            );

            // compute coset fft of selector polynomial
            let mut selector_poly_fft = selector_poly.clone();
            selector_poly_fft.resize(subgroup_size * 4 + 4, Fr::zero());
            pkey.large_domain
                .coset_fft_inplace(&mut selector_poly_fft.coefficients);

            if properties.requires_lagrange_base_polynomial {
                pkey.polynomial_store.put(
                    properties.name.clone() + "_lagrange",
                    selector_poly_lagrange,
                );
            }
            pkey.polynomial_store
                .put(properties.name.clone(), selector_poly);
            pkey.polynomial_store
                .put(properties.name.clone() + "_fft", selector_poly_fft);
        }
        cbd.circuit_proving_key.clone().unwrap()
    }

    /**
     * @brief Computes the verification key by computing the:
     * (1) commitments to the selector and permutation polynomials,
     * (2) sets the polynomial manifest using the data from proving key.
     */
    fn compute_verification_key_base(
        &mut self,
        proving_key: Arc<RwLock<ProvingKey<Fr>>>,
        vrs: Arc<RwLock<<<Self as ComposerBase>::RSF as ReferenceStringFactory>::Ver>>,
    ) -> Result<Arc<RwLock<VerificationKey<Fr>>>> {
        let proving_key = proving_key.read().unwrap();

        let circuit_verification_key = Arc::new(RwLock::new(VerificationKey::new(
            proving_key.circuit_size,
            proving_key.num_public_inputs,
            vrs.clone(),
            proving_key.composer_type,
        )));

        for i in 0..proving_key.polynomial_manifest.len() {
            let selector_poly_info = &proving_key.polynomial_manifest[i.into()];

            let selector_poly_label = selector_poly_info.polynomial_label.clone();
            let selector_commitment_label = selector_poly_info.commitment_label.clone();

            if selector_poly_info.source == PolynomialSource::Selector
                || selector_poly_info.source == PolynomialSource::Permutation
            {
                // Fetch the constraint selector polynomial in its coefficient form.
                let selector_poly = proving_key.polynomial_store.get(&selector_poly_label)?;
                let mut selector_poly = (*selector_poly).write().unwrap();
                let selector_poly_coefficients = &mut selector_poly.coefficients;

                let reference_string = (*proving_key.reference_string).write().unwrap();
                let mut pippenger_runtime_state = proving_key.pippenger_runtime_state.clone();

                // Commit to the constraint selector polynomial and insert the commitment in the verification key.
                let selector_poly_commitment = pippenger_runtime_state.pippenger(
                    selector_poly_coefficients,
                    &reference_string.get_monomial_points(),
                    proving_key.circuit_size,
                    false,
                );

                (*circuit_verification_key)
                    .write()
                    .unwrap()
                    .commitments
                    .insert(selector_commitment_label, selector_poly_commitment);
            }
        }

        // Set the polynomial manifest in verification key.
        (*circuit_verification_key)
            .write()
            .unwrap()
            .polynomial_manifest = PolynomialManifest::new_from_type(proving_key.composer_type);

        Ok(circuit_verification_key)
    }
}

// // Set the polynomial manifest in verification key.
// circuit_verification_key->polynomial_manifest = PolynomialManifest(proving_key->composer_type);

// return circuit_verification_key;
// }

// }
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
//  * This example is reproduced exactly in the stdlib FieldExt test `test_FieldExt_pythagorean`.
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