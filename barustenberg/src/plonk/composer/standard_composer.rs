use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use super::composer_base::{ComposerBase, ComposerBaseData, ComposerType};
use crate::plonk::composer::composer_base::SelectorProperties;
use crate::plonk::proof_system::commitment_scheme::KateCommitmentScheme;
use crate::plonk::proof_system::prover::Prover;
use crate::plonk::proof_system::types::polynomial_manifest::STANDARD_MANIFEST_SIZE;
use crate::plonk::proof_system::types::prover_settings::Settings;
use crate::plonk::proof_system::types::prover_settings::StandardSettings;
use crate::plonk::proof_system::verification_key::VerificationKey;
use crate::plonk::proof_system::verifier::Verifier;
use crate::plonk::proof_system::widgets::random_widgets::permutation_widget::ProverPermutationWidget;
use crate::plonk::proof_system::widgets::transition_widgets::arithmetic_widget::ProverArithmeticWidget;
use crate::proof_system::arithmetization::{
    AccumulatorTriple, AddQuad, AddTriple, MulQuad, MulTriple, PolyTriple,
};
use crate::srs::reference_string::file_reference_string::FileReferenceStringFactory;
use crate::transcript::{Keccak256, Manifest, ManifestEntry, RoundManifest};
use crate::{
    plonk::proof_system::proving_key::ProvingKey, srs::reference_string::ReferenceStringFactory,
};

use ark_bn254::Fr;
use ark_ff::{BigInteger, Field, One, PrimeField, Zero};

#[derive(Default)]
pub(crate) struct StandardComposer<RSF: ReferenceStringFactory> {
    /// base data from composer
    cbd: Arc<RwLock<ComposerBaseData<RSF>>>,
    /// These are variables that we have used a gate on, to enforce that they are
    /// equal to a defined value.
    constant_variable_indices: HashMap<Fr, u32>,
    contains_recursive_proof: bool,
    own_type: ComposerType,
    settings: StandardSettings<Keccak256>,
}

impl<RSF: ReferenceStringFactory> ComposerBase for StandardComposer<RSF> {
    type RSF = RSF;

    #[inline(always)]
    fn composer_base_data(&self) -> Arc<RwLock<ComposerBaseData<Self::RSF>>> {
        self.cbd.clone()
    }

    fn with_crs_factory(
        crs_factory: Arc<RSF>,
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
    ) -> Self {
        let cbd = ComposerBaseData {
            selectors: vec![Vec::with_capacity(size_hint); num_selectors],
            num_selectors,
            selector_properties,
            crs_factory,
            ..Default::default()
        };

        let cbd = Arc::new(RwLock::new(cbd));
        Self {
            cbd,
            constant_variable_indices: HashMap::new(),
            contains_recursive_proof: false,
            own_type: ComposerType::Standard,
            settings: StandardSettings::default(),
        }
    }

    fn with_keys(
        p_key: Arc<RwLock<ProvingKey<Fr>>>,
        v_key: Arc<RwLock<VerificationKey<Fr>>>,
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
        crs_factory: Arc<Self::RSF>,
    ) -> Self {
        let cbd = ComposerBaseData {
            num_gates: 0,
            crs_factory,
            num_selectors,
            selector_properties,
            circuit_proving_key: Some(p_key),
            circuit_verification_key: Some(v_key),
            selectors: vec![Vec::with_capacity(size_hint); num_selectors],
            ..Default::default()
        };

        let cbd = Arc::new(RwLock::new(cbd));
        Self {
            cbd,
            constant_variable_indices: HashMap::new(),
            contains_recursive_proof: false,
            own_type: ComposerType::Standard,
            settings: StandardSettings::default(),
        }
    }
}

enum StandardSelectors {
    QM,
    QC,
    Q1,
    Q2,
    Q3,
}

impl StandardComposer<FileReferenceStringFactory> {
    fn new(
        num_selectors: usize,
        size_hint: usize,
        selector_properties: Vec<SelectorProperties>,
    ) -> Self {
        let crs_factory = Arc::new(FileReferenceStringFactory::new(
            "./src/srs_db/ignition".to_string(),
        ));
        Self::with_crs_factory(crs_factory, num_selectors, size_hint, selector_properties)
    }
}

impl<RSF: ReferenceStringFactory> StandardComposer<RSF> {
    /// Create an addition gate.
    ///
    /// # Arguments
    /// - `in` - An add_triple containing the indexes of variables to be placed into the
    /// wires w_l, w_r, w_o and addition coefficients to be placed into q_1, q_2, q_3, q_c.
    fn create_add_gate(&mut self, ins: &AddTriple<Fr>) {
        let mut cbd = self.cbd.write().unwrap();
        cbd.w_l.push(ins.a);
        cbd.w_r.push(ins.b);
        cbd.w_o.push(ins.c);
        cbd.selectors[StandardSelectors::QM as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q1 as usize].push(ins.a_scaling);
        cbd.selectors[StandardSelectors::Q2 as usize].push(ins.b_scaling);
        cbd.selectors[StandardSelectors::Q3 as usize].push(ins.c_scaling);
        cbd.selectors[StandardSelectors::QC as usize].push(ins.const_scaling);
        cbd.num_gates += 1;
    }

    /// Create a big addition gate.
    /// (a*a_c + b*b_c + c*c_c + d*d_c + q_c = 0)
    ///
    /// # Arguments
    /// - `in` - An add quad containing the indexes of variables a, b, c, d and
    /// the scaling factors.
    fn create_big_add_gate(&mut self, ins: &AddQuad<Fr>) {
        // (a terms + b terms = temp)
        // (c terms + d  terms + temp = 0 )
        let t0: Fr = self.get_variable(ins.a) * ins.a_scaling;
        let t1: Fr = self.get_variable(ins.b) * ins.b_scaling;
        let temp: Fr = t0 + t1;
        let temp_idx: u32 = self.add_variable(temp);

        self.create_add_gate(&AddTriple {
            a: ins.a,
            b: ins.b,
            c: temp_idx,
            a_scaling: ins.a_scaling,
            b_scaling: ins.b_scaling,
            c_scaling: -Fr::one(),
            const_scaling: Fr::zero(),
        });

        self.create_add_gate(&AddTriple {
            a: ins.c,
            b: ins.d,
            c: temp_idx,
            a_scaling: ins.c_scaling,
            b_scaling: ins.d_scaling,
            c_scaling: Fr::one(),
            const_scaling: ins.const_scaling,
        });
    }

    /// Create a balanced addition gate.
    /// (a*a_c + b*b_c + c*c_c + d*d_c + q_c = 0, where d is in [0,3])
    ///
    /// # Arguments
    /// - `in` - An add quad containing the indexes of variables a, b, c, d and
    /// the scaling factors.
    fn create_balanced_add_gate(&mut self, ins: &AddQuad<Fr>) {
        self.assert_valid_variables(&[ins.a, ins.b, ins.c, ins.d]);

        // (a terms + b terms = temp)
        // (c terms + d  terms + temp = 0 )
        let t0: Fr = self.get_variable(ins.a) * ins.a_scaling;
        let t1: Fr = self.get_variable(ins.b) * ins.b_scaling;
        let temp: Fr = t0 + t1;
        let temp_idx: u32 = self.add_variable(temp);

        let cbd = self.cbd.clone();
        let mut cbd = cbd.write().unwrap();

        cbd.w_l.push(ins.a);
        cbd.w_r.push(ins.b);
        cbd.w_o.push(temp_idx);
        cbd.selectors[StandardSelectors::QM as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q1 as usize].push(ins.a_scaling);
        cbd.selectors[StandardSelectors::Q2 as usize].push(ins.b_scaling);
        cbd.selectors[StandardSelectors::Q3 as usize].push(-Fr::one());
        cbd.selectors[StandardSelectors::QC as usize].push(Fr::zero());

        cbd.num_gates += 1;

        cbd.w_l.push(temp_idx);
        cbd.w_r.push(ins.c);
        cbd.w_o.push(ins.d);
        cbd.selectors[StandardSelectors::QM as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q1 as usize].push(Fr::one());
        cbd.selectors[StandardSelectors::Q2 as usize].push(ins.c_scaling);
        cbd.selectors[StandardSelectors::Q3 as usize].push(ins.d_scaling);
        cbd.selectors[StandardSelectors::QC as usize].push(ins.const_scaling);

        cbd.num_gates += 1;

        // in.d must be between 0 and 3
        // i.e. in.d * (in.d - 1) * (in.d - 2) = 0
        let temp_2: Fr = self.get_variable(ins.d).square() - self.get_variable(ins.d);
        let temp_2_idx: u32 = self.add_variable(temp_2);

        cbd.w_l.push(ins.d);
        cbd.w_r.push(ins.d);
        cbd.w_o.push(temp_2_idx);
        cbd.selectors[StandardSelectors::QM as usize].push(Fr::one());
        cbd.selectors[StandardSelectors::Q1 as usize].push(-Fr::one());
        cbd.selectors[StandardSelectors::Q2 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q3 as usize].push(-Fr::one());
        cbd.selectors[StandardSelectors::QC as usize].push(Fr::zero());

        cbd.num_gates += 1;

        let neg_two: Fr = -Fr::from(2);
        cbd.w_l.push(temp_2_idx);
        cbd.w_r.push(ins.d);
        let zero_idx = cbd.zero_idx;
        cbd.w_o.push(zero_idx);
        cbd.selectors[StandardSelectors::QM as usize].push(Fr::one());
        cbd.selectors[StandardSelectors::Q1 as usize].push(neg_two);
        cbd.selectors[StandardSelectors::Q2 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q3 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::QC as usize].push(Fr::zero());

        cbd.num_gates += 1;
    }

    /// Create a big addition gate with bit extraction.
    ///
    /// # Arguments
    /// - `in` - An add quad containing the indexes of variables a, b, c, d and
    /// the scaling factors.
    fn create_big_add_gate_with_bit_extraction(&mut self, ins: &AddQuad<Fr>) {
        let delta: Fr = self.get_variable(ins.d) * Fr::from(4);
        let delta = self.get_variable(ins.c) - delta;

        let delta_idx: u32 = self.add_variable(delta);
        let neg_four: Fr = -Fr::from(4);
        self.create_add_gate(&AddTriple {
            a: ins.c,
            b: ins.d,
            c: delta_idx,
            a_scaling: Fr::one(),
            b_scaling: neg_four,
            c_scaling: -Fr::one(),
            const_scaling: Fr::zero(),
        });

        let two: Fr = Fr::from(2);
        let seven: Fr = Fr::from(7);
        let nine: Fr = Fr::from(9);
        let r_0: Fr = (delta * nine) - ((delta.square() * two) + seven);
        let r_0_idx: u32 = self.add_variable(r_0);
        self.create_poly_gate(&PolyTriple {
            a: delta_idx,
            b: delta_idx,
            c: r_0_idx,
            q_m: -two,
            q_l: nine,
            q_r: Fr::zero(),
            q_o: -Fr::one(),
            q_c: -seven,
        });

        let r_1: Fr = r_0 * delta;
        let r_1_idx: u32 = self.add_variable(r_1);
        self.create_mul_gate(&MulTriple {
            a: r_0_idx,
            b: delta_idx,
            c: r_1_idx,
            mul_scaling: Fr::one(),
            c_scaling: -Fr::one(),
            const_scaling: Fr::zero(),
        });

        let r_2: Fr = r_1 + (self.get_variable(ins.d) * ins.d_scaling);
        let r_2_idx: u32 = self.add_variable(r_2);
        self.create_add_gate(&AddTriple {
            a: ins.d,
            b: r_1_idx,
            c: r_2_idx,
            a_scaling: ins.d_scaling,
            b_scaling: Fr::one(),
            c_scaling: -Fr::one(),
            const_scaling: Fr::zero(),
        });

        let new_add_quad = AddQuad {
            a: ins.a,
            b: ins.b,
            c: ins.c,
            d: r_2_idx,
            a_scaling: ins.a_scaling,
            b_scaling: ins.b_scaling,
            c_scaling: ins.c_scaling,
            d_scaling: Fr::one(),
            const_scaling: ins.const_scaling,
        };
        self.create_big_add_gate(&new_add_quad);
    }

    /// Create a big multiplication gate.
    ///
    /// # Arguments
    /// - `in` - A mul quad containing the indexes of variables a, b, c, d and
    /// the scaling factors.
    fn create_big_mul_gate(&mut self, ins: &MulQuad<Fr>) {
        let temp: Fr =
            (self.get_variable(ins.c) * ins.c_scaling) + (self.get_variable(ins.d) * ins.d_scaling);
        let temp_idx: u32 = self.add_variable(temp);
        self.create_add_gate(&AddTriple {
            a: ins.c,
            b: ins.d,
            c: temp_idx,
            a_scaling: ins.c_scaling,
            b_scaling: ins.d_scaling,
            c_scaling: -Fr::one(),
            const_scaling: Fr::zero(),
        });

        self.create_poly_gate(&PolyTriple {
            a: ins.a,
            b: ins.b,
            c: temp_idx,
            q_m: ins.mul_scaling,
            q_l: ins.a_scaling,
            q_r: ins.b_scaling,
            q_o: Fr::one(),
            q_c: ins.const_scaling,
        });
    }

    /// Create a multiplication gate.
    ///
    /// # Arguments
    /// - `in` - A mul_triple containing the indexes of variables to be placed into the wires w_l, w_r, w_o
    /// and scaling coefficients to be placed into q_m, q_3, q_c.
    fn create_mul_gate(&mut self, ins: &MulTriple<Fr>) {
        self.assert_valid_variables(&[ins.a, ins.b, ins.c]);

        let mut cbd = self.cbd.write().unwrap();

        cbd.w_l.push(ins.a);
        cbd.w_r.push(ins.b);
        cbd.w_o.push(ins.c);

        cbd.selectors[StandardSelectors::QM as usize].push(ins.mul_scaling);
        cbd.selectors[StandardSelectors::Q1 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q2 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q3 as usize].push(ins.c_scaling);
        cbd.selectors[StandardSelectors::QC as usize].push(ins.const_scaling);

        cbd.num_gates += 1;
    }

    /// Create a bool gate.
    /// This gate constrains a variable to two possible values: 0 or 1.
    ///
    /// # Arguments
    /// - `variable_index` - The index of the variable.
    fn create_bool_gate(&mut self, variable_index: u32) {
        self.assert_valid_variables(&[variable_index]);

        let mut cbd = self.cbd.write().unwrap();

        cbd.w_l.push(variable_index);
        cbd.w_r.push(variable_index);
        cbd.w_o.push(variable_index);

        cbd.selectors[StandardSelectors::QM as usize].push(Fr::one());
        cbd.selectors[StandardSelectors::Q1 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q2 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q3 as usize].push(-Fr::one());
        cbd.selectors[StandardSelectors::QC as usize].push(Fr::zero());

        cbd.num_gates += 1;
    }

    /// Create a gate where you set all the indexes and coefficients yourself.
    ///
    /// # Arguments
    /// - `in` - A poly_triple containing all the information.
    fn create_poly_gate(&mut self, ins: &PolyTriple<Fr>) {
        self.assert_valid_variables(&[ins.a, ins.b, ins.c]);

        let mut cbd = self.cbd.write().unwrap();

        cbd.w_l.push(ins.a);
        cbd.w_r.push(ins.b);
        cbd.w_o.push(ins.c);
        cbd.selectors[StandardSelectors::QM as usize].push(ins.q_m);
        cbd.selectors[StandardSelectors::Q1 as usize].push(ins.q_l);
        cbd.selectors[StandardSelectors::Q2 as usize].push(ins.q_r);
        cbd.selectors[StandardSelectors::Q3 as usize].push(ins.q_o);
        cbd.selectors[StandardSelectors::QC as usize].push(ins.q_c);

        cbd.num_gates += 1;
    }

    fn decompose_into_base4_accumulators(
        &mut self,
        witness_index: u32,
        num_bits: usize,
        msg: String,
    ) -> Vec<u32> {
        assert!(num_bits > 0, "num_bits must be greater than 0");

        let target = self.get_variable(witness_index).into_bigint();

        let mut accumulators: Vec<u32> = Vec::new();

        let mut num_quads = num_bits >> 1;
        num_quads = if (num_quads << 1) == num_bits {
            num_quads
        } else {
            num_quads + 1
        };

        let four = Fr::from(4);
        let mut accumulator = Fr::zero();
        let mut accumulator_idx: u32 = 0;

        for i in (0..num_quads).rev() {
            let is_edge_case = i == num_quads - 1 && ((num_bits & 1) == 1);
            let lo = target.get_bit(2 * i);
            let lo_idx = self.add_variable(if lo { Fr::one() } else { Fr::zero() });
            self.create_bool_gate(lo_idx);

            let quad_idx;

            if is_edge_case {
                quad_idx = lo_idx;
            } else {
                let hi = target.get_bit(2 * i + 1);
                let hi_idx = self.add_variable(if hi { Fr::one() } else { Fr::zero() });
                self.create_bool_gate(hi_idx);

                let quad = (if lo { 1 } else { 0 }) + (if hi { 2 } else { 0 });
                quad_idx = self.add_variable(Fr::from(quad));

                self.create_add_gate(&AddTriple {
                    a: lo_idx,
                    b: hi_idx,
                    c: quad_idx,
                    a_scaling: Fr::one(),
                    b_scaling: Fr::one() + Fr::one(),
                    c_scaling: -Fr::one(),
                    const_scaling: Fr::zero(),
                });
            }

            if i == num_quads - 1 {
                accumulators.push(quad_idx);
                accumulator = self.get_variable(quad_idx);
                accumulator_idx = quad_idx;
            } else {
                let mut new_accumulator = accumulator + accumulator;
                new_accumulator += new_accumulator;
                new_accumulator += self.get_variable(quad_idx);
                let new_accumulator_idx = self.add_variable(new_accumulator);
                self.create_add_gate(&AddTriple {
                    a: accumulator_idx,
                    b: quad_idx,
                    c: new_accumulator_idx,
                    a_scaling: four,
                    b_scaling: Fr::one(),
                    c_scaling: -Fr::one(),
                    const_scaling: Fr::zero(),
                });
                accumulators.push(new_accumulator_idx);
                accumulator = new_accumulator;
                accumulator_idx = new_accumulator_idx;
            }
        }

        self.assert_equal(witness_index, accumulator_idx, msg);
        accumulators
    }

    fn create_logic_constraint(
        &mut self,
        a: u32,
        b: u32,
        num_bits: usize,
        is_xor_gate: bool,
    ) -> AccumulatorTriple {
        self.assert_valid_variables(&vec![a, b][..]);

        let mut accumulators = AccumulatorTriple::default();

        let left_witness_value = self.get_variable(a).into_bigint();
        let right_witness_value = self.get_variable(b).into_bigint();

        let mut left_accumulator = Fr::zero();
        let mut right_accumulator = Fr::zero();
        let mut out_accumulator = Fr::zero();

        let mut left_accumulator_idx = self.cbd.read().unwrap().zero_idx;
        let mut right_accumulator_idx = self.cbd.read().unwrap().zero_idx;
        let mut out_accumulator_idx = self.cbd.read().unwrap().zero_idx;

        let four = Fr::from(4);
        let neg_two = -Fr::from(2);

        for i in (0..num_bits).rev().step_by(2) {
            let left_hi_val = left_witness_value.get_bit(i);
            let left_lo_val = left_witness_value.get_bit(i - 1);
            let right_hi_val = right_witness_value.get_bit(i);
            let right_lo_val = right_witness_value.get_bit(i - 1);

            let left_hi_idx = self.add_variable(if left_hi_val { Fr::one() } else { Fr::zero() });
            let left_lo_idx = self.add_variable(if left_lo_val { Fr::one() } else { Fr::zero() });
            let right_hi_idx = self.add_variable(if right_hi_val { Fr::one() } else { Fr::zero() });
            let right_lo_idx = self.add_variable(if right_lo_val { Fr::one() } else { Fr::zero() });

            let out_hi_val = if is_xor_gate {
                left_hi_val ^ right_hi_val
            } else {
                left_hi_val & right_hi_val
            };
            let out_lo_val = if is_xor_gate {
                left_lo_val ^ right_lo_val
            } else {
                left_lo_val & right_lo_val
            };

            let out_hi_idx = self.add_variable(if out_hi_val { Fr::one() } else { Fr::zero() });
            let out_lo_idx = self.add_variable(if out_lo_val { Fr::one() } else { Fr::zero() });

            self.create_bool_gate(left_hi_idx);
            self.create_bool_gate(right_hi_idx);
            self.create_bool_gate(out_hi_idx);

            self.create_bool_gate(left_lo_idx);
            self.create_bool_gate(right_lo_idx);
            self.create_bool_gate(out_lo_idx);

            // a & b = ab
            // a ^ b = a + b - ab
            self.create_poly_gate(&PolyTriple {
                a: left_hi_idx,
                b: right_hi_idx,
                c: out_hi_idx,
                q_m: if is_xor_gate { neg_two } else { Fr::one() },
                q_l: if is_xor_gate { Fr::one() } else { Fr::zero() },
                q_r: if is_xor_gate { Fr::one() } else { Fr::zero() },
                q_o: -Fr::one(),
                q_c: Fr::zero(),
            });

            self.create_poly_gate(&PolyTriple {
                a: left_lo_idx,
                b: right_lo_idx,
                c: out_lo_idx,
                q_m: if is_xor_gate { neg_two } else { Fr::one() },
                q_l: if is_xor_gate { Fr::one() } else { Fr::zero() },
                q_r: if is_xor_gate { Fr::one() } else { Fr::zero() },
                q_o: -Fr::one(),
                q_c: Fr::zero(),
            });

            let left_quad = self.get_variable(left_lo_idx)
                + self.get_variable(left_hi_idx)
                + self.get_variable(left_hi_idx);
            let right_quad = self.get_variable(right_lo_idx)
                + self.get_variable(right_hi_idx)
                + self.get_variable(right_hi_idx);
            let out_quad = self.get_variable(out_lo_idx)
                + self.get_variable(out_hi_idx)
                + self.get_variable(out_hi_idx);

            let left_quad_idx = self.add_variable(left_quad);
            let right_quad_idx = self.add_variable(right_quad);
            let out_quad_idx = self.add_variable(out_quad);

            let mut new_left_accumulator = left_accumulator + left_accumulator;
            new_left_accumulator += new_left_accumulator;
            new_left_accumulator += left_quad;
            let new_left_accumulator_idx = self.add_variable(new_left_accumulator);

            self.create_add_gate(&AddTriple {
                a: left_accumulator_idx,
                b: left_quad_idx,
                c: new_left_accumulator_idx,
                a_scaling: four,
                b_scaling: Fr::one(),
                c_scaling: -Fr::one(),
                const_scaling: Fr::zero(),
            });

            let mut new_right_accumulator = right_accumulator + right_accumulator;
            new_right_accumulator += new_right_accumulator;
            new_right_accumulator += right_quad;
            let new_right_accumulator_idx = self.add_variable(new_right_accumulator);

            self.create_add_gate(&AddTriple {
                a: right_accumulator_idx,
                b: right_quad_idx,
                c: new_right_accumulator_idx,
                a_scaling: four,
                b_scaling: Fr::one(),
                c_scaling: -Fr::one(),
                const_scaling: Fr::zero(),
            });

            let mut new_out_accumulator = out_accumulator + out_accumulator;
            new_out_accumulator += new_out_accumulator;
            new_out_accumulator += out_quad;
            let new_out_accumulator_idx = self.add_variable(new_out_accumulator);

            self.create_add_gate(&AddTriple {
                a: out_accumulator_idx,
                b: out_quad_idx,
                c: new_out_accumulator_idx,
                a_scaling: four,
                b_scaling: Fr::one(),
                c_scaling: -Fr::one(),
                const_scaling: Fr::zero(),
            });

            accumulators.left.push(new_left_accumulator_idx);
            accumulators.right.push(new_right_accumulator_idx);
            accumulators.out.push(new_out_accumulator_idx);

            left_accumulator = new_left_accumulator;
            left_accumulator_idx = new_left_accumulator_idx;

            right_accumulator = new_right_accumulator;
            right_accumulator_idx = new_right_accumulator_idx;

            out_accumulator = new_out_accumulator;
            out_accumulator_idx = new_out_accumulator_idx;
        }

        accumulators
    }

    fn fix_witness(&mut self, witness_index: u32, witness_value: &Fr) {
        self.assert_valid_variables(&vec![witness_index][..]);

        let mut cbd = self.cbd.write().unwrap();

        cbd.w_l.push(witness_index);
        let zero_idx = cbd.zero_idx;
        cbd.w_r.push(zero_idx);
        cbd.w_o.push(zero_idx);
        cbd.selectors[StandardSelectors::QM as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q1 as usize].push(Fr::one());
        cbd.selectors[StandardSelectors::Q2 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::Q3 as usize].push(Fr::zero());
        cbd.selectors[StandardSelectors::QC as usize].push(-*witness_value);
        cbd.num_gates += 1;
    }

    /// Stores a constant variable.
    ///
    /// # Arguments
    ///
    /// * `variable` - A constant value of type `Fr` to be stored.
    ///
    /// # Returns
    ///
    /// * Returns the index of the stored variable.

    fn put_constant_variable(&mut self, variable: Fr) -> u32 {
        #[allow(clippy::map_entry)]
        // todo fix map entry syntax
        if self.constant_variable_indices.contains_key(&variable) {
            *self.constant_variable_indices.get(&variable).unwrap()
        } else {
            let variable_index = self.add_variable(variable);
            self.fix_witness(variable_index, &variable);
            self.constant_variable_indices
                .insert(variable, variable_index);
            variable_index
        }
    }

    /// Creates a logical AND constraint between two variables over a certain number of bits.
    ///
    /// # Arguments
    ///
    /// * `a` - The index of the first variable.
    /// * `b` - The index of the second variable.
    /// * `num_bits` - The number of bits over which the AND operation is performed.
    ///
    /// # Returns
    ///
    /// * Returns an `AccumulatorTriple` that represents the AND constraint.

    fn create_and_constraint(&mut self, a: u32, b: u32, num_bits: usize) -> AccumulatorTriple {
        self.create_logic_constraint(a, b, num_bits, false)
    }
    /// Creates a logical XOR constraint between two variables over a certain number of bits.
    ///
    /// # Arguments
    ///
    /// * `a` - The index of the first variable.
    /// * `b` - The index of the second variable.
    /// * `num_bits` - The number of bits over which the XOR operation is performed.
    ///
    /// # Returns
    ///
    /// * Returns an `AccumulatorTriple` that represents the XOR constraint.

    fn create_xor_constraint(&mut self, a: u32, b: u32, num_bits: usize) -> AccumulatorTriple {
        self.create_logic_constraint(a, b, num_bits, true)
    }
    /// Computes the proving key.
    ///
    /// This function first checks if the circuit_proving_key is already available. If not,
    /// it computes the base proving key and the sigma permutations. It also sets up indices
    /// for recursive proof public inputs and flags if the key contains a recursive proof.
    ///
    /// # Returns
    ///
    /// * Returns a `Rc<ProvingKey>`, a reference counted proving key.

    fn compute_proving_key(&mut self) -> Arc<RwLock<ProvingKey<Fr>>> {
        let cbd = self.cbd.clone();
        let cbd = cbd.read().unwrap();

        if let Some(proving_key) = cbd.circuit_proving_key.clone() {
            return proving_key.clone();
        }
        let composer_type = self.own_type;
        self.compute_proving_key_base(composer_type, 0, 0);
        self.compute_sigma_permutations(cbd.circuit_proving_key.clone().unwrap(), 3, false);

        (*cbd.circuit_proving_key.clone().unwrap())
            .write()
            .unwrap()
            .recursive_proof_public_input_indices = cbd
            .circuit_proving_key
            .clone()
            .unwrap()
            .read()
            .unwrap()
            .recursive_proof_public_input_indices
            .clone();

        (*cbd.circuit_proving_key.clone().unwrap())
            .write()
            .unwrap()
            .contains_recursive_proof = self.contains_recursive_proof;

        cbd.circuit_proving_key.clone().unwrap()
    }

    /// Computes the verification key consisting of selector precommitments.
    ///
    /// If the `circuit_verification_key` already exists, it's returned. Otherwise,
    /// it first ensures the `circuit_proving_key` is computed and then computes
    /// the `circuit_verification_key` using the `compute_verification_key_base` method.
    /// It also sets up indices for recursive proof public inputs and flags if the key
    /// contains a recursive proof.
    ///
    /// # Returns
    ///
    /// * Returns an `Rc<VerificationKey>`, a reference counted verification key.
    fn compute_verification_key(&mut self) -> Result<Arc<RwLock<VerificationKey<Fr>>>> {
        let cbd = self.cbd.clone();
        let mut cbd = cbd.write().unwrap();

        if let Some(ref key) = cbd.circuit_verification_key {
            return Ok(key.clone());
        }
        if cbd.circuit_proving_key.is_none() {
            self.compute_proving_key();
        }

        let circuit_verification_key = self.compute_verification_key_base(
            cbd.circuit_proving_key.clone().unwrap(),
            cbd.crs_factory.get_verifier_crs()?.unwrap(),
        )?;
        cbd.circuit_verification_key = Some(circuit_verification_key.clone());

        {
            let mut verification_key = circuit_verification_key.write().unwrap();
            verification_key.composer_type = self.own_type;
            verification_key.recursive_proof_public_input_indices = cbd
                .circuit_proving_key
                .clone()
                .unwrap()
                .read()
                .unwrap()
                .recursive_proof_public_input_indices
                .clone();
            verification_key.contains_recursive_proof = self.contains_recursive_proof;
        }

        Ok(circuit_verification_key)
    }

    /// Computes the witness with standard settings (wire width = 3).
    ///
    /// Calls the `compute_witness_base` method from `ComposerBase` with the standard
    /// program width.
    fn compute_witness(&mut self) {
        self.compute_witness_base(self.settings.program_width(), None);
    }

    fn create_manifest(&self, num_public_inputs: usize) -> Manifest {
        let g1_size: usize = 64;
        let fr_size: usize = 32;
        let public_input_size = fr_size * num_public_inputs;

        let mut manifest = Manifest::default();
        // round 0
        manifest.add_round_manifest(RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "circuit_size".to_string(),
                    num_bytes: 4,
                    derived_by_verifier: true,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "public_input_size".to_string(),
                    num_bytes: 4,
                    derived_by_verifier: true,
                    challenge_map_index: 0,
                },
            ],
            challenge: "init".to_string(),
            num_challenges: 1,
            map_challenges: false,
        });

        // round 1
        manifest.add_round_manifest(RoundManifest {
            elements: vec![],
            challenge: "eta".to_string(),
            num_challenges: 0,
            map_challenges: false,
        });

        // round 2
        /*
                       {
                   { .name = "public_inputs", .num_bytes = public_input_size, .derived_by_verifier = false },
                   { .name = "W_1",           .num_bytes = g1_size,           .derived_by_verifier = false },
                   { .name = "W_2",           .num_bytes = g1_size,           .derived_by_verifier = false },
                   { .name = "W_3",           .num_bytes = g1_size,           .derived_by_verifier = false },
               },
               /* challenge_name = */ "beta",
               /* num_challenges_in = */ 2),
        */
        manifest.add_round_manifest(RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "public_inputs".to_string(),
                    num_bytes: public_input_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "W_1".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "W_2".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "W_3".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
            ],
            challenge: "beta".to_string(),
            num_challenges: 2,
            map_challenges: false,
        });

        // Round 3
        //   transcript::Manifest::RoundManifest(
        //     { { .name = "Z_PERM", .num_bytes = g1_size, .derived_by_verifier = false } },
        //     /* challenge_name = */ "alpha",
        //     /* num_challenges_in = */ 1),

        manifest.add_round_manifest(RoundManifest {
            elements: vec![ManifestEntry {
                name: "Z_PERM".to_string(),
                num_bytes: g1_size,
                derived_by_verifier: false,
                challenge_map_index: 0,
            }],
            challenge: "alpha".to_string(),
            num_challenges: 1,
            map_challenges: false,
        });

        // Round 4
        /*
                     transcript::Manifest::RoundManifest(
               { { .name = "T_1", .num_bytes = g1_size, .derived_by_verifier = false },
                 { .name = "T_2", .num_bytes = g1_size, .derived_by_verifier = false },
                 { .name = "T_3", .num_bytes = g1_size, .derived_by_verifier = false } },
               /* challenge_name = */ "z",
               /* num_challenges_in = */ 1),
        */
        manifest.add_round_manifest(RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "T_1".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "T_2".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "T_3".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
            ],
            challenge: "z".to_string(),
            num_challenges: 1,
            map_challenges: false,
        });

        // Round 5
        /*
        transcript::Manifest::RoundManifest(
                {
                    { .name = "t",            .num_bytes = fr_size, .derived_by_verifier = true,  .challenge_map_index = -1 },
                    { .name = "w_1",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 0 },
                    { .name = "w_2",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 1 },
                    { .name = "w_3",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 2 },
                    { .name = "sigma_1",      .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 3 },
                    { .name = "sigma_2",      .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 4 },
                    { .name = "sigma_3",      .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 5 },
                    { .name = "q_1",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 6 },
                    { .name = "q_2",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 7 },
                    { .name = "q_3",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 8 },
                    { .name = "q_m",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 9 },
                    { .name = "q_c",          .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 10 },
                    { .name = "z_perm",       .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = 11 },
                    { .name = "z_perm_omega", .num_bytes = fr_size, .derived_by_verifier = false, .challenge_map_index = -1 },
                },
                /* challenge_name = */ "nu",
                /* num_challenges_in = */ STANDARD_MANIFEST_SIZE,
                /* map_challenges_in = */ true),
         */
        manifest.add_round_manifest(RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "t".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: true,
                    challenge_map_index: -1,
                },
                ManifestEntry {
                    name: "w_1".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "w_2".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 1,
                },
                ManifestEntry {
                    name: "w_3".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 2,
                },
                ManifestEntry {
                    name: "sigma_1".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 3,
                },
                ManifestEntry {
                    name: "sigma_2".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 4,
                },
                ManifestEntry {
                    name: "sigma_3".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 5,
                },
                ManifestEntry {
                    name: "q_1".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 6,
                },
                ManifestEntry {
                    name: "q_2".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 7,
                },
                ManifestEntry {
                    name: "q_3".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 8,
                },
                ManifestEntry {
                    name: "q_m".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 9,
                },
                ManifestEntry {
                    name: "q_c".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 10,
                },
                ManifestEntry {
                    name: "z_perm".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: 11,
                },
                ManifestEntry {
                    name: "z_perm_omega".to_string(),
                    num_bytes: fr_size,
                    derived_by_verifier: false,
                    challenge_map_index: -1,
                },
            ],
            challenge: "nu".to_string(),
            num_challenges: *STANDARD_MANIFEST_SIZE,
            map_challenges: true,
        });

        // Round 6
        /*
                             transcript::Manifest::RoundManifest(
               { { .name = "PI_Z",       .num_bytes = g1_size, .derived_by_verifier = false },
                 { .name = "PI_Z_OMEGA", .num_bytes = g1_size, .derived_by_verifier = false } },
               /* challenge_name = */ "separator",
               /* num_challenges_in = */ 1) }
        */
        manifest.add_round_manifest(RoundManifest {
            elements: vec![
                ManifestEntry {
                    name: "PI_Z".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
                ManifestEntry {
                    name: "PI_Z_OMEGA".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                },
            ],
            challenge: "separator".to_string(),
            num_challenges: 1,
            map_challenges: false,
        });

        manifest
    }

    /// Creates a verifier.
    ///
    /// It first computes the verification key, then constructs a `Verifier`
    /// using the computed key and the manifest of public inputs.
    /// Finally, it adds a `KateCommitmentScheme` to the verifier and returns it.
    fn create_verifier(&mut self) -> Result<Verifier<Keccak256, StandardSettings<Keccak256>>> {
        let cbd = self.cbd.clone();
        let cbd = cbd.read().unwrap();

        self.compute_verification_key()?;
        let mut output_state = Verifier::new(
            Some(cbd.circuit_verification_key.as_ref().unwrap().clone()),
            self.create_manifest(cbd.public_inputs.len()),
        );

        output_state.commitment_scheme = Box::new(KateCommitmentScheme::new());

        Ok(output_state)
    }

    /// Creates a prover.
    ///
    /// This involves several steps:
    ///   1. Computing the starting polynomials (q_l, sigma, witness polynomials).
    ///   2. Initializing a `Prover` with the computed key and manifest of public inputs.
    ///   3. Adding `Permutation` and `Arithmetic` widgets to the prover.
    ///   4. Adding a `KateCommitmentScheme` to the prover.
    ///
    /// # Returns
    ///
    /// * Returns an initialized `Prover`.
    fn create_prover(&mut self) -> Prover<Keccak256, StandardSettings<Keccak256>> {
        self.compute_proving_key();
        self.compute_witness();

        let cbd = self.cbd.read().unwrap();

        let mut output_state = Prover::new(
            Some(Arc::clone(cbd.circuit_proving_key.as_ref().unwrap())),
            Some(self.create_manifest(cbd.public_inputs.len())),
            None,
        );

        output_state.random_widgets.push(Box::new(
            ProverPermutationWidget::<Keccak256, 3, false, 4>::new(
                cbd.circuit_proving_key.clone().unwrap(),
            ),
        ));

        let arithmetic_widget =
            ProverArithmeticWidget::new(cbd.circuit_proving_key.clone().unwrap());

        output_state
            .transition_widgets
            .push(Box::new(arithmetic_widget));

        output_state.commitment_scheme = KateCommitmentScheme::new();

        output_state
    }

    /// Asserts that the value at the given index equals a constant.
    ///
    /// If the value at the index `a_idx` is not equal to the constant `b` and the `failed` method returns `false`,
    /// it will call the `failure` method with the provided message.
    /// Then, it gets the index of the constant variable `b` and asserts the equality between variables at `a_idx` and `b_idx`.
    fn assert_equal_constant(&mut self, a_idx: usize, b: Fr, msg: String) {
        if self.cbd.read().unwrap().variables[a_idx] != b && !self.failed() {
            self.failure(msg.clone());
        }
        let b_idx = self.put_constant_variable(b);
        self.assert_equal(a_idx as u32, b_idx, msg);
    }
    /// Checks if all the circuit gates are correct given the witnesses.
    ///
    /// It iterates through each gate and checks if the identity holds.
    /// If the sum of the gate's selectors and variables isn't zero, the circuit is incorrect.
    ///
    /// # Returns
    ///
    /// * Returns `true` if the circuit is correct, `false` otherwise.
    fn check_circuit(&self) -> bool {
        let cbd = self.cbd.read().unwrap();

        for i in 0..cbd.num_gates {
            let left = self.get_variable(cbd.w_l[i]);
            let right = self.get_variable(cbd.w_r[i]);
            let output = self.get_variable(cbd.w_o[i]);
            let q_m = cbd.selectors[StandardSelectors::QM as usize][i];
            let q_1 = cbd.selectors[StandardSelectors::Q1 as usize][i];
            let q_2 = cbd.selectors[StandardSelectors::Q2 as usize][i];
            let q_3 = cbd.selectors[StandardSelectors::Q3 as usize][i];
            let q_c = cbd.selectors[StandardSelectors::QC as usize][i];
            let gate_sum = q_m * left * right + q_1 * left + q_2 * right + q_3 * output + q_c;
            if gate_sum != Fr::zero() {
                return false;
            }
        }
        true
    }
}
