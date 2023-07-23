pub(crate) mod crypto {
    pub mod pedersen_hash {
        use crate::generators::{get_generator_data, generator_index_t, fixed_base_ladder};
        use crate::grumpkin::{g1::element as g1_element, fq};
        use crate::barretenberg::fr;
        use crate::barretenberg::wnaf::fixed_wnaf;
        use std::ops::SubAssign;

        pub fn hash_single(input: fr, index: generator_index_t) -> g1_element {
            let gen_data = get_generator_data(index);
            let mut scalar_multiplier = input.from_montgomery_form();

            const NUM_BITS: usize = 254;
            const NUM_QUADS_BASE: usize = (NUM_BITS - 1) >> 1;
            const NUM_QUADS: usize = if ((NUM_QUADS_BASE << 1) + 1 < NUM_BITS) { NUM_QUADS_BASE + 1 } else { NUM_QUADS_BASE };
            const NUM_WNAF_BITS: usize = (NUM_QUADS << 1) + 1;

            let ladder = gen_data.get_hash_ladder(NUM_BITS);

            let mut wnaf_entries = [0u64; NUM_QUADS + 2];
            let mut skew = false;
            fixed_wnaf::<NUM_WNAF_BITS, 1, 2>(&mut scalar_multiplier.data[0], &mut wnaf_entries[0], &mut skew, 0);

            let mut accumulator = g1_element::new(ladder[0].one);
            if skew {
                accumulator.sub_assign(gen_data.skew_generator);
            }

            for i in 0..NUM_QUADS {
                let entry = wnaf_entries[i + 1];
                let point_to_add = if (entry & WNAF_MASK) == 1 { 
                    ladder[i + 1].three 
                } else { 
                    ladder[i + 1].one 
                };
                let predicate = (entry >> 31) & 1;
                accumulator.self_mixed_add_or_sub(point_to_add, predicate);
            }
            accumulator
        }

        pub fn hash_multiple(inputs: Vec<fq>, hash_index: usize) -> fq {
            assert!(inputs.len() < (1 << 16), "too many inputs for 16 bit index");
            let mut out = Vec::with_capacity(inputs.len());

            for i in 0..inputs.len() {
                let index = generator_index_t { hash_index, i };
                out.push(hash_single(inputs[i], index));
            }

            let mut r = out[0];
            for i in 1..inputs.len() {
                r += out[i];
            }
            let result = if r.is_point_at_infinity() {
                g1_element::new(0, 0)
            } else {
                g1_element::new(r)
            };
            result.x
        }
    }
}
