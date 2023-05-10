use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::Read;
use std::sync::Arc;
use std::vec::Vec;

use crate::ecc::curves::bn254::scalar_multiplication::runtime_states::PippengerRuntimeState;
use crate::ecc::fields::field::Field;
use crate::plonk::proof_system::constants::NUM_QUOTIENT_PARTS;

use crate::plonk::composer::composer_base::ComposerType;
use crate::polynomials::evaluation_domain::EvaluationDomain;
use crate::polynomials::Polynomial;
use crate::proof_system::polynomial_store::PolynomialStore;
use crate::srs::reference_string::ProverReferenceString;

use super::types::PolynomialManifest;

const MIN_THREAD_BLOCK: usize = 4;

struct ProvingKeyData<Fr: Field> {
    composer_type: u32,
    circuit_size: u32,
    num_public_inputs: u32,
    contains_recursive_proof: bool,
    recursive_proof_public_input_indices: Vec<u32>,
    memory_read_records: Vec<u32>,
    memory_write_records: Vec<u32>,
    polynomial_store: PolynomialStore<Fr>,
}
pub struct ProvingKey<Fr: Field> {
    pub composer_type: u32,
    pub circuit_size: usize,
    pub log_circuit_size: usize,
    pub num_public_inputs: usize,
    pub contains_recursive_proof: bool,
    pub recursive_proof_public_input_indices: Vec<u32>,
    /// Used by UltraComposer only; for ROM, RAM reads.
    pub memory_read_records: Vec<u32>,
    /// Used by UltraComposer only, for RAM writes.
    pub memory_write_records: Vec<u32>,
    pub polynomial_store: PolynomialStore<Fr>,
    pub small_domain: EvaluationDomain<Fr>,
    pub large_domain: EvaluationDomain<Fr>,
    /// The reference_string object contains the monomial SRS. We can access it using:
    /// Monomial SRS: reference_string->get_monomial_points()
    pub reference_string: Arc<dyn ProverReferenceString>,
    pub quotient_polynomial_parts: [Polynomial<Fr>; NUM_QUOTIENT_PARTS],
    pub pippenger_runtime_state: PippengerRuntimeState,
    pub polynomial_manifest: PolynomialManifest,
}

impl<Fr: Field> ProvingKey<Fr> {
    pub fn new_with_data(data: ProvingKeyData<Fr>, crs: Arc<dyn ProverReferenceString>) -> Self {
        let ProvingKeyData {
            composer_type,
            circuit_size,
            num_public_inputs,
            contains_recursive_proof,
            recursive_proof_public_input_indices,
            memory_read_records,
            memory_write_records,
            polynomial_store,
        } = data;

        let log_circuit_size = (circuit_size as f64).log2().ceil() as usize;
        let small_domain = EvaluationDomain::new(circuit_size, None).unwrap();
        let large_domain = EvaluationDomain::new(1usize << log_circuit_size, None).unwrap();

        let mut ret = Self {
            composer_type,
            circuit_size,
            log_circuit_size,
            num_public_inputs,
            contains_recursive_proof,
            recursive_proof_public_input_indices,
            memory_read_records,
            memory_write_records,
            polynomial_store,
            small_domain,
            large_domain,
            reference_string: crs,
            quotient_polynomial_parts: Default::default(),
            pippenger_runtime_state: PippengerRuntimeState::default(),
            polynomial_manifest: PolynomialManifest::default(),
        };
        ret.init();
        ret
    }

    pub fn new(
        num_gates: usize,
        num_inputs: usize,
        crs: Arc<dyn ProverReferenceString>,
        type_: ComposerType,
    ) -> Self {
        let data = ProvingKeyData {
            composer_type: type_ as u32,
            circuit_size: num_gates + num_inputs,
            num_public_inputs: num_inputs,
            contains_recursive_proof: false,
            recursive_proof_public_input_indices: vec![],
            memory_read_records: vec![],
            memory_write_records: vec![],
            polynomial_store: PolynomialStore::new(),
        };
        let mut ret = Self::new_with_data(data, crs);
        ret.init();
        ret
    }

    /// Initialize the proving key.
    ///
    /// 1. Compute lookup tables for small, mid and large domains.
    /// 2. Set capacity for polynomial store cache.
    /// 3. Initialize quotient_polynomial_parts(n+1) to zeroes.

    pub fn init(&mut self) {
        if self.circuit_size != 0 {
            self.small_domain.compute_lookup_table();
            self.large_domain.compute_lookup_table();
        }

        // t_i for i = 1,2,3 have n+1 coefficients after blinding. t_4 has only n coefficients.
        self.quotient_polynomial_parts[0] = Polynomial::new(self.circuit_size + 1);
        self.quotient_polynomial_parts[1] = Polynomial::new(self.circuit_size + 1);
        self.quotient_polynomial_parts[2] = Polynomial::new(self.circuit_size + 1);
        self.quotient_polynomial_parts[3] = Polynomial::new(self.circuit_size);

        // Initialize quotient_polynomial_parts to zeroes
        let zero_fr = Fr::zero();
        let size_t_fr_len = self.circuit_size + 1;
        let fr_len = self.circuit_size;
        self.quotient_polynomial_parts[0]
            .iter_mut()
            .for_each(|c| *c = zero_fr);
        self.quotient_polynomial_parts[1]
            .iter_mut()
            .for_each(|c| *c = zero_fr);
        self.quotient_polynomial_parts[2]
            .iter_mut()
            .for_each(|c| *c = zero_fr);
        self.quotient_polynomial_parts[3]
            .iter_mut()
            .for_each(|c| *c = zero_fr);
    }

    pub fn from_reader<R: Read>(reader: &mut R, crs_path: &str) -> Result<Self, std::io::Error> {
        let crs = Arc::new(ProverReferenceString::read_from_path(crs_path)?);
    }
}

impl<Fr: Field> Serialize for ProvingKey<Fr> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // TODO
        /*
                // Write the pre-computed polynomials
        template <typename B> inline void write(B& buf, proving_key const& key)
        {
            using serialize::write;
            write(buf, key.composer_type);
            write(buf, (uint32_t)key.circuit_size);
            write(buf, (uint32_t)key.num_public_inputs);

            // Write only the pre-computed polys from the store
            PrecomputedPolyList precomputed_poly_list(key.composer_type);
            size_t num_polys = precomputed_poly_list.size();
            write(buf, static_cast<uint32_t>(num_polys));

            for (size_t i = 0; i < num_polys; ++i) {
                std::string poly_id = precomputed_poly_list[i];
                const barretenberg::polynomial& value = ((proving_key&)key).polynomial_store.get(poly_id);
                write(buf, poly_id);
                write(buf, value);
            }

            write(buf, key.contains_recursive_proof);
            write(buf, key.recursive_proof_public_input_indices);
            write(buf, key.memory_read_records);
            write(buf, key.memory_write_records);
        }
                template <typename B> inline void write_mmap(B& os, std::string const& path, proving_key const& key)
        {
            using serialize::write;

            size_t file_num = 0;
            write(os, key.composer_type);
            write(os, static_cast<uint32_t>(key.circuit_size));
            write(os, static_cast<uint32_t>(key.num_public_inputs));

            // Write only the pre-computed polys from the store
            PrecomputedPolyList precomputed_poly_list(key.composer_type);
            size_t num_polys = precomputed_poly_list.size();
            write(os, static_cast<uint32_t>(num_polys));

            for (size_t i = 0; i < num_polys; ++i) {
                std::string poly_id = precomputed_poly_list[i];
                auto filename = format(path, "/", file_num++, "_", poly_id);
                write(os, poly_id);
                const barretenberg::polynomial& value = ((proving_key&)key).polynomial_store.get(poly_id);
                auto size = value.size();
                std::ofstream ofs(filename);
                ofs.write((char*)&value[0], (std::streamsize)(size * sizeof(barretenberg::fr)));
                if (!ofs.good()) {
                    throw_or_abort(format("Failed to write: ", filename));
                }
            }
            write(os, key.contains_recursive_proof);
            write(os, key.recursive_proof_public_input_indices);
            write(os, key.memory_read_records);
            write(os, key.memory_write_records);
        }
                 */
        todo!("ProvingKey::serialize")
    }
}

impl<'de, Fr: Field> Deserialize<'de> for ProvingKey<Fr> {
    fn deserialize<D>(deserializer: D) -> Result<ProvingKey<Fr>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // TODO

        /*
                // Read the pre-computed polynomials
        template <typename B> inline void read(B& any, proving_key_data& key)
        {
            using serialize::read;
            using std::read;

            read(any, key.composer_type);
            read(any, (uint32_t&)key.circuit_size);
            read(any, (uint32_t&)key.num_public_inputs);

            uint32_t amount = 0;
            read(any, (uint32_t&)amount);

            for (size_t next = 0; next < amount; ++next) {
                std::string label;
                barretenberg::polynomial value;

                read(any, label);
                read(any, value);

                key.polynomial_store.put(label, std::move(value));
            }

            read(any, key.contains_recursive_proof);
            read(any, key.recursive_proof_public_input_indices);
            read(any, key.memory_read_records);
            read(any, key.memory_write_records);
        }

        template <typename B> inline void read_mmap(B& is, std::string const& path, proving_key_data& key)
        {
            using serialize::read;

            size_t file_num = 0;
            read(is, key.composer_type);
            read(is, key.circuit_size);
            read(is, key.num_public_inputs);

            uint32_t size;
            read(is, size);
            for (size_t i = 0; i < size; ++i) {
                std::string name;
                read(is, name);
                barretenberg::polynomial value(format(path, "/", file_num++, "_", name));
                key.polynomial_store.put(name, std::move(value));
            }
            read(is, key.contains_recursive_proof);
            read(is, key.recursive_proof_public_input_indices);
            read(is, key.memory_read_records);
            read(is, key.memory_write_records);
        }

                 */
        todo!("ProvingKey::deserialize")
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_stub() {
        todo!("copy these contents from proving_key.test.cpp");
    }
}
