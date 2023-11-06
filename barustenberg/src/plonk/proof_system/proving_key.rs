use ark_ff::{FftField, Field};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::Read;
use std::sync::{Arc, RwLock};
use std::vec::Vec;

use crate::ecc::curves::bn254_scalar_multiplication::PippengerRuntimeState;
use crate::plonk::proof_system::constants::NUM_QUOTIENT_PARTS;

use crate::plonk::composer::composer_base::ComposerType;
use crate::polynomials::evaluation_domain::EvaluationDomain;
use crate::polynomials::Polynomial;
use crate::proof_system::polynomial_store::PolynomialStore;
use crate::srs::reference_string::file_reference_string::FileReferenceString;
use crate::srs::reference_string::ProverReferenceString;

use super::types::PolynomialManifest;

const MIN_THREAD_BLOCK: usize = 4;

pub(crate) struct ProvingKeyData<F: Field + FftField> {
    composer_type: ComposerType,
    circuit_size: u32,
    num_public_inputs: u32,
    contains_recursive_proof: bool,
    recursive_proof_public_input_indices: Vec<u32>,
    memory_read_records: Vec<usize>,
    memory_write_records: Vec<usize>,
    polynomial_store: PolynomialStore<F>,
}

#[derive(Debug)]
pub struct ProvingKey<Fr: Field + FftField> {
    pub(crate) composer_type: ComposerType,
    pub(crate) circuit_size: usize,
    pub(crate) log_circuit_size: usize,
    pub(crate) num_public_inputs: usize,
    pub(crate) contains_recursive_proof: bool,
    pub(crate) recursive_proof_public_input_indices: Vec<u32>,
    /// Used by UltraComposer only; for ROM, RAM reads.
    pub(crate) memory_read_records: Vec<usize>,
    /// Used by UltraComposer only, for RAM writes.
    pub(crate) memory_write_records: Vec<usize>,
    pub(crate) polynomial_store: PolynomialStore<Fr>,
    pub(crate) small_domain: Box<EvaluationDomain<Fr>>,
    pub(crate) large_domain: Box<EvaluationDomain<Fr>>,
    /// The reference_string object contains the monomial SRS. We can access it using:
    /// Monomial SRS: reference_string->get_monomial_points()
    pub(crate) reference_string: Arc<RwLock<dyn ProverReferenceString>>,
    pub(crate) quotient_polynomial_parts:
        [Arc<RwLock<Polynomial<Fr>>>; NUM_QUOTIENT_PARTS as usize],
    pub(crate) pippenger_runtime_state: PippengerRuntimeState<ark_bn254::g1::Config>,
    pub(crate) polynomial_manifest: PolynomialManifest,
}

impl<Fr: Field + FftField> Default for ProvingKey<Fr> {
    fn default() -> Self {
        Self {
            polynomial_store: PolynomialStore::new(),
            small_domain: Box::new(EvaluationDomain::new(0, None)),
            large_domain: Box::new(EvaluationDomain::new(0, None)),
            composer_type: Default::default(),
            circuit_size: 0,
            log_circuit_size: 0,
            num_public_inputs: 0,
            contains_recursive_proof: false,
            recursive_proof_public_input_indices: vec![],
            memory_read_records: vec![],
            memory_write_records: vec![],
            reference_string: Arc::new(RwLock::new(FileReferenceString::default())),
            quotient_polynomial_parts: Default::default(),
            pippenger_runtime_state: Default::default(),
            polynomial_manifest: Default::default(),
        }
    }
}

impl<Fr: Field + FftField> ProvingKey<Fr> {
    pub(crate) fn new_with_data(
        data: ProvingKeyData<Fr>,
        crs: Arc<RwLock<dyn ProverReferenceString>>,
    ) -> Self {
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
        let small_domain = Box::new(EvaluationDomain::new(circuit_size as usize, None));
        let large_domain = Box::new(EvaluationDomain::new(1usize << log_circuit_size, None));

        let mut ret = Self {
            composer_type,
            circuit_size: circuit_size as usize,
            log_circuit_size,
            num_public_inputs: num_public_inputs as usize,
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

    pub(crate) fn new(
        num_gates: usize,
        num_inputs: usize,
        crs: Arc<RwLock<dyn ProverReferenceString>>,
        type_: ComposerType,
    ) -> Self {
        let data = ProvingKeyData {
            composer_type: type_,
            circuit_size: (num_gates + num_inputs) as u32,
            num_public_inputs: num_inputs as u32,
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

    pub(crate) fn init(&mut self) {
        if self.circuit_size != 0 {
            self.small_domain.compute_lookup_table();
            self.large_domain.compute_lookup_table();
        }

        // t_i for i = 1,2,3 have n+1 coefficients after blinding. t_4 has only n coefficients.
        // TODO unclear if this is necessary
        self.quotient_polynomial_parts[0] =
            Arc::new(RwLock::new(Polynomial::new(self.circuit_size + 1)));
        self.quotient_polynomial_parts[1] =
            Arc::new(RwLock::new(Polynomial::new(self.circuit_size + 1)));
        self.quotient_polynomial_parts[2] =
            Arc::new(RwLock::new(Polynomial::new(self.circuit_size + 1)));
        self.quotient_polynomial_parts[3] =
            Arc::new(RwLock::new(Polynomial::new(self.circuit_size)));
    }

    pub(crate) fn from_reader<R: Read>(
        _reader: &mut R,
        crs_path: &str,
    ) -> Result<Self, std::io::Error> {
        let _crs = Arc::new(FileReferenceString::read_from_path(crs_path)?);
        todo!();
    }
}

impl<Fr: Field + FftField> Serialize for ProvingKey<Fr> {
    fn serialize<S: Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
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

impl<'de, Fr: Field + FftField> Deserialize<'de> for ProvingKey<Fr> {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
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
    fn test_proving_key_from_serialized_key() {
        /*
                TEST(proving_key, proving_key_from_serialized_key)
        {
            auto builder = StandardCircuitBuilder();
            auto composer = StandardComposer();
            fr a = fr::one();
            builder.add_public_variable(a);

            plonk::proving_key& p_key = *composer.compute_proving_key(builder);
            auto pk_buf = to_buffer(p_key);
            auto pk_data = from_buffer<plonk::proving_key_data>(pk_buf);
            auto crs = std::make_unique<barretenberg::srs::factories::FileCrsFactory<curve::BN254>>("../srs_db/ignition");
            auto proving_key =
                std::make_shared<plonk::proving_key>(std::move(pk_data), crs->get_prover_crs(pk_data.circuit_size + 1));

            // Loop over all pre-computed polys for the given composer type and ensure equality
            // between original proving key polynomial store and the polynomial store that was
            // serialized/deserialized from buffer
            plonk::PrecomputedPolyList precomputed_poly_list(p_key.circuit_type);
            bool all_polys_are_equal{ true };
            for (size_t i = 0; i < precomputed_poly_list.size(); ++i) {
                std::string poly_id = precomputed_poly_list[i];
                auto input_poly = p_key.polynomial_store.get(poly_id);
                auto output_poly = proving_key->polynomial_store.get(poly_id);
                all_polys_are_equal = all_polys_are_equal && (input_poly == output_poly);
            }

            // Check that all pre-computed polynomials are equal
            EXPECT_EQ(all_polys_are_equal, true);

            // Check equality of other proving_key_data data
            EXPECT_EQ(p_key.circuit_type, proving_key->circuit_type);
            EXPECT_EQ(p_key.circuit_size, proving_key->circuit_size);
            EXPECT_EQ(p_key.num_public_inputs, proving_key->num_public_inputs);
            EXPECT_EQ(p_key.contains_recursive_proof, proving_key->contains_recursive_proof);
        }
                 */
        todo!("copy these contents from proving_key.test.cpp");
    }
}
