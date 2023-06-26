use anyhow::{Error, Ok};
use ark_ec::AffineRepr;
use generic_array::{ArrayLength, GenericArray};
use sha3::{Digest, Sha3_256};

use std::collections::HashMap;
use std::fmt::Debug;
use tracing::info;
use typenum::{Unsigned, U16, U32};

/// BarretenHasher is a trait that defines the hash function used for Fiat-Shamir.
pub trait BarretenHasher: std::fmt::Debug {
    /// The size of the security parameter in bytes.
    type SecurityParameterSize: ArrayLength<u8>;
    /// The size of the PRNG output in bytes.
    type PrngOutputSize: ArrayLength<u8> + Debug;

    /// Hashes the given buffer.
    fn hash(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize>;
}

/// Keccak256 hasher.
#[derive(Debug)]
pub(crate) struct Keccak256 {}

impl BarretenHasher for Keccak256 {
    type SecurityParameterSize = U32;
    type PrngOutputSize = U32;

    fn hash(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        Sha3_256::digest(buffer)
    }
}

/// Pedersen with blake3s.
#[derive(Debug)]
pub(crate) struct PedersenBlake3s {}

impl BarretenHasher for PedersenBlake3s {
    type SecurityParameterSize = U16;
    type PrngOutputSize = U32;

    fn hash(_input: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        // TODO from original codebase
        /*
                std::vector<uint8_t> hash_result = blake3::blake3s(buffer);
        std::array<uint8_t, PRNG_OUTPUT_SIZE> result;
        for (size_t i = 0; i < PRNG_OUTPUT_SIZE; ++i) {
            result[i] = hash_result[i];
        }
        return result;
             */
        todo!("check comment to see what gpt told us to do")
    }
}

/// PlookupPedersenBlake3s
#[derive(Debug)]
pub(crate) struct PlookupPedersenBlake3s {}

impl BarretenHasher for PlookupPedersenBlake3s {
    type SecurityParameterSize = U16;
    type PrngOutputSize = U32;
    fn hash(_buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        // TODO from original codebase
        /*
        std::vector<uint8_t> compressed_buffer = crypto::pedersen_commitment::lookup::compress_native(buffer);
        std::array<uint8_t, PRNG_OUTPUT_SIZE> result;
        for (size_t i = 0; i < PRNG_OUTPUT_SIZE; ++i) {
            result[i] = compressed_buffer[i];
        }
        return result;
         */
        todo!("check comment to see what gpt told us to do")
    }
}

/// ManifestEntry describes one piece of data that is used in a particular round of the protocol.
#[derive(Clone, Default, Debug)]
pub(crate) struct ManifestEntry {
    /// The name of the element for fiat-shamir.
    pub(crate) name: String,
    /// The number of bytes in the element.
    pub(crate) num_bytes: usize,
    /// Whether the element is derived by the verifier.
    pub(crate) derived_by_verifier: bool,
    /// The index of the element in the challenge map.
    pub(crate) challenge_map_index: i32,
}

/// The RoundManifest struct describes the data used in one round of the protocol
/// and the challenge(s) created from that data.
#[derive(Clone, Default, Debug)]
pub(crate) struct RoundManifest {
    /// Data used in the round.
    pub(crate) elements: Vec<ManifestEntry>,
    /// The name of the challenge (alpha, beta, etc.).
    pub(crate) challenge: String,
    /// The number of challenges to generate (sometimes we need more than one, e.g in permutation_widget).
    pub(crate) num_challenges: usize,
    /// Whether to put elements in a challenge_map in the transcript.
    pub(crate) map_challenges: bool,
}

impl RoundManifest {
    /// Checks if there is an element in the list with the given name.
    ///
    /// # Arguments
    /// * `element_name` - The name to search for.
    ///
    /// # Returns
    /// * `true` if the name is found in the list, `false` otherwise.
    pub(crate) fn includes_element(&self, element_name: &str) -> bool {
        self.elements.iter().any(|e| e.name == element_name)
    }
}

/// Manifest is used by composers to define the structure of the protocol. It specifies:
/// 1. What data is used in each round of the protocols.
/// 2. Which information is used to create challenges.
#[derive(Clone, Default, Debug)]
pub struct Manifest {
    /// The list of round manifests.
    pub(crate) round_manifests: Vec<RoundManifest>,
    /// The number of rounds in the protocol.
    pub(crate) num_rounds: usize,
}

impl Manifest {
    fn new(round_manifests: Vec<RoundManifest>) -> Self {
        let num_rounds = round_manifests.len();
        Self {
            round_manifests,
            num_rounds,
        }
    }
    fn get_num_rounds(&self) -> usize {
        self.num_rounds
    }
    fn get_round_manifest(&self, round: usize) -> &RoundManifest {
        &self.round_manifests[round]
    }
}

#[derive(Clone, Debug, Default)]
struct Challenge<H: BarretenHasher> {
    data: GenericArray<u8, H::PrngOutputSize>,
}

#[derive(Debug)]
pub struct Transcript<H: BarretenHasher> {
    current_round: usize,
    pub(crate) num_challenge_bytes: usize,
    elements: HashMap<String, Vec<u8>>,
    challenges: HashMap<String, Vec<Challenge<H>>>,
    current_challenge: Challenge<H>,
    manifest: Manifest,
    challenge_map: HashMap<String, i32>,
}

impl<H: BarretenHasher> Default for Transcript<H> {
    fn default() -> Self {
        Self {
            current_round: 0,
            num_challenge_bytes: 0,
            elements: HashMap::new(),
            challenges: HashMap::new(),
            current_challenge: Challenge {
                data: GenericArray::default(),
            },
            manifest: Manifest::default(),
            challenge_map: HashMap::new(),
        }
    }
}

impl<H: BarretenHasher> Transcript<H> {
    pub(crate) fn add_element(&mut self, element_name: &str, buffer: Vec<u8>) {
        info!("Adding element {} to transcript", element_name);
        // from elements.insert({ element_name, buffer });
        self.elements.insert(element_name.to_string(), buffer);
    }

    /// Create a new transcript based on the manifest
    /// # Arguments
    /// input_manifest:  The manifest with round descriptions.
    /// hash_type: The hash used for Fiat-Shamir.
    /// challenge_bytes: The number of bytes per challenge to generate.
    pub(crate) fn new(input_manifest: Option<Manifest>, num_challenge_bytes: usize) -> Self {
        let mut ret = Transcript::<H> {
            num_challenge_bytes,
            manifest: input_manifest.unwrap_or_default(),
            ..Default::default()
        };
        ret.compute_challenge_map();
        ret
    }
    /// Constructs a new `Transcript` from a serialized transcript, a `Manifest`, a `HashType` and a challenge byte size.
    ///
    /// # Arguments
    ///
    /// * `input_transcript` - The serialized transcript.
    /// * `input_manifest` - The manifest which governs the parsing.
    /// * `hash_type` - The hash used for Fiat-Shamir.
    /// * `challenge_bytes` - The number of bytes per challenge to generate.
    ///
    /// # Panics
    ///
    /// If the serialized transcript does not contain the required number of bytes, a panic occurs.
    ///
    pub(crate) fn new_from_transcript(
        input_transcript: &[u8],
        input_manifest: Manifest,
        num_challenge_bytes: usize,
    ) -> Self {
        let num_rounds = input_manifest.get_num_rounds();
        let mut count = 0;
        // Compute how much data we need according to the manifest
        let mut total_required_size = 0;
        for i in 0..num_rounds {
            for manifest_element in &input_manifest.get_round_manifest(i).elements {
                if !manifest_element.derived_by_verifier {
                    total_required_size += manifest_element.num_bytes;
                }
            }
        }
        // Check that the total required size is equal to the size of the input_transcript
        if total_required_size != input_transcript.len() {
            panic!("Serialized transcript does not contain the required number of bytes");
        }

        let mut elements = std::collections::HashMap::new();
        for i in 0..num_rounds {
            for manifest_element in &input_manifest.get_round_manifest(i).elements {
                if !manifest_element.derived_by_verifier {
                    let end = count + manifest_element.num_bytes;
                    let element_data = input_transcript[count..end].to_vec();
                    elements.insert(manifest_element.name.clone(), element_data);
                    count += manifest_element.num_bytes;
                }
            }
        }

        let mut transcript = Self {
            num_challenge_bytes,
            manifest: input_manifest,
            elements,
            challenges: std::collections::HashMap::new(),
            current_round: 0,
            current_challenge: Challenge {
                data: GenericArray::default(),
            },
            challenge_map: std::collections::HashMap::new(),
        };
        transcript.compute_challenge_map();
        transcript
    }

    fn from_serialized(
        _input_transcript: Vec<u8>,
        _manifest: Manifest,
        _challenge_bytes: usize,
    ) -> Self {
        todo!()
    }

    pub(crate) fn get_manifest(&self) -> Manifest {
        self.manifest.clone()
    }

    /// Apply the Fiat-Shamir transform to create challenges for the current round.
    /// The challenges are saved to transcript. Round number is increased.
    ///
    /// # Arguments
    ///
    /// * `challenge_name` - Challenge name (needed to check if the challenge fits the current round).
    ///
    pub(crate) fn apply_fiat_shamir(&mut self, _challenge_name: &str) {
        // implementation

        // TODO
        /*
            // For reference, see the relevant manifest, which is defined in
        // plonk/composer/[standard/turbo/ultra]_composer.hpp
        ASSERT(current_round <= manifest.get_num_rounds());
        // TODO(Cody): Coupling: this line insists that the challenges in the manifest
        // are encountered in the order that matches the order of the proof construction functions.
        // Future architecture should specify this data in a single place (?).
        info_togglable("apply_fiat_shamir(): challenge name match:");
        info_togglable("\t challenge_name in: ", challenge_name);
        info_togglable("\t challenge_name expected: ", manifest.get_round_manifest(current_round).challenge, "\n");
        ASSERT(challenge_name == manifest.get_round_manifest(current_round).challenge);

        const size_t num_challenges = manifest.get_round_manifest(current_round).num_challenges;
        if (num_challenges == 0) {
            ++current_round;
            return;
        }

        // Combine the very last challenge from the previous fiat-shamir round (which is, inductively, a hash containing the
        // manifest data of all previous rounds), plus the manifest data for this round, into a buffer. This buffer will
        // ultimately be hashed, to form this round's fiat-shamir challenge(s).
        std::vector<uint8_t> buffer;
        if (current_round > 0) {
            buffer.insert(buffer.end(), current_challenge.data.begin(), current_challenge.data.end());
        }
        for (auto manifest_element : manifest.get_round_manifest(current_round).elements) {
            info_togglable("apply_fiat_shamir(): manifest element name match:");
            info_togglable("\t element name: ", manifest_element.name);
            info_togglable(
                "\t element exists and is unique: ", (elements.count(manifest_element.name) == 1) ? "true" : "false", "\n");
            ASSERT(elements.count(manifest_element.name) == 1);

            std::vector<uint8_t>& element_data = elements.at(manifest_element.name);
            if (!manifest_element.derived_by_verifier) {
                ASSERT(manifest_element.num_bytes == element_data.size());
            }
            buffer.insert(buffer.end(), element_data.begin(), element_data.end());
        }

        std::vector<challenge> round_challenges;
        std::array<uint8_t, PRNG_OUTPUT_SIZE> base_hash{};

        switch (hasher) {
        case HashType::Keccak256: {
            base_hash = Keccak256Hasher::hash(buffer);
            break;
        }
        case HashType::PedersenBlake3s: {
            std::vector<uint8_t> compressed_buffer = to_buffer(crypto::pedersen_commitment::compress_native(buffer));
            base_hash = Blake3sHasher::hash(compressed_buffer);
            break;
        }
        case HashType::PlookupPedersenBlake3s: {
            std::vector<uint8_t> compressed_buffer = crypto::pedersen_commitment::lookup::compress_native(buffer);
            base_hash = Blake3sHasher::hash_plookup(compressed_buffer);
            break;
        }
        default: {
            throw_or_abort("no hasher was selected for the transcript");
        }
        }

        // Depending on the settings, we might be able to chunk the bytes of a single hash across multiple challenges:
        const size_t challenges_per_hash = PRNG_OUTPUT_SIZE / num_challenge_bytes;

        for (size_t j = 0; j < challenges_per_hash; ++j) {
            if (j < num_challenges) {
                // Each challenge still occupies PRNG_OUTPUT_SIZE number of bytes, but only num_challenge_bytes rhs bytes
                // are nonzero.
                std::array<uint8_t, PRNG_OUTPUT_SIZE> challenge{};
                std::copy(base_hash.begin() + (j * num_challenge_bytes),
                          base_hash.begin() + (j + 1) * num_challenge_bytes,
                          challenge.begin() +
                              (PRNG_OUTPUT_SIZE -
                               num_challenge_bytes)); // Left-pad the challenge with zeros, and then copy the next
                                                      // num_challange_bytes slice of the hash to the rhs of the challenge.
                round_challenges.push_back({ challenge });
            }
        }

        std::vector<uint8_t> rolling_buffer(base_hash.begin(), base_hash.end());
        rolling_buffer.push_back(0);

        // Compute how many hashes we need so that we have enough distinct chunks of 'random' bytes to distribute
        // across the num_challenges.
        size_t num_hashes = (num_challenges / challenges_per_hash);
        if (num_hashes * challenges_per_hash != num_challenges) {
            ++num_hashes;
        }

        for (size_t i = 1; i < num_hashes; ++i) {
            // Compute hash_output = hash(base_hash, i);
            rolling_buffer[rolling_buffer.size() - 1] = static_cast<uint8_t>(i);
            std::array<uint8_t, PRNG_OUTPUT_SIZE> hash_output{};
            switch (hasher) {
            case HashType::Keccak256: {
                hash_output = Keccak256Hasher::hash(rolling_buffer);
                break;
            }
            case HashType::PedersenBlake3s: {
                hash_output = Blake3sHasher::hash(rolling_buffer);
                break;
            }
            case HashType::PlookupPedersenBlake3s: {
                hash_output = Blake3sHasher::hash_plookup(rolling_buffer);
                break;
            }
            default: {
                throw_or_abort("no hasher was selected for the transcript");
            }
            }
            for (size_t j = 0; j < challenges_per_hash; ++j) {
                // Only produce as many challenges as we need.
                if (challenges_per_hash * i + j < num_challenges) {
                    std::array<uint8_t, PRNG_OUTPUT_SIZE> challenge{};
                    std::copy(hash_output.begin() + (j * num_challenge_bytes),
                              hash_output.begin() + (j + 1) * num_challenge_bytes,
                              challenge.begin() + (PRNG_OUTPUT_SIZE - num_challenge_bytes));
                    round_challenges.push_back({ challenge });
                }
            }
        }

        // Remember the very last challenge, as it will be included in the buffer of the next fiat-shamir round (since this
        // challenge is effectively a hash of _all_ previous rounds' manifest data).
        current_challenge = round_challenges[round_challenges.size() - 1];

        challenges.insert({ challenge_name, round_challenges });
        ++current_round;
             */
        todo!("see comment...")
    }

    /// Get the challenge with the given name at index.
    /// Will fail if there is no challenge with such name
    /// or there are not enough subchallenges in the vector.
    ///
    /// # Arguments
    ///
    /// * `challenge_name` - The name of the challenge.
    /// * `idx` - The idx of subchallenge.
    ///
    /// # Returns
    ///
    /// The challenge value.
    pub(crate) fn get_challenge(
        &self,
        challenge_name: &str,
        idx: Option<usize>,
    ) -> Result<&GenericArray<u8, H::PrngOutputSize>, Error> {
        let idx = idx.unwrap_or(0);
        info!("get_challenge(): {}", challenge_name);
        assert!(self.challenges.contains_key(challenge_name));
        Ok(&self.challenges.get(challenge_name).unwrap()[idx].data)
    }

    /// Get the challenge index from map (needed when we name subchallenges).
    ///
    /// # Arguments
    ///
    /// * `challenge_map_name` - The name of the subchallenge.
    ///
    /// # Returns
    ///
    /// The index of the subchallenge in the vector corresponding to the challenge.
    pub(crate) fn get_challenge_index_from_map(
        &self,
        challenge_map_name: &str,
    ) -> Result<isize, Error> {
        Ok(self.challenge_map[challenge_map_name].try_into()?)
    }

    /// Check if a challenge exists.
    ///
    /// # Arguments
    ///
    /// * `challenge_name` - The name of the challenge.
    ///
    /// # Returns
    ///
    /// true if exists, false if not.
    pub(crate) fn has_challenge(&self, challenge_name: &str) -> bool {
        self.challenges.contains_key(challenge_name)
    }

    /// Get a particular subchallenge value by the name of the subchallenge.
    /// For example, we use it with (nu, r).
    ///
    /// # Arguments
    ///
    /// * `challenge_name` - The name of the challenge.
    /// * `challenge_map_name` - The name of the subchallenge.
    ///
    /// # Returns
    ///
    /// The value of the subchallenge.
    pub(crate) fn get_challenge_from_map(
        &self,
        challenge_name: &str,
        challenge_map_name: &str,
    ) -> GenericArray<u8, H::PrngOutputSize> {
        let key = self.challenge_map[challenge_map_name];
        if key == -1 {
            let mut result = GenericArray::default();
            result[<H::PrngOutputSize as Unsigned>::USIZE - 1] = 1;
            return result;
        }
        let value = &self.challenges[challenge_name][key as usize];
        value.data.clone()
    }

    /// Get the number of challenges in the transcript.
    /// fails if no challenges with such name.
    /// we use it with beta/gamma which need to be created in one fiat-shamir transform
    ///
    /// # Arguments
    /// * `challenge_name` - The name of the challenge.
    ///
    /// # Returns
    ///
    /// The number of challenges.
    pub(crate) fn get_num_challenges(&self, challenge_name: &str) -> usize {
        assert!(self.challenges.contains_key(challenge_name));
        self.challenges.get(challenge_name).unwrap().len()
    }

    /// gets the value of an element in the transcript.
    /// fails if no element with such name.
    ///
    /// # Arguments
    /// * `element_name` - The name of the element.
    ///
    /// # Returns
    ///
    /// The value of the element.
    pub(crate) fn get_element(&self, element_name: &str) -> Vec<u8> {
        assert!(self.elements.contains_key(element_name));
        self.elements.get(element_name).unwrap().clone()
    }

    /// gets the size of an element in the transcript.
    ///
    /// # Arguments
    /// * `element_name` - The name of the element.
    ///
    /// # Returns
    ///
    /// The size of the element. otherwise -1
    pub(crate) fn get_element_size(&self, element_name: &str) -> usize {
        for manifest in &self.manifest.round_manifests {
            for element in &manifest.elements {
                if element.name == element_name {
                    return element.num_bytes;
                }
            }
        }
        usize::MAX
    }

    /// serialize the transcript to a byte vector.
    ///
    /// # Returns
    ///
    /// The serialized transcript.
    pub(crate) fn export_transcript(&self) -> Vec<u8> {
        let buf: Vec<u8> = vec![];
        for manifest in &self.manifest.round_manifests {
            for _element in &manifest.elements {
                /*
                    ASSERT(elements.count(manifest_element.name) == 1);
                const std::vector<uint8_t>& element_data = elements.at(manifest_element.name);
                if (!manifest_element.derived_by_verifier) {
                    ASSERT(manifest_element.num_bytes == element_data.size());
                }
                if (!manifest_element.derived_by_verifier) {
                    // printf("writing element %s ", manifest_element.name.c_str());
                    // for (size_t j = 0; j < element_data.size(); ++j) {
                    //     printf("%x", element_data[j]);
                    // }
                    // printf("\n");
                    buffer.insert(buffer.end(), element_data.begin(), element_data.end());
                }
                     */
                todo!("check comment");
            }
        }
        buf
    }

    /// Insert element names from all rounds of the manifest
    /// into the challenge_map.
    pub(crate) fn compute_challenge_map(&mut self) {
        self.challenge_map.clear();
        for manifest in &self.manifest.round_manifests {
            if manifest.map_challenges {
                for element in &manifest.elements {
                    self.challenge_map
                        .insert(element.name.clone(), element.challenge_map_index);
                }
            }
        }
    }

    /// Mock prover transcript interactions up to fiat-shamir of a given challenge.
    ///
    /// This is useful for testing individual parts of the prover since all
    /// transcript interactions must occur sequentially according to the manifest.
    /// Function allows for optional input of circuit_size since this is needed in some
    /// test cases, e.g. instantiating a Sumcheck from a mocked transcript.
    ///
    /// # Arguments
    ///
    /// * `challenge_in` - The challenge name up to which to mock the transcript interactions.
    pub(crate) fn mock_inputs_prior_to_challenge(
        &mut self,
        challenge_in: &str,
        circuit_size: usize,
    ) {
        // Perform operations only up to fiat-shamir of challenge_in
        // TODO this clone isn't great but it satisfies the borrow checker
        for manifest in &self.manifest.round_manifests.clone() {
            // loop over RoundManifests
            for entry in &manifest.elements {
                // loop over ManifestEntrys
                if entry.name == "circuit_size" {
                    self.add_element(
                        "circuit_size",
                        vec![
                            (circuit_size >> 24) as u8,
                            (circuit_size >> 16) as u8,
                            (circuit_size >> 8) as u8,
                            circuit_size as u8,
                        ],
                    );
                } else {
                    let buffer = vec![1; entry.num_bytes]; // arbitrary buffer of 1's
                    self.add_element(&entry.name, buffer);
                }
            }
            if challenge_in == manifest.challenge {
                break;
            } else {
                self.apply_fiat_shamir(&manifest.challenge);
            }
        }
    }

    pub(crate) fn add_field_element<Fr: ark_ff::Field>(
        &mut self,
        element_name: &str,
        element: &Fr,
    ) {
        let mut buf = vec![0u8; Fr::serialized_size(element, ark_serialize::Compress::No)];
        Fr::serialize_uncompressed(element, &mut buf).unwrap();
        self.add_element(element_name, buf);
    }

    pub(crate) fn add_group_element<G: AffineRepr>(&mut self, element_name: &str, element: &G) {
        let mut buf = vec![0u8; G::serialized_size(element, ark_serialize::Compress::No)];
        G::serialize_uncompressed(element, &mut buf).unwrap();
        self.add_element(element_name, buf);
    }
    pub(crate) fn get_field_element<Fr: ark_ff::Field>(&self, element_name: &str) -> Fr {
        let buf = self.get_element(element_name);
        Fr::deserialize_uncompressed(buf.as_slice()).unwrap()
    }
    pub(crate) fn get_group_element<G: AffineRepr>(&self, element_name: &str) -> G {
        let buf = self.get_element(element_name);
        G::deserialize_uncompressed(buf.as_slice()).unwrap()
    }
    pub(crate) fn get_field_element_vector<Fr: ark_ff::Field>(
        &self,
        element_name: &str,
    ) -> Vec<Fr> {
        // TODO is this right
        let serialized_size = Fr::serialized_size(&Fr::ONE, ark_serialize::Compress::No);
        let buf = self.get_element(element_name);
        let mut res = Vec::new();
        for i in 0..buf.len() / serialized_size {
            res.push(
                Fr::deserialize_uncompressed(&buf[i * serialized_size..(i + 1) * serialized_size])
                    .unwrap(),
            );
        }
        res
    }
    pub(crate) fn put_field_element_vector<Fr: ark_ff::Field>(
        &mut self,
        element_name: &str,
        elements: &[Fr],
    ) {
        let mut buf = Vec::new();
        for element in elements {
            let mut tmp = vec![0u8; Fr::serialized_size(element, ark_serialize::Compress::No)];
            Fr::serialize_uncompressed(element, &mut tmp).unwrap();
            buf.extend(tmp);
        }
        self.add_element(element_name, buf);
    }

    pub(crate) fn get_challenge_field_element<Fr: ark_ff::Field>(
        &self,
        challenge_name: &str,
        idx: Option<usize>,
    ) -> Fr {
        let buf = self.get_challenge(challenge_name, idx);
        Fr::deserialize_uncompressed(buf.unwrap().as_slice()).unwrap()
    }
    fn get_challenge_field_element_from_map<Fr: ark_ff::Field>(
        &self,
        challenge_name: &str,
        challenge_map_name: &str,
    ) -> Fr {
        let buf = self.get_challenge_from_map(challenge_name, challenge_map_name);
        Fr::deserialize_uncompressed(buf.as_slice()).unwrap()
    }
}
