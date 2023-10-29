use anyhow::{Error, Ok};
use ark_ec::AffineRepr;
use ark_ff::{BigInteger, Field, PrimeField};
use ark_serialize::CanonicalDeserialize;
use generic_array::{ArrayLength, GenericArray};
use grumpkin::Fq;
use sha3::{Digest, Sha3_256};

use std::collections::HashMap;
use std::fmt::Debug;
use tracing::info;
use typenum::{Unsigned, U16, U32};

use crate::crypto::{generator::GENERATOR_CONTEXT, pedersen::pederson_hash::hash};

/// BarretenHasher is a trait that defines the hash function used for Fiat-Shamir.
pub trait BarretenHasher: std::fmt::Debug + Send + Sync + Clone + Default {
    /// The size of the security parameter in bytes.
    type SecurityParameterSize: ArrayLength<u8>;
    /// The size of the PRNG output in bytes.
    type PrngOutputSize: ArrayLength<u8> + Debug;

    /// Hashes the given buffer.
    fn hash(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize>;

    /// Hashes the given buffer for a transcript in the fiat-shamir. uses the same hash function as hash() but may do a pederson compression first
    fn hash_for_transcript(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize>;
}

/// Keccak256 hasher.
#[derive(Debug, Default, Clone)]
pub(crate) struct Keccak256 {}

impl BarretenHasher for Keccak256 {
    type SecurityParameterSize = U32;
    type PrngOutputSize = U32;

    fn hash(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        Sha3_256::digest(buffer)
    }

    fn hash_for_transcript(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        Self::hash(buffer)
    }
}

/// Pedersen with blake3s.
#[derive(Debug, Clone, Default)]
pub(crate) struct PedersenBlake3s {}

impl BarretenHasher for PedersenBlake3s {
    type SecurityParameterSize = U16;
    type PrngOutputSize = U32;

    fn hash(input: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        println!("input: {:?}", input);
        //Some cases I received a deserialization failure: Failing case
        // [1, 172, 212, 56, 254, 187, 127, 207, 167, 252, 131, 52, 217, 105, 127, 94, 196, 243, 16, 56, 115, 235, 25, 214, 132, 144, 84, 164, 51, 94, 91, 117, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        let input = Fq::from_random_bytes(input).unwrap_or(
            // if we can't deserialize then create representative element by taking its length + last bit
            Fq::from_random_bytes(&(input.len() + *input.last().unwrap() as usize).to_be_bytes())
                .unwrap(),
        );
        println!("input Fq: {:?}", input);

        //Note: ended up fighting the compiler a lot to grab resulting bytes. Open to suggestions to make this cleaner
        let mut res = GenericArray::default();
        //Hashes and returns compressed form of grumpkin point (x coordinate)
        res.copy_from_slice(
            &hash(&[input], &mut GENERATOR_CONTEXT.lock().unwrap())
                .into_bigint()
                .to_bytes_be(),
        );
        res
    }

    fn hash_for_transcript(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        Self::hash(buffer)
    }
}

/// PlookupPedersenBlake3s
#[derive(Debug, Clone, Default)]
pub(crate) struct PlookupPedersenBlake3s {}

impl BarretenHasher for PlookupPedersenBlake3s {
    type SecurityParameterSize = U16;
    type PrngOutputSize = U32;
    fn hash(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        let input = Fq::deserialize_uncompressed_unchecked(buffer).unwrap();

        //Note: ended up fighting the compiler a lot to grab resulting bytes. Open to suggestions to make this cleaner
        let mut res = GenericArray::default();
        //Hashes and returns compressed form of grumpkin point (x coordinate)
        res.copy_from_slice(
            &hash(&[input], &mut GENERATOR_CONTEXT.lock().unwrap())
                .into_bigint()
                .to_bytes_be(),
        );
        res
    }

    fn hash_for_transcript(buffer: &[u8]) -> GenericArray<u8, Self::PrngOutputSize> {
        Self::hash(buffer)
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
    pub(crate) fn new(round_manifests: Vec<RoundManifest>) -> Self {
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

    pub(crate) fn add_round_manifest(&mut self, round_manifest: RoundManifest) {
        self.round_manifests.push(round_manifest);
        self.num_rounds += 1;
    }
}

#[derive(Clone, Debug, Default)]
struct Challenge<H: BarretenHasher> {
    data: GenericArray<u8, H::PrngOutputSize>,
}

#[derive(Debug)]
pub(crate) struct Transcript<H: BarretenHasher> {
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
            //Prevent divide by zero in fiat-shamir
            num_challenge_bytes: 1,
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
        // Check number of chellenge bytes to prevent divide by 0
        if num_challenge_bytes == 0 {
            panic!("The required number of challenge bytes > 0");
        }

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

        // Check number of challenge bytes to prevent divide by 0
        if num_challenge_bytes == 0 {
            panic!("The required number of challenge bytes > 0");
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

    pub(crate) fn from_serialized(
        input_transcript: Vec<u8>,
        input_manifest: Manifest,
        num_challenge_bytes: usize,
    ) -> Self {
        let num_rounds = input_manifest.get_num_rounds();
        let mut count = 0;
        let mut total_required_size = 0;
        for i in 0..num_rounds {
            for manifest_element in &input_manifest.get_round_manifest(i).elements {
                if !manifest_element.derived_by_verifier {
                    total_required_size += manifest_element.num_bytes;
                }
            }
        }
        if total_required_size != input_transcript.len() {
            panic!("Serialized transcript does not contain the required number of bytes");
        }

        // Check number of chellenge bytes to prevent divide by 0
        if num_challenge_bytes == 0 {
            panic!("The required number of challenge bytes > 0");
        }

        let mut elements = HashMap::new();
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
            challenges: HashMap::new(),
            current_round: 0,
            current_challenge: Challenge {
                data: GenericArray::default(),
            },
            challenge_map: HashMap::new(),
        };
        transcript.compute_challenge_map();
        transcript
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
    pub(crate) fn apply_fiat_shamir(&mut self, challenge_name: &str) {
        assert!(self.current_round <= self.manifest.get_num_rounds());
        info!("apply_fiat_shamir(): challenge name match:");
        info!("\t challenge_name in: {}", challenge_name);
        info!(
            "\t challenge_name expected: {}",
            self.manifest
                .get_round_manifest(self.current_round)
                .challenge
        );

        assert_eq!(
            challenge_name,
            self.manifest
                .get_round_manifest(self.current_round)
                .challenge
        );

        let num_challenges = self
            .manifest
            .get_round_manifest(self.current_round)
            .num_challenges;
        if num_challenges == 0 {
            self.current_round += 1;
            return;
        }

        let mut buffer = Vec::new();
        if self.current_round > 0 {
            buffer.extend_from_slice(&self.current_challenge.data);
        }
        for manifest_element in &self
            .manifest
            .get_round_manifest(self.current_round)
            .elements
        {
            info!("apply_fiat_shamir(): manifest element name match:");
            info!("\t element name: {}", manifest_element.name);
            info!(
                "\t element exists and is unique: {}",
                self.elements.contains_key(&manifest_element.name)
            );
            assert!(self.elements.contains_key(&manifest_element.name));

            let element_data = &self.elements[&manifest_element.name];
            if !manifest_element.derived_by_verifier {
                assert_eq!(manifest_element.num_bytes, element_data.len());
            }
            buffer.extend_from_slice(element_data);
        }

        let mut round_challenges: Vec<Challenge<H>> = Vec::new();
        println!("Manifest: {:?}", self.manifest.round_manifests);
        println!();
        println!("Buffer: {:?}", buffer);
        println!();
        let base_hash: GenericArray<u8, H::PrngOutputSize> = H::hash_for_transcript(&buffer);
        // Depending on the settings, we might be able to chunk the bytes of a single hash across multiple challenges:
        let challenges_per_hash = H::PrngOutputSize::to_usize() / self.num_challenge_bytes;

        for j in 0..challenges_per_hash {
            if j < num_challenges {
                let mut challenge = vec![0u8; H::PrngOutputSize::to_usize()];
                let start = j * self.num_challenge_bytes;
                let end = (j + 1) * self.num_challenge_bytes;
                challenge[H::PrngOutputSize::to_usize() - self.num_challenge_bytes..]
                    .copy_from_slice(&base_hash[start..end]);
                round_challenges.push(Challenge {
                    data: GenericArray::clone_from_slice(&challenge),
                });
            }
        }

        let mut rolling_buffer = base_hash.to_vec();
        rolling_buffer.push(0);

        println!("Rolling Buffer: {:?}", rolling_buffer);
        println!();
        let num_hashes = (num_challenges / challenges_per_hash)
            + if num_challenges % challenges_per_hash != 0 {
                1
            } else {
                0
            };

        for i in 1..num_hashes {
            let roll_buf_len = rolling_buffer.len();
            rolling_buffer[roll_buf_len - 1] = i as u8;
            let hash_output = H::hash(&rolling_buffer);

            for j in 0..challenges_per_hash {
                if challenges_per_hash * i + j < num_challenges {
                    let mut challenge = vec![0u8; H::PrngOutputSize::to_usize()];
                    let start = j * self.num_challenge_bytes;
                    let end = (j + 1) * self.num_challenge_bytes;
                    challenge[H::PrngOutputSize::to_usize() - self.num_challenge_bytes..]
                        .copy_from_slice(&hash_output[start..end]);
                    round_challenges.push(Challenge {
                        data: GenericArray::clone_from_slice(&challenge),
                    });
                }
            }
        }
        // Remember the very last challenge, as it will be included in the buffer of the next fiat-shamir round (since this
        // challenge is effectively a hash of _all_ previous rounds' manifest data).
        self.current_challenge = round_challenges[round_challenges.len() - 1].clone();
        println!("Current Challenge: {:?}", self.current_challenge);
        println!();

        self.challenges
            .insert(challenge_name.to_string(), round_challenges);
        self.current_round += 1;
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
        let mut buf: Vec<u8> = vec![];
        for manifest in &self.manifest.round_manifests {
            for element in &manifest.elements {
                assert!(self.elements.contains_key(&element.name));
                let element_data = self.elements.get(&element.name).unwrap();
                //NOTE: this derived check was split into two separate if's in original implementation
                if !element.derived_by_verifier {
                    assert!(element.num_bytes == element_data.len());
                    buf.extend_from_slice(element_data);
                }
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
    pub(crate) fn get_challenge_field_element_from_map<Fr: ark_ff::Field>(
        &self,
        challenge_name: &str,
        challenge_map_name: &str,
    ) -> Fr {
        let buf = self.get_challenge_from_map(challenge_name, challenge_map_name);
        Fr::deserialize_uncompressed(buf.as_slice()).unwrap()
    }
}

#[cfg(test)]

pub(crate) mod test {
    use rand::random;

    use super::*;

    //NOTE: default for challenge_map_index = 0

    fn create_manifest(num_public_inputs: usize) -> Manifest {
        let g1_size = 64usize;
        let fr_size = 32usize;

        let public_input_size = fr_size * num_public_inputs;
        let round_manifests = vec![
            RoundManifest {
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
            },
            RoundManifest {
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
            },
            RoundManifest {
                elements: vec![ManifestEntry {
                    name: "Z_PERM".to_string(),
                    num_bytes: g1_size,
                    derived_by_verifier: false,
                    challenge_map_index: 0,
                }],
                challenge: "alpha".to_string(),
                num_challenges: 1,
                map_challenges: false,
            },
            RoundManifest {
                elements: vec![
                    ManifestEntry {
                        name: "T_1".to_string(),
                        num_bytes: g1_size,
                        derived_by_verifier: false,
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "T_1".to_string(),
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
            },
            RoundManifest {
                elements: vec![
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
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "w_3".to_string(),
                        num_bytes: fr_size,
                        derived_by_verifier: false,
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "w_3_omega".to_string(),
                        num_bytes: fr_size,
                        derived_by_verifier: false,
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "z_perm_omega".to_string(),
                        num_bytes: fr_size,
                        derived_by_verifier: false,
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "sigma_1".to_string(),
                        num_bytes: fr_size,
                        derived_by_verifier: false,
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "sigma_2".to_string(),
                        num_bytes: fr_size,
                        derived_by_verifier: false,
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "r".to_string(),
                        num_bytes: fr_size,
                        derived_by_verifier: false,
                        challenge_map_index: 0,
                    },
                    ManifestEntry {
                        name: "t".to_string(),
                        num_bytes: fr_size,
                        derived_by_verifier: true,
                        challenge_map_index: 0,
                    },
                ],
                challenge: "nu".to_string(),
                num_challenges: 10,
                map_challenges: false,
            },
            RoundManifest {
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
            },
        ];
        Manifest::new(round_manifests)
    }

    #[test]
    fn validate_transcript() {
        let g1_vector = vec![1u8; 64];
        let fr_vector = vec![1u8; 32];

        let mut transcript = Transcript::<PedersenBlake3s>::new(Some(create_manifest(0)), 32);
        transcript.add_element("circuit_size", vec![1, 2, 3, 4]);
        transcript.add_element("public_input_size", vec![1, 2, 3, 4]);
        transcript.apply_fiat_shamir("init");

        transcript.add_element("public_inputs", vec![]);

        transcript.add_element("W_1", g1_vector.clone());
        transcript.add_element("W_2", g1_vector.clone());
        transcript.add_element("W_3", g1_vector.clone());

        transcript.apply_fiat_shamir("beta");

        transcript.add_element("Z_PERM", g1_vector.clone());

        transcript.apply_fiat_shamir("alpha");

        transcript.add_element("T_1", g1_vector.clone());
        transcript.add_element("T_2", g1_vector.clone());
        transcript.add_element("T_3", g1_vector.clone());

        transcript.apply_fiat_shamir("z");

        transcript.add_element("w_1", fr_vector.clone());
        transcript.add_element("w_2", fr_vector.clone());
        transcript.add_element("w_3", fr_vector.clone());
        transcript.add_element("w_3_omega", fr_vector.clone());
        transcript.add_element("z_perm_omega", fr_vector.clone());
        transcript.add_element("sigma_1", fr_vector.clone());
        transcript.add_element("sigma_2", fr_vector.clone());
        transcript.add_element("r", fr_vector.clone());
        transcript.add_element("t", fr_vector.clone());

        transcript.apply_fiat_shamir("nu");

        transcript.add_element("PI_Z", g1_vector.clone());
        transcript.add_element("PI_Z_OMEGA", g1_vector.clone());

        transcript.apply_fiat_shamir("separator");

        let res = transcript.get_element("PI_Z_OMEGA");
        assert_eq!(res, g1_vector);
    }
}
