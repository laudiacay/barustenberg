use std::sync::{Arc, RwLock};

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use anyhow::Result;

use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::polynomials::Polynomial;
use crate::transcript::{BarretenHasher, Transcript};

#[derive(PartialEq, Eq, Clone, Copy)]
pub(crate) enum WorkType {
    Fft,
    SmallFft,
    Ifft,
    ScalarMultiplication,
}

pub(crate) struct WorkItemInfo {
    num_scalar_multiplications: usize,
    num_ffts: usize,
    num_iffts: usize,
}

pub(crate) struct WorkItem<Fr: Field> {
    work_type: WorkType,
    mul_scalars: Arc<Vec<Fr>>,
    tag: String,
    constant: Fr,
    index: usize,
}

pub(crate) struct QueuedFftInputs<Fr: Field> {
    data: Vec<Fr>,
    shift_factor: Fr,
}

pub(crate) struct WorkQueue<'a, H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr> {
    key: Arc<RwLock<ProvingKey<'a, Fr, G1Affine>>>,
    transcript: Arc<RwLock<Transcript<H, Fr, G1Affine>>>,
    work_items: Vec<WorkItem<Fr>>,
}

impl<'a, H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr>
    WorkQueue<'a, H, Fr, G1Affine>
{
    pub(crate) fn new(
        prover_key: Option<Arc<RwLock<ProvingKey<'a, Fr, G1Affine>>>>,
        prover_transcript: Option<Arc<RwLock<Transcript<H, Fr, G1Affine>>>>,
    ) -> Self {
        let prover_key = prover_key.unwrap_or_default();
        let prover_transcript = prover_transcript.unwrap_or_default();
        WorkQueue {
            key: prover_key,
            transcript: prover_transcript,
            work_items: Vec::new(),
        }
    }

    pub(crate) fn get_queued_work_item_info(&self) -> WorkItemInfo {
        let mut num_scalar_multiplications = 0;
        let mut num_ffts = 0;
        let mut num_iffts = 0;
        for item in &self.work_items {
            match item.work_type {
                //WorkType::Fft => num_ffts += 1,
                WorkType::Fft => (),
                WorkType::SmallFft => num_ffts += 1,
                WorkType::Ifft => num_iffts += 1,
                WorkType::ScalarMultiplication => num_scalar_multiplications += 1,
            }
        }
        WorkItemInfo {
            num_scalar_multiplications,
            num_ffts,
            num_iffts,
        }
    }

    pub(crate) fn get_scalar_multiplication_data(
        &self,
        work_item_number: usize,
    ) -> Option<Arc<Vec<Fr>>> {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if item.work_type == WorkType::ScalarMultiplication {
                if count == work_item_number {
                    return Some(item.mul_scalars.clone());
                };
                count += 1;
            }
        }
        None
    }

    pub(crate) fn get_scalar_multiplication_size(&self, work_item_number: usize) -> usize {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if item.work_type == WorkType::ScalarMultiplication {
                if count == work_item_number {
                    todo!("look at this code");
                    // return static_cast<size_t>(static_cast<uint256_t>(item.constant));
                };
                count += 1;
            }
        }
        0
    }

    pub(crate) fn get_ifft_data(
        &self,
        work_item_number: usize,
    ) -> Result<Option<Arc<Polynomial<Fr>>>> {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if item.work_type == WorkType::Ifft {
                if count == work_item_number {
                    //todo!("look at this code");
                    return Ok(Some(
                        self.key
                            .read()?
                            .polynomial_store
                            .get(format!("{}_lagrange", item.tag)?.get_coefficients())?,
                    ));
                    // barretenberg::polynomial& wire = key->polynomial_store.get(item.tag + "_lagrange");
                    // return wire.get_coefficients();
                };
                count += 1;
            }
        }
        Ok(None)
    }

    pub(crate) fn put_ifft_data(&self, result: &mut Vec<Fr>, work_item_number: usize) {
        for (ix, item) in self.work_items.iter().enumerate() {
            if (ix == work_item_number) && (item.work_type == WorkType::Ifft) {
                todo!("looook");
                //  barretenberg::polynomial wire(key->circuit_size);
                // memcpy((void*)wire.get_coefficients(), result, key->circuit_size * sizeof(barretenberg::fr));
                // key->polynomial_store.put(item.tag, std::move(wire));
                return;
            }
        }
    }

    pub(crate) fn get_fft_data(&self, work_item_number: usize) -> Option<Arc<QueuedFftInputs<Fr>>> {
        let mut count = 0;
        for item in &self.work_item_queue {
            if let WorkType::SmallFft = item.work_type {
                if count == work_item_number {
                    let wire = self.key.polynomial_store.get(&item.tag).unwrap();
                    return QueuedFftInputs {
                        coefficients: wire.get_coefficients(),
                        power: self.key.large_domain.root.pow(item.index as u64),
                    };
                }
                count += 1;
            }
        }
        QueuedFftInputs {
            coefficients: None,
            power: Fr::zero(),
        }
    }

    pub(crate) fn put_fft_data(&self, result: Vec<Fr>, work_item_number: usize) {
        let mut count = 0;
        for item in &self.work_items {
            if let WorkType::SmallFft = item.work_type {
                if count == work_item_number {
                    let n = self.key.read().unwrap().circuit_size;
                    let mut wire_fft = Polynomial::new(4 * n + 4);

                    for i in 0..n {
                        wire_fft[4 * i + item.index] = result[i];
                    }
                    wire_fft[4 * n + item.index] = result[0];

                    self.key
                        .write()
                        .unwrap()
                        .polynomial_store
                        .insert(format!("{}_fft", item.tag), wire_fft);
                    return;
                }
                count += 1;
            }
        }
    }

    pub(crate) fn put_scalar_multiplication_data(
        &self,
        result: G1Affine,
        work_item_number: usize,
    ) -> Result<()> {
        for (idx, item) in self.work_items.iter().enumerate() {
            if item.work_type == WorkType::ScalarMultiplication && idx == work_item_number {
                self.transcript
                    .write()
                    .unwrap()
                    .add_group_element(&item.tag, &result);
                return Ok(());
            }
        }
        Ok(())
    }

    pub(crate) fn flush_queue(&mut self) {
        self.work_items = vec![];
    }
    pub(crate) fn add_to_queue(&mut self, work_item: WorkItem<Fr>) {
        todo!("whole wasm thing")
        /*
                #if defined(__wasm__)
            // #if 1
            if (item.work_type == WorkType::FFT) {
                const auto large_root = key->large_domain.root;
                barretenberg::fr coset_shifts[4]{
                    barretenberg::fr(1), large_root, large_root.sqr(), large_root.sqr() * large_root
                };
                work_item_queue.push_back({
                    WorkType::SMALL_FFT,
                    nullptr,
                    item.tag,
                    coset_shifts[0],
                    0,
                });
                work_item_queue.push_back({
                    WorkType::SMALL_FFT,
                    nullptr,
                    item.tag,
                    coset_shifts[1],
                    1,
                });
                work_item_queue.push_back({
                    WorkType::SMALL_FFT,
                    nullptr,
                    item.tag,
                    coset_shifts[2],
                    2,
                });
                work_item_queue.push_back({
                    WorkType::SMALL_FFT,
                    nullptr,
                    item.tag,
                    coset_shifts[3],
                    3,
                });
            } else {
                work_item_queue.push_back(item);
            }
        #else
            work_item_queue.push_back(item);
        #endif
                 */
    }
    pub(crate) fn process_queue(&self) -> Result<()> {
        let key = self.key.write().unwrap();
        for item in &self.work_items {
            match item.work_type {
                WorkType::ScalarMultiplication => {
                    let msm_size =
                        usize::try_from(item.constant).expect("unable to convert constant to size");

                    assert!(msm_size <= key.reference_string.get_monomial_size());

                    let srs_points = key.reference_string.get_monomial_points();

                    let runtime_state =
                        barretenberg::scalar_multiplication::pippenger_runtime_state(msm_size);
                    let result =
                        G1Affine::from(barretenberg::scalar_multiplication::pippenger_unsafe(
                            &item.mul_scalars,
                            srs_points,
                            msm_size,
                            runtime_state,
                        ));

                    self.transcript
                        .write()?
                        .add_group_element(&item.tag, &result);
                }
                WorkType::SmallFft => {
                    let n = self.key.circuit_size;
                    let mut wire = key.polynomial_store.get_mut(&item.tag).unwrap().clone();

                    wire.coset_fft_with_generator_shift(&self.key.small_domain, item.constant);

                    if item.index != 0 {
                        let old_wire_fft = self
                            .key
                            .polynomial_store
                            .get_mut(&(item.tag.clone() + "_fft"))
                            .unwrap();
                        for i in 0..n {
                            old_wire_fft[4 * i + item.index] = wire[i];
                        }
                        old_wire_fft[4 * n + item.index] = wire[0];
                    } else {
                        let mut wire_fft = vec![0; 4 * n + 4];
                        for i in 0..n {
                            wire_fft[4 * i + item.index] = wire[i];
                        }
                        self.key
                            .polynomial_store
                            .insert(item.tag.clone() + "_fft", wire_fft);
                    }
                }
                WorkType::FFT => {
                    key.polynomial_store.get_mut(&item.tag).unwrap().clone();

                    let mut wire_fft = wire.clone();
                    wire_fft.resize(4 * self.key.circuit_size + 4, 0);

                    wire_fft.coset_fft(&self.key.large_domain);
                    for i in 0..4 {
                        wire_fft[4 * self.key.circuit_size + i] = wire_fft[i];
                    }

                    self.key
                        .polynomial_store
                        .insert(item.tag.clone() + "_fft", wire_fft);
                }
                WorkType::IFFT => {
                    let wire_lagrange = key
                        .polynomial_store
                        .get_mut(&(item.tag.clone() + "_lagrange"))
                        .unwrap();

                    let mut wire_monomial = vec![0; self.key.circuit_size];
                    polynomial_arithmetic::ifft(
                        &wire_lagrange,
                        &mut wire_monomial,
                        &self.key.read().unwrap().small_domain,
                    );
                    self.key
                        .polynomial_store
                        .insert(item.tag.clone(), wire_monomial);
                }
                _ => {}
            }
        }
        self.work_items.clear();
        Ok(())
    }

    fn get_queue(&self) -> Vec<WorkItem<Fr>> {
        self.work_items
    }
}
