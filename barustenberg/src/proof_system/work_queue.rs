use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};
use std::rc::Rc;

use anyhow::Result;

use crate::ecc::PippengerRuntimeState;
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

pub(crate) struct WorkItem<'a, Fr: Field> {
    work_type: WorkType,
    mul_scalars: Option<Rc<Polynomial<'a, Fr>>>,
    tag: String,
    constant: Fr,
    index: usize,
}

pub(crate) struct QueuedFftInputs<'a, Fr: Field> {
    data: Rc<Polynomial<'a, Fr>>,
    shift_factor: Fr,
}

pub(crate) struct WorkQueue<'a, H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr> {
    key: Rc<ProvingKey<'a, Fr, G1Affine>>,
    transcript: Rc<Transcript<H, Fr, G1Affine>>,
    work_items: Vec<WorkItem<'a, Fr>>,
}

impl<'a, H: BarretenHasher, Fr: Field + FftField, G1Affine: AffineRepr>
    WorkQueue<'a, H, Fr, G1Affine>
{
    pub(crate) fn new(
        prover_key: Option<ProvingKey<'a, Fr, G1Affine>>,
        prover_transcript: Option<Rc<Transcript<H, Fr, G1Affine>>>,
    ) -> Self {
        let prover_key = Rc::new(prover_key.unwrap_or_default());
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
    ) -> Option<Rc<Polynomial<'a, Fr>>> {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if item.work_type == WorkType::ScalarMultiplication {
                if count == work_item_number {
                    return item.mul_scalars.clone();
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
    ) -> Result<Option<Rc<Polynomial<'a, Fr>>>> {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if item.work_type == WorkType::Ifft {
                if count == work_item_number {
                    //todo!("look at this code");
                    return Ok(Some(
                        self.key
                            .polynomial_store
                            .get(&format!("{}_lagrange", item.tag))
                            .unwrap()
                            .clone(),
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
                // barretenberg::polynomial wire(key->circuit_size);
                // memcpy((void*)wire.get_coefficients(), result, key->circuit_size * sizeof(barretenberg::fr));
                // key->polynomial_store.put(item.tag, std::move(wire));
                let mut wire = Polynomial::new(self.key.circuit_size);
                result.copy_from_slice(wire.get_coefficients());
                self.key.polynomial_store.put(item.tag, Rc::new(wire));
                return;
            }
        }
    }

    pub(crate) fn get_fft_data(&self, work_item_number: usize) -> Option<QueuedFftInputs<'a, Fr>> {
        let mut count = 0;
        for item in &self.work_items {
            if let WorkType::SmallFft = item.work_type {
                if count == work_item_number {
                    let wire = self.key.polynomial_store.get(&item.tag).unwrap();
                    return Some(QueuedFftInputs {
                        data: wire.clone(),
                        shift_factor: self.key.large_domain.root.pow([item.index as u64]),
                    });
                }
                count += 1;
            }
        }
        return None;
    }

    pub(crate) fn put_fft_data(&self, result: Vec<Fr>, work_item_number: usize) {
        let mut count = 0;
        for item in &self.work_items {
            if let WorkType::SmallFft = item.work_type {
                if count == work_item_number {
                    let n = self.key.circuit_size;
                    let mut wire_fft = Polynomial::new(4 * n + 4);

                    for i in 0..n {
                        wire_fft[4 * i + item.index] = result[i];
                    }
                    wire_fft[4 * n + item.index] = result[0];

                    self.key
                        .polynomial_store
                        .insert(&format!("{}_fft", item.tag), wire_fft);
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
                self.transcript.add_group_element(&item.tag, &result);
                return Ok(());
            }
        }
        Ok(())
    }

    pub(crate) fn flush_queue(&mut self) {
        self.work_items = vec![];
    }
    pub(crate) fn add_to_queue(&mut self, work_item: WorkItem<'a, Fr>) {
        #[cfg(target_arch = "wasm32")]
        // #[cfg(debug_assertions)]
        todo!("unimplemented");
        // if let WorkType::FFT = item.work_type {
        //     let large_root = &self.key.large_domain.root;
        //     let coset_shifts = [
        //         Fr(1),
        //         *large_root,
        //         large_root.sqr(),
        //         large_root.sqr() * *large_root,
        //     ];
        //     for i in 0..4 {
        //         self.work_item_queue.push(WorkItem {
        //             work_type: WorkType::SmallFFT,
        //             data: None,
        //             tag: item.tag.clone(),
        //             shift: coset_shifts[i],
        //             index: i,
        //         });
        //     }
        // } else {
        //     self.work_item_queue.push(item);
        // }
        #[cfg(not(target_arch = "wasm32"))]
        self.work_items.push(work_item);
    }

    pub(crate) fn process_queue(&mut self) -> Result<()> {
        for item in &self.work_items {
            match item.work_type {
                WorkType::ScalarMultiplication => {
                    // TODO: become a bigint then become a usize
                    let msm_size = item.constant.into();
                    let mut mul_scalars = item.mul_scalars.unwrap().get_mut_coefficients();

                    assert!(msm_size <= self.key.reference_string.get_monomial_size());

                    let srs_points = self.key.reference_string.get_monomial_points();

                    let runtime_state: PippengerRuntimeState<Fr, G1Affine> =
                        PippengerRuntimeState::new(msm_size);
                    let result = G1Affine::from(
                        runtime_state
                            .pippenger_unsafe(mul_scalars, *srs_points, msm_size)
                            .into(),
                    );

                    self.transcript.add_group_element(&item.tag, &result);
                }
                WorkType::SmallFft => {
                    let n = self.key.circuit_size;
                    let mut wire = self.key.polynomial_store.get(&item.tag).unwrap();
                    self.key
                        .small_domain
                        .coset_fft_with_generator_shift(wire.get_mut_coefficients(), item.constant);

                    if item.index != 0 {
                        let old_wire_fft = self
                            .key
                            .polynomial_store
                            .get(&(item.tag.clone() + "_fft"))
                            .unwrap();
                        for i in 0..n {
                            old_wire_fft[4 * i + item.index] = wire[i];
                        }
                        old_wire_fft[4 * n + item.index] = wire[0];
                    } else {
                        let mut wire_fft = Polynomial::new(4 * n + 4);
                        for i in 0..n {
                            wire_fft[4 * i + item.index] = wire[i];
                        }
                        self.key
                            .polynomial_store
                            .insert(&format!("{}_fft", item.tag.clone()), wire_fft);
                    }
                }
                WorkType::Fft => {
                    let mut wire = self.key.polynomial_store.get(&item.tag).unwrap().clone();

                    let mut wire_fft = *Rc::make_mut(&mut wire);
                    wire_fft.resize(4 * self.key.circuit_size + 4, Fr::zero());

                    self.key
                        .large_domain
                        .coset_fft_inplace(wire_fft.get_mut_coefficients());
                    for i in 0..4 {
                        wire_fft[4 * self.key.circuit_size + i] = wire_fft[i];
                    }

                    self.key
                        .polynomial_store
                        .insert(&format!("{}_fft", item.tag.clone()), wire_fft);
                }
                WorkType::Ifft => {
                    let wire_lagrange = self
                        .key
                        .polynomial_store
                        .get(&(format!("{}_lagrange", item.tag.clone())))
                        .unwrap();

                    let mut wire_monomial = Polynomial::new(self.key.circuit_size);
                    self.key.small_domain.ifft(
                        wire_lagrange.get_mut_coefficients(),
                        wire_monomial.get_mut_coefficients(),
                    );
                    self.key.polynomial_store.insert(&item.tag, wire_monomial);
                }
                _ => {}
            }
        }
        self.work_items.clear();
        Ok(())
    }

    fn get_queue(&self) -> Vec<WorkItem<'a, Fr>> {
        self.work_items
    }
}
