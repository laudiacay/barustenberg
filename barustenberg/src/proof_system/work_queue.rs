use std::sync::{Arc, RwLock};

use ark_ec::AffineRepr;
use ark_ff::{FftField, Field};

use anyhow::Result;

use crate::plonk::proof_system::proving_key::ProvingKey;
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

    pub(crate) fn get_ifft_data(&self, work_item_number: usize) -> Result<Option<Arc<Vec<Fr>>>> {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if item.work_type == WorkType::Ifft {
                if count == work_item_number {
                    //todo!("look at this code");
                    return Ok(Some(self.key.read()?.polynomial_store.get(format!("{}_lagrange", item.tag)?.get_coefficients()));
                    // barretenberg::polynomial& wire = key->polynomial_store.get(item.tag + "_lagrange");
                    // return wire.get_coefficients();
                };
                count += 1;
            }
        }
        Ok(None)
    }

    pub(crate) fn put_ifft_data(&self, result: &mut Vec<Fr>, _work_item_number: usize) {
        todo!("do it");
        let mut count = 0;
        for item in self.work_items {
            if (count == work_item_number) && (item.work_type == WorkType::Ifft) {
                todo!("looook");
                //  barretenberg::polynomial wire(key->circuit_size);
                // memcpy((void*)wire.get_coefficients(), result, key->circuit_size * sizeof(barretenberg::fr));
                // key->polynomial_store.put(item.tag, std::move(wire));
                return;
            }
            count += 1;
        }
    }

    pub(crate) fn get_fft_data(
        &self,
        _work_item_number: usize,
    ) -> Option<Arc<QueuedFftInputs<Fr>>> {
        todo!("do it");
    }

    pub(crate) fn put_fft_data(&self, _result: Vec<Fr>, _work_item_number: usize) {
        todo!("do it")
    }

    pub(crate) fn put_scalar_multiplication_data(
        &self,
        _result: G1Affine,
        _work_item_number: usize,
    ) {
        todo!("do it")
    }

    pub(crate) fn flush_queue(&mut self) {
        self.work_items = vec![];
    }
    pub(crate) fn woradd_to_queue(&mut self, work_item: WorkItem<Fr>) {
        todo!("whole wasm thing")
    }
    pub(crate) fn process_queue(&self) {
        todo!("aaaaagh")
    }
    fn get_queue(&self) {
        todo!("aagh")
    }
}
