use ark_ff::{FftField, Field, Zero};
use std::sync::{Arc, RwLock};

use anyhow::Result;

use ark_bn254::{Fr, G1Affine};

use crate::ecc::curves::bn254_scalar_multiplication::PippengerRuntimeState;
use crate::plonk::proof_system::proving_key::ProvingKey;
use crate::polynomials::Polynomial;
use crate::transcript::{BarretenHasher, Transcript};

#[derive(Clone, Debug)]
pub(crate) enum Work {
    Fft {
        index: usize,
    },
    SmallFft {
        constant: Fr,
        index: usize,
    },
    Ifft,
    ScalarMultiplication {
        constant: Fr,
        mul_scalars: Arc<RwLock<Polynomial<Fr>>>,
    },
}

pub(crate) struct WorkItemInfo {
    num_scalar_multiplications: usize,
    num_ffts: usize,
    num_iffts: usize,
}

pub(crate) enum WorkItemConstant<Fr: Field + FftField> {
    Fr(Fr),
    USize(usize),
}

impl<Fr: Field + FftField> From<usize> for WorkItemConstant<Fr> {
    fn from(item: usize) -> Self {
        WorkItemConstant::USize(item)
    }
}

#[derive(Debug)]
pub(crate) struct WorkItem {
    pub(crate) work: Work,
    pub(crate) tag: String,
}

pub(crate) struct QueuedFftInputs<Fr: Field + FftField> {
    data: Arc<RwLock<Polynomial<Fr>>>,
    shift_factor: Fr,
}

#[derive(Debug)]
pub(crate) struct WorkQueue<H: BarretenHasher> {
    key: Arc<RwLock<ProvingKey<Fr>>>,
    transcript: Arc<RwLock<Transcript<H>>>,
    work_items: Vec<WorkItem>,
}

/* I do not think this works as intended, so I changed it
/// TODO this is super fucked up...
unsafe fn field_element_to_usize<F: Field + FftField>(element: F) -> usize {
    // pretending to be this: static_cast<size_t>(static_cast<uint256_t>(item.constant));
    // first turn it into a u256 (by memtransmute into a slice!)
    let u256_bytes: [u8; 32] = std::mem::transmute_copy(&element);
    eprintln!("{:?}", u256_bytes);
    std::mem::transmute_copy(&u256_bytes)
}
*/

// ... although it seems like ark_ff has a scuffed API, so I have a similarly
// scuffed implementation...
fn field_element_to_usize<F: Field + FftField>(element: F) -> usize {
    format!("{}", element).parse::<usize>().expect("WorkQueue: element larger than usize")
}

impl<H: BarretenHasher> WorkQueue<H> {
    pub(crate) fn new(
        prover_key: Option<Arc<RwLock<ProvingKey<Fr>>>>,
        prover_transcript: Option<Arc<RwLock<Transcript<H>>>>,
    ) -> Self {
        WorkQueue {
            key: prover_key.unwrap_or_default(),
            transcript: prover_transcript.unwrap_or_default(),
            work_items: Vec::new(),
        }
    }

    pub(crate) fn get_queued_work_item_info(&self) -> WorkItemInfo {
        let mut num_scalar_multiplications = 0;
        let mut num_ffts = 0;
        let mut num_iffts = 0;
        for item in &self.work_items {
            match item.work {
                //WorkType::Fft => num_ffts += 1,
                Work::Fft { .. } => (),
                Work::SmallFft { .. } => num_ffts += 1,
                Work::Ifft => num_iffts += 1,
                Work::ScalarMultiplication { .. } => num_scalar_multiplications += 1,
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
    ) -> Option<Arc<RwLock<Polynomial<Fr>>>> {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if let Work::ScalarMultiplication { mul_scalars, .. } = item.work.clone() {
                if count == work_item_number {
                    return Some(mul_scalars);
                };
                count += 1;
            }
        }
        None
    }

    pub(crate) fn get_scalar_multiplication_size(&self, work_item_number: usize) -> usize {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if let Work::ScalarMultiplication { constant, .. } = item.work {
                if count == work_item_number {
                    return unsafe { field_element_to_usize(constant) };
                };
                count += 1;
            }
        }
        0
    }

    pub(crate) fn get_ifft_data(
        &self,
        work_item_number: usize,
    ) -> Result<Option<Arc<RwLock<Polynomial<Fr>>>>> {
        let mut count: usize = 0;
        for item in self.work_items.iter() {
            if let Work::Ifft = item.work {
                if count == work_item_number {
                    //todo!("look at this code");
                    return Ok(Some(
                        self.key
                            .read()
                            .unwrap()
                            .polynomial_store
                            .get(&format!("{}_lagrange", item.tag))
                            .unwrap(),
                    ));
                };
                count += 1;
            }
        }
        Ok(None)
    }

    pub(crate) fn put_ifft_data(&mut self, result: &mut [Fr], work_item_number: usize) {
        for (ix, item) in self.work_items.iter().enumerate() {
            if let Work::Ifft = item.work {
                if ix == work_item_number {
                    // barretenberg::polynomial wire(key->circuit_size);
                    // memcpy((void*)wire.get_coefficients(), result, key->circuit_size * sizeof(barretenberg::fr));
                    // key->polynomial_store.put(item.tag, std::move(wire));
                    let wire = Polynomial::new(self.key.read().unwrap().circuit_size);
                    result.copy_from_slice(wire.coefficients.as_slice());
                    (*self.key)
                        .write()
                        .unwrap()
                        .polynomial_store
                        .put(item.tag.clone(), wire);
                    return;
                }
            }
        }
    }

    pub(crate) fn get_fft_data(&self, work_item_number: usize) -> Option<QueuedFftInputs<Fr>> {
        let mut count = 0;
        for item in &self.work_items {
            if let Work::SmallFft { index, .. } = item.work {
                if count == work_item_number {
                    let wire = self
                        .key
                        .read()
                        .unwrap()
                        .polynomial_store
                        .get(&item.tag)
                        .unwrap();
                    return Some(QueuedFftInputs {
                        data: wire,
                        shift_factor: self
                            .key
                            .read()
                            .unwrap()
                            .large_domain
                            .root
                            .pow([index as u64]),
                    });
                }
                count += 1;
            }
        }
        None
    }

    pub(crate) fn put_fft_data(&self, result: Vec<Fr>, work_item_number: usize) {
        let mut count = 0;
        for item in &self.work_items {
            if let Work::SmallFft { index, .. } = item.work {
                if count == work_item_number {
                    let n = self.key.read().unwrap().circuit_size;
                    let mut wire_fft = Polynomial::new(4 * n + 4);

                    for i in 0..n {
                        wire_fft[4 * i + index] = result[i];
                    }
                    wire_fft[4 * n + index] = result[0];

                    (*self.key)
                        .write()
                        .unwrap()
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
            if let Work::ScalarMultiplication { .. } = &item.work {
                if idx == work_item_number {
                    (*self.transcript)
                        .write()
                        .unwrap()
                        .add_group_element(&item.tag, &result);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub(crate) fn flush_queue(&mut self) {
        self.work_items = vec![];
    }
    pub(crate) fn add_to_queue(&mut self, work_item: WorkItem) {
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
            match &item.work {
                Work::ScalarMultiplication {
                    constant,
                    mul_scalars,
                } => {
                    let msm_size = unsafe { field_element_to_usize(*constant) };

                    assert!(
                        msm_size
                            <= (*(*self.key).read().unwrap().reference_string)
                                .write()
                                .unwrap()
                                .get_monomial_size()
                    );

                    let srs_points: Arc<Vec<G1Affine>> =
                        (*self.key.read().unwrap().reference_string)
                            .write()
                            .unwrap()
                            .get_monomial_points();

                    let mut runtime_state: PippengerRuntimeState<ark_bn254::g1::Config> =
                        PippengerRuntimeState::new(msm_size);
                    let result = G1Affine::from(runtime_state.pippenger_unsafe(
                        (*mul_scalars).write().unwrap().coefficients.as_mut_slice(),
                        &(*srs_points)[..],
                        msm_size,
                    ));

                    (*self.transcript)
                        .write()
                        .unwrap()
                        .add_group_element(&item.tag, &result);
                }
                Work::SmallFft { index, constant } => {
                    let n = self.key.read().unwrap().circuit_size;
                    let wire = self
                        .key
                        .read()
                        .unwrap()
                        .polynomial_store
                        .get(&item.tag)
                        .unwrap();
                    self.key
                        .read()
                        .unwrap()
                        .small_domain
                        .coset_fft_with_generator_shift(
                            (*wire).write().unwrap().coefficients.as_mut_slice(),
                            *constant,
                        );

                    if *index != 0 {
                        let old_wire_fft = self
                            .key
                            .read()
                            .unwrap()
                            .polynomial_store
                            .get(&(item.tag.clone() + "_fft"))
                            .unwrap()
                            .clone();
                        for i in 0..n {
                            (*old_wire_fft).write().unwrap()[4 * i + index] =
                                wire.read().unwrap()[i];
                        }
                        (*old_wire_fft).write().unwrap()[4 * n + index] = wire.read().unwrap()[0];
                    } else {
                        let mut wire_fft = Polynomial::new(4 * n + 4);
                        for i in 0..n {
                            wire_fft[4 * i + index] = wire.read().unwrap()[i];
                        }

                        (*self.key)
                            .write()
                            .unwrap()
                            .polynomial_store
                            .insert(&format!("{}_fft", item.tag.clone()), wire_fft);
                    }
                }
                Work::Fft { .. } => {
                    let mut wire_fft = self
                        .key
                        .read()
                        .unwrap()
                        .polynomial_store
                        .get(&item.tag)
                        .unwrap()
                        .read()
                        .unwrap()
                        .clone();

                    wire_fft.resize(4 * self.key.read().unwrap().circuit_size + 4, Fr::zero());

                    (*self.key)
                        .read()
                        .unwrap()
                        .large_domain
                        .coset_fft_inplace(wire_fft.coefficients.as_mut_slice());
                    for i in 0..4 {
                        wire_fft[4 * (*self.key).read().unwrap().circuit_size + i] = wire_fft[i];
                    }

                    (*self.key)
                        .write()
                        .unwrap()
                        .polynomial_store
                        .insert(&format!("{}_fft", item.tag.clone()), wire_fft);
                }
                Work::Ifft => {
                    let wire_lagrange = (*self.key)
                        .read()
                        .unwrap()
                        .polynomial_store
                        .get(&(format!("{}_lagrange", item.tag.clone())))
                        .unwrap();

                    let mut wire_monomial =
                        Polynomial::new((*self.key).read().unwrap().circuit_size);
                    (*self.key).read().unwrap().small_domain.ifft(
                        wire_lagrange.write().unwrap().coefficients.as_mut_slice(),
                        wire_monomial.coefficients.as_mut_slice(),
                    );
                    (*self.key)
                        .write()
                        .unwrap()
                        .polynomial_store
                        .insert(&item.tag, wire_monomial);
                }
            }
        }
        self.work_items.clear();
        Ok(())
    }

    fn get_queue(&self) -> &Vec<WorkItem> {
        &self.work_items
    }
}
