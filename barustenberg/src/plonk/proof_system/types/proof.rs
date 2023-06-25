use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    pub(crate) proof_data: Vec<u8>,
}

#[derive(Default, Debug)]
pub(crate) struct CommitmentOpenProof {
    pub(crate) proof_data: Vec<u8>,
}

impl fmt::Display for Proof {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // // REFACTOR: This is copied from barretenberg/common/streams.hpp,
        // // which means we could just cout proof_data directly, but that breaks the build in the CI with
        // // a redefined operator<< error in barretenberg/stdlib/hash/keccak/keccak.test.cpp,
        // // which is something we really don't want to deal with right now.
        // let mut flags = f.flags();
        // write!(f, "[")?;
        // for byte in self.proof_data.iter() {
        //     write!(f, " {:02x}", byte)?;
        // }
        // write!(f, " ]")?;
        // flags.remove(fmt::Flags::ALTERNATE);
        // f.set_flags(flags);
        // Ok(())
        // TODO
        todo!("Proof::fmt")
    }
}
