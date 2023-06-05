use generic_array::GenericArray;
use once_cell::sync::Lazy;
use std::ops::{Index, IndexMut};
use typenum::Unsigned;

use crate::{
    plonk::proof_system::types::polynomial_manifest::PolynomialIndex, polynomials::Polynomial,
};

use ark_ff::Field;
use std::collections::HashMap;

pub(crate) enum ChallengeIndex {
    Alpha,
    Beta,
    Gamma,
    Eta,
    Zeta,
    MaxNumChallenges,
}

pub(crate) const CHALLENGE_BIT_ALPHA: usize = 1 << (ChallengeIndex::Alpha as usize);
pub(crate) const CHALLENGE_BIT_BETA: usize = 1 << (ChallengeIndex::Beta as usize);
pub(crate) const CHALLENGE_BIT_GAMMA: usize = 1 << (ChallengeIndex::Gamma as usize);
pub(crate) const CHALLENGE_BIT_ETA: usize = 1 << (ChallengeIndex::Eta as usize);
pub(crate) const CHALLENGE_BIT_ZETA: usize = 1 << (ChallengeIndex::Zeta as usize);

// need maxnumchallenges as a typenum, not just as an enum
pub(crate) type MaxNumChallengesTN = typenum::consts::U5;

// and check its correspondance with the enum before we continue...
static _MAX_NUM_CHALLENGES_CHECK: Lazy<()> = Lazy::new(|| {
    assert_eq!(
        MaxNumChallengesTN::to_usize(),
        ChallengeIndex::MaxNumChallenges as usize
    );
});

pub(crate) trait PolyContainer<F: Field> {}

#[derive(Default)]
pub(crate) struct ChallengeArray<F: Field, NumRelations: generic_array::ArrayLength<F>> {
    pub(crate) elements: GenericArray<F, MaxNumChallengesTN>,
    pub(crate) alpha_powers: GenericArray<F, NumRelations>,
}

pub(crate) struct PolyArray<Field>(
    pub(crate) [(Field, Field); PolynomialIndex::MaxNumPolynomials as usize],
);

impl<F: Field> Index<PolynomialIndex> for PolyArray<F> {
    type Output = (F, F);

    fn index(&self, index: PolynomialIndex) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl<F: Field> IndexMut<PolynomialIndex> for PolyArray<F> {
    fn index_mut(&mut self, index: PolynomialIndex) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl<F: Field> Default for PolyArray<F> {
    fn default() -> Self {
        Self([(F::zero(), F::zero()); PolynomialIndex::MaxNumPolynomials as usize])
    }
}

impl<F: Field> PolyContainer<F> for PolyArray<F> {}

pub(crate) struct PolyPtrMap<F: Field> {
    pub(crate) coefficients: HashMap<PolynomialIndex, Polynomial<F>>,
    pub(crate) block_mask: usize,
    pub(crate) index_shift: usize,
}

impl<F: Field> PolyPtrMap<F> {
    pub(crate) fn new() -> Self {
        Self {
            coefficients: HashMap::new(),
            block_mask: 0,
            index_shift: 0,
        }
    }
}

impl<F: Field> PolyContainer<F> for PolyPtrMap<F> {}

impl<F: Field> Index<PolynomialIndex> for PolyPtrMap<F> {
    type Output = Polynomial<F>;
    fn index(&self, index: PolynomialIndex) -> &Self::Output {
        &self.coefficients[&index]
    }
}

pub(crate) struct CoefficientArray<F: Field>([F; PolynomialIndex::MaxNumPolynomials as usize]);
impl<F: Field> Index<PolynomialIndex> for CoefficientArray<F> {
    type Output = F;

    fn index(&self, index: PolynomialIndex) -> &Self::Output {
        &self.0[index as usize]
    }
}
impl<F: Field> Default for CoefficientArray<F> {
    fn default() -> Self {
        Self([F::zero(); PolynomialIndex::MaxNumPolynomials as usize])
    }
}
