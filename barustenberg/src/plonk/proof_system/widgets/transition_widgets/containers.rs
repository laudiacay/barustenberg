use generic_array::GenericArray;
use once_cell::sync::Lazy;
use std::{
    cell::RefCell,
    ops::{Index, IndexMut},
    rc::Rc,
};
use typenum::Unsigned;

use crate::{
    ecc::fieldext::FieldExt, plonk::proof_system::types::polynomial_manifest::PolynomialIndex,
    polynomials::Polynomial,
};

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

pub(crate) trait PolyContainer<F: ark_ff::Field + ark_ff::FftField + FieldExt> {}

#[derive(Default)]
pub(crate) struct ChallengeArray<
    F: ark_ff::Field + ark_ff::FftField + FieldExt,
    NumRelations: generic_array::ArrayLength<F>,
> {
    pub(crate) elements: GenericArray<F, MaxNumChallengesTN>,
    pub(crate) alpha_powers: GenericArray<F, NumRelations>,
}

pub(crate) struct PolyArray<F: ark_ff::Field + ark_ff::FftField + FieldExt>(
    pub(crate) [(F, F); PolynomialIndex::MaxNumPolynomials as usize],
);

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> Index<PolynomialIndex> for PolyArray<F> {
    type Output = (F, F);

    fn index(&self, index: PolynomialIndex) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> IndexMut<PolynomialIndex> for PolyArray<F> {
    fn index_mut(&mut self, index: PolynomialIndex) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> Default for PolyArray<F> {
    fn default() -> Self {
        Self([(F::zero(), F::zero()); PolynomialIndex::MaxNumPolynomials as usize])
    }
}

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> PolyContainer<F> for PolyArray<F> {}

pub(crate) struct PolyPtrMap<F: ark_ff::Field + ark_ff::FftField + FieldExt> {
    pub(crate) coefficients: HashMap<PolynomialIndex, Rc<RefCell<Polynomial<F>>>>,
    pub(crate) block_mask: usize,
    pub(crate) index_shift: usize,
}

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> PolyPtrMap<F> {
    pub(crate) fn new() -> Self {
        Self {
            coefficients: HashMap::new(),
            block_mask: 0,
            index_shift: 0,
        }
    }
}

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> PolyContainer<F> for PolyPtrMap<F> {}

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> Index<PolynomialIndex> for PolyPtrMap<F> {
    type Output = Rc<RefCell<Polynomial<F>>>;
    fn index(&self, index: PolynomialIndex) -> &Self::Output {
        &self.coefficients[&index]
    }
}

pub(crate) struct CoefficientArray<F: ark_ff::Field + ark_ff::FftField + FieldExt>(
    [F; PolynomialIndex::MaxNumPolynomials as usize],
);
impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> Index<PolynomialIndex>
    for CoefficientArray<F>
{
    type Output = F;

    fn index(&self, index: PolynomialIndex) -> &Self::Output {
        &self.0[index as usize]
    }
}
impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> IndexMut<PolynomialIndex>
    for CoefficientArray<F>
{
    fn index_mut(&mut self, index: PolynomialIndex) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl<F: ark_ff::Field + ark_ff::FftField + FieldExt> Default for CoefficientArray<F> {
    fn default() -> Self {
        Self([F::zero(); PolynomialIndex::MaxNumPolynomials as usize])
    }
}
