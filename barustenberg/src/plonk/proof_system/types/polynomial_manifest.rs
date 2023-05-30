use std::ops::Index;

#[derive(Debug, Clone)]
struct PolynomialDescriptor {
    commitment_label: String,
    pub polynomial_label: String,
    pub requires_shifted_evaluation: bool,
    source: PolynomialSource,
    pub index: PolynomialIndex,
}

impl PolynomialDescriptor {
    fn new(
        commitment_label: String,
        polynomial_label: String,
        requires_shifted_evaluation: bool,
        source: PolynomialSource,
        index: PolynomialIndex,
    ) -> Self {
        PolynomialDescriptor {
            commitment_label,
            polynomial_label,
            requires_shifted_evaluation,
            source,
            index,
        }
    }
}
#[derive(Clone, Default)]
pub(crate) struct PolynomialManifest {
    manifest: Vec<PolynomialDescriptor>,
}

impl PolynomialManifest {
    pub fn len(&self) -> usize {
        todo!()
    }
    pub fn get(&self, index: PolynomialIndex) -> Vec<PolynomialDescriptor> {
        self.manifest
    }
}

impl IntoIterator for PolynomialManifest {
    type Item = PolynomialDescriptor;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.manifest.into_iter()
    }
}

impl Index<PolynomialIndex> for PolynomialManifest {
    type Output = PolynomialDescriptor;

    fn index(&self, index: PolynomialIndex) -> &Self::Output {
        self.manifest.get(index as usize).unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PolynomialSource {
    WITNESS,
    SELECTOR,
    PERMUTATION,
    OTHER,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EvaluationType {
    NonShifted,
    Shifted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PolynomialIndex {
    Q1,
    Q2,
    Q3,
    Q4,
    Q5,
    QM,
    QC,
    QArithmetic,
    QFixedBase,
    QRange,
    QSort,
    QLogic,
    Table1,
    Table2,
    Table3,
    Table4,
    TableIndex,
    TableType,
    QElliptic,
    QAux,
    Sigma1,
    Sigma2,
    Sigma3,
    Sigma4,
    Id1,
    Id2,
    Id3,
    Id4,
    W1,
    W2,
    W3,
    W4,
    S,
    Z,
    ZLookup,
    LagrangeFirst,
    LagrangeLast,
    // SUBGROUP_GENERATOR,
    MaxNumPolynomials,
}

impl From<usize> for PolynomialIndex {
    fn from(index: usize) -> Self {
        match index {
            0 => PolynomialIndex::Q1,
            1 => PolynomialIndex::Q2,
            2 => PolynomialIndex::Q3,
            3 => PolynomialIndex::Q4,
            4 => PolynomialIndex::Q5,
            5 => PolynomialIndex::QM,
            6 => PolynomialIndex::QC,
            7 => PolynomialIndex::QArithmetic,
            8 => PolynomialIndex::QFixedBase,
            9 => PolynomialIndex::QRange,
            10 => PolynomialIndex::QSort,
            11 => PolynomialIndex::QLogic,
            12 => PolynomialIndex::Table1,
            13 => PolynomialIndex::Table2,
            14 => PolynomialIndex::Table3,
            15 => PolynomialIndex::Table4,
            16 => PolynomialIndex::TableIndex,
            17 => PolynomialIndex::TableType,
            18 => PolynomialIndex::QElliptic,
            19 => PolynomialIndex::QAux,
            20 => PolynomialIndex::Sigma1,
            21 => PolynomialIndex::Sigma2,
            22 => PolynomialIndex::Sigma3,
            23 => PolynomialIndex::Sigma4,
            24 => PolynomialIndex::Id1,
            25 => PolynomialIndex::Id2,
            26 => PolynomialIndex::Id3,
            27 => PolynomialIndex::Id4,
            28 => PolynomialIndex::W1,
            29 => PolynomialIndex::W2,
            30 => PolynomialIndex::W3,
            31 => PolynomialIndex::W4,
            32 => PolynomialIndex::S,
            33 => PolynomialIndex::Z,
            34 => PolynomialIndex::ZLookup,
            35 => PolynomialIndex::LagrangeFirst,
            36 => PolynomialIndex::LagrangeLast,
            // 37 => PolynomialIndex::SUBGROUP_GENERATOR,
            38 => PolynomialIndex::MaxNumPolynomials,
            _ => panic!("Invalid polynomial index"),
        }
    }
}
