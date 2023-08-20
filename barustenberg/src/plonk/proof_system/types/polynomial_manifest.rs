use std::ops::Index;

use crate::plonk::composer::composer_base::ComposerType;

#[derive(Debug, Clone)]
pub(crate) struct PolynomialDescriptor {
    pub(crate) commitment_label: String,
    pub(crate) polynomial_label: String,
    pub(crate) requires_shifted_evaluation: bool,
    pub(crate) source: PolynomialSource,
    pub(crate) index: PolynomialIndex,
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
#[derive(Debug, Clone, Default)]
pub(crate) struct PolynomialManifest {
    pub(crate) manifest: Vec<PolynomialDescriptor>,
}

lazy_static::lazy_static! {
/*
    PolynomialDescriptor("W_1", "w_1", false, WITNESS, W_1),                 //
    PolynomialDescriptor("W_2", "w_2", false, WITNESS, W_2),                 //
    PolynomialDescriptor("W_3", "w_3", false, WITNESS, W_3),                 //
    PolynomialDescriptor("Z_PERM", "z_perm", true, WITNESS, Z),              //
    PolynomialDescriptor("Q_1", "q_1", false, SELECTOR, Q_1),                //
    PolynomialDescriptor("Q_2", "q_2", false, SELECTOR, Q_2),                //
    PolynomialDescriptor("Q_3", "q_3", false, SELECTOR, Q_3),                //
    PolynomialDescriptor("Q_M", "q_m", false, SELECTOR, Q_M),                //
    PolynomialDescriptor("Q_C", "q_c", false, SELECTOR, Q_C),                //
    PolynomialDescriptor("SIGMA_1", "sigma_1", false, PERMUTATION, SIGMA_1), //
    PolynomialDescriptor("SIGMA_2", "sigma_2", false, PERMUTATION, SIGMA_2), //
    PolynomialDescriptor("SIGMA_3", "sigma_3", false, PERMUTATION, SIGMA_3), //
};
 */

    pub(crate) static ref STANDARD_POLYNOMIAL_MANIFEST: PolynomialManifest = {
    let manifest = vec![
        PolynomialDescriptor::new("W_1".to_string(), "w_1".to_string(), false, PolynomialSource::Witness, PolynomialIndex::W1),
        PolynomialDescriptor::new("W_2".to_string(), "w_2".to_string(), false, PolynomialSource::Witness, PolynomialIndex::W2),
        PolynomialDescriptor::new("W_3".to_string(), "w_3".to_string(), false, PolynomialSource::Witness, PolynomialIndex::W3),
        PolynomialDescriptor::new("Z_PERM".to_string(), "z_perm".to_string(), true, PolynomialSource::Witness, PolynomialIndex::Z),
        PolynomialDescriptor::new("Q_1".to_string(), "q_1".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q1),
        PolynomialDescriptor::new("Q_2".to_string(), "q_2".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q2),
        PolynomialDescriptor::new("Q_3".to_string(), "q_3".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q3),
        PolynomialDescriptor::new("Q_M".to_string(), "q_m".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QM),
        PolynomialDescriptor::new("Q_C".to_string(), "q_c".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QC),
        PolynomialDescriptor::new("SIGMA_1".to_string(), "sigma_1".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma1),
        PolynomialDescriptor::new("SIGMA_2".to_string(), "sigma_2".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma2),
        PolynomialDescriptor::new("SIGMA_3".to_string(), "sigma_3".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma3),

    ] ;
    PolynomialManifest { manifest }};
    pub(crate) static ref STANDARD_MANIFEST_SIZE: usize = STANDARD_POLYNOMIAL_MANIFEST.len();

    /*
        PolynomialDescriptor("W_1", "w_1", true, WITNESS, W_1),                              //
    PolynomialDescriptor("W_2", "w_2", true, WITNESS, W_2),                              //
    PolynomialDescriptor("W_3", "w_3", true, WITNESS, W_3),                              //
    PolynomialDescriptor("W_4", "w_4", true, WITNESS, W_4),                              //
    PolynomialDescriptor("Z_PERM", "z_perm", true, WITNESS, Z),                          //
    PolynomialDescriptor("Q_1", "q_1", false, SELECTOR, Q_1),                            //
    PolynomialDescriptor("Q_2", "q_2", false, SELECTOR, Q_2),                            //
    PolynomialDescriptor("Q_3", "q_3", false, SELECTOR, Q_3),                            //
    PolynomialDescriptor("Q_4", "q_4", false, SELECTOR, Q_4),                            //
    PolynomialDescriptor("Q_5", "q_5", false, SELECTOR, Q_5),                            //
    PolynomialDescriptor("Q_M", "q_m", false, SELECTOR, Q_M),                            //
    PolynomialDescriptor("Q_C", "q_c", false, SELECTOR, Q_C),                            //
    PolynomialDescriptor("Q_ARITHMETIC", "q_arith", false, SELECTOR, Q_ARITHMETIC),      //
    PolynomialDescriptor("Q_RANGE", "q_range", false, SELECTOR, Q_RANGE),                //
    PolynomialDescriptor("Q_FIXED_BASE", "q_fixed_base", false, SELECTOR, Q_FIXED_BASE), //
    PolynomialDescriptor("Q_LOGIC", "q_logic", false, SELECTOR, Q_LOGIC),                //
    PolynomialDescriptor("SIGMA_1", "sigma_1", false, PERMUTATION, SIGMA_1),             //
    PolynomialDescriptor("SIGMA_2", "sigma_2", false, PERMUTATION, SIGMA_2),             //
    PolynomialDescriptor("SIGMA_3", "sigma_3", false, PERMUTATION, SIGMA_3),             //
    PolynomialDescriptor("SIGMA_4", "sigma_4", false, PERMUTATION, SIGMA_4),             //
};
 */
    pub(crate) static ref TURBO_POLYNOMIAL_MANIFEST: PolynomialManifest = {
        let manifest = vec![
            PolynomialDescriptor::new("W_1".to_string(), "w_1".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W1),
            PolynomialDescriptor::new("W_2".to_string(), "w_2".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W2),
            PolynomialDescriptor::new("W_3".to_string(), "w_3".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W3),
            PolynomialDescriptor::new("W_4".to_string(), "w_4".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W4),
            PolynomialDescriptor::new("Z_PERM".to_string(), "z_perm".to_string(), true, PolynomialSource::Witness, PolynomialIndex::Z),
            PolynomialDescriptor::new("Q_1".to_string(), "q_1".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q1),
            PolynomialDescriptor::new("Q_2".to_string(), "q_2".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q2),
            PolynomialDescriptor::new("Q_3".to_string(), "q_3".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q3),
            PolynomialDescriptor::new("Q_4".to_string(), "q_4".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q4),
            PolynomialDescriptor::new("Q_5".to_string(), "q_5".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q5),
            PolynomialDescriptor::new("Q_M".to_string(), "q_m".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QM),
            PolynomialDescriptor::new("Q_C".to_string(), "q_c".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QC),
            PolynomialDescriptor::new("Q_ARITHMETIC".to_string(), "q_arith".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QArithmetic),
            PolynomialDescriptor::new("Q_RANGE".to_string(), "q_range".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QRange),
            PolynomialDescriptor::new("Q_FIXED_BASE".to_string(), "q_fixed_base".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QFixedBase),
            PolynomialDescriptor::new("Q_LOGIC".to_string(), "q_logic".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QLogic),
            PolynomialDescriptor::new("SIGMA_1".to_string(), "sigma_1".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma1),
            PolynomialDescriptor::new("SIGMA_2".to_string(), "sigma_2".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma2),
            PolynomialDescriptor::new("SIGMA_3".to_string(), "sigma_3".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma3),
            PolynomialDescriptor::new("SIGMA_4".to_string(), "sigma_4".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma4),

        ];
        PolynomialManifest { manifest }
    };

    pub(crate) static ref TURBO_MANIFEST_SIZE: usize = TURBO_POLYNOMIAL_MANIFEST.len();
    /*
        PolynomialDescriptor("W_1", "w_1", true, WITNESS, W_1),                         //
    PolynomialDescriptor("W_2", "w_2", true, WITNESS, W_2),                         //
    PolynomialDescriptor("W_3", "w_3", true, WITNESS, W_3),                         //
    PolynomialDescriptor("W_4", "w_4", true, WITNESS, W_4),                         //
    PolynomialDescriptor("S", "s", true, WITNESS, S),                               //
    PolynomialDescriptor("Z_PERM", "z_perm", true, WITNESS, Z),                     //
    PolynomialDescriptor("Z_LOOKUP", "z_lookup", true, WITNESS, Z_LOOKUP),          //
    PolynomialDescriptor("Q_1", "q_1", false, SELECTOR, Q_1),                       //
    PolynomialDescriptor("Q_2", "q_2", false, SELECTOR, Q_2),                       //
    PolynomialDescriptor("Q_3", "q_3", false, SELECTOR, Q_3),                       //
    PolynomialDescriptor("Q_4", "q_4", false, SELECTOR, Q_4),                       //
    PolynomialDescriptor("Q_M", "q_m", false, SELECTOR, Q_M),                       //
    PolynomialDescriptor("Q_C", "q_c", false, SELECTOR, Q_C),                       //
    PolynomialDescriptor("Q_ARITHMETIC", "q_arith", false, SELECTOR, Q_ARITHMETIC), //
    PolynomialDescriptor("Q_SORT", "q_sort", false, SELECTOR, Q_SORT),              //
    PolynomialDescriptor("Q_ELLIPTIC", "q_elliptic", false, SELECTOR, Q_ELLIPTIC),  //
    PolynomialDescriptor("Q_AUX", "q_aux", false, SELECTOR, Q_AUX),                 //
    PolynomialDescriptor("SIGMA_1", "sigma_1", false, PERMUTATION, SIGMA_1),        //
    PolynomialDescriptor("SIGMA_2", "sigma_2", false, PERMUTATION, SIGMA_2),        //
    PolynomialDescriptor("SIGMA_3", "sigma_3", false, PERMUTATION, SIGMA_3),        //
    PolynomialDescriptor("SIGMA_4", "sigma_4", false, PERMUTATION, SIGMA_4),        //
    PolynomialDescriptor("TABLE_1", "table_value_1", true, SELECTOR, TABLE_1),      //
    PolynomialDescriptor("TABLE_2", "table_value_2", true, SELECTOR, TABLE_2),      //
    PolynomialDescriptor("TABLE_3", "table_value_3", true, SELECTOR, TABLE_3),      //
    PolynomialDescriptor("TABLE_4", "table_value_4", true, SELECTOR, TABLE_4),      //
    PolynomialDescriptor("TABLE_TYPE", "table_type", false, SELECTOR, TABLE_TYPE),  //
    PolynomialDescriptor("ID_1", "id_1", false, PERMUTATION, ID_1),                 //
    PolynomialDescriptor("ID_2", "id_2", false, PERMUTATION, ID_2),                 //
    PolynomialDescriptor("ID_3", "id_3", false, PERMUTATION, ID_3),                 //
    PolynomialDescriptor("ID_4", "id_4", false, PERMUTATION, ID_4),                 //
     */
    // ultra_polynomial_manifest
    pub(crate) static ref ULTRA_POLYNOMIAL_MANIFEST: PolynomialManifest = {
        let manifest = vec![
            PolynomialDescriptor::new("W_1".to_string(), "w_1".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W1),
            PolynomialDescriptor::new("W_2".to_string(), "w_2".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W2),
            PolynomialDescriptor::new("W_3".to_string(), "w_3".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W3),
            PolynomialDescriptor::new("W_4".to_string(), "w_4".to_string(), true, PolynomialSource::Witness, PolynomialIndex::W4),
            PolynomialDescriptor::new("S".to_string(), "s".to_string(), true, PolynomialSource::Witness, PolynomialIndex::S),
            PolynomialDescriptor::new("Z_PERM".to_string(), "z_perm".to_string(), true, PolynomialSource::Witness, PolynomialIndex::Z),
            PolynomialDescriptor::new("Z_LOOKUP".to_string(), "z_lookup".to_string(), true, PolynomialSource::Witness, PolynomialIndex::ZLookup),
            PolynomialDescriptor::new("Q_1".to_string(), "q_1".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q1),
            PolynomialDescriptor::new("Q_2".to_string(), "q_2".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q2),
            PolynomialDescriptor::new("Q_3".to_string(), "q_3".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q3),
            PolynomialDescriptor::new("Q_4".to_string(), "q_4".to_string(), false, PolynomialSource::Selector, PolynomialIndex::Q4),
            PolynomialDescriptor::new("Q_M".to_string(), "q_m".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QM),
            PolynomialDescriptor::new("Q_C".to_string(), "q_c".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QC),
            PolynomialDescriptor::new("Q_ARITHMETIC".to_string(), "q_arith".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QArithmetic),
            PolynomialDescriptor::new("Q_SORT".to_string(), "q_sort".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QSort),
            PolynomialDescriptor::new("Q_ELLIPTIC".to_string(), "q_elliptic".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QElliptic),
            PolynomialDescriptor::new("Q_AUX".to_string(), "q_aux".to_string(), false, PolynomialSource::Selector, PolynomialIndex::QAux),
            PolynomialDescriptor::new("SIGMA_1".to_string(), "sigma_1".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma1),
            PolynomialDescriptor::new("SIGMA_2".to_string(), "sigma_2".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma2),
            PolynomialDescriptor::new("SIGMA_3".to_string(), "sigma_3".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma3),
            PolynomialDescriptor::new("SIGMA_4".to_string(), "sigma_4".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Sigma4),
            PolynomialDescriptor::new("TABLE_1".to_string(), "table_value_1".to_string(), true, PolynomialSource::Selector, PolynomialIndex::Table1),
            PolynomialDescriptor::new("TABLE_2".to_string(), "table_value_2".to_string(), true, PolynomialSource::Selector, PolynomialIndex::Table2),
            PolynomialDescriptor::new("TABLE_3".to_string(), "table_value_3".to_string(), true, PolynomialSource::Selector, PolynomialIndex::Table3),
            PolynomialDescriptor::new("TABLE_4".to_string(), "table_value_4".to_string(), true, PolynomialSource::Selector, PolynomialIndex::Table4),
            PolynomialDescriptor::new("TABLE_TYPE".to_string(), "table_type".to_string(), false, PolynomialSource::Selector, PolynomialIndex::TableType),
            PolynomialDescriptor::new("ID_1".to_string(), "id_1".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Id1),
            PolynomialDescriptor::new("ID_2".to_string(), "id_2".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Id2),
            PolynomialDescriptor::new("ID_3".to_string(), "id_3".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Id3),
            PolynomialDescriptor::new("ID_4".to_string(), "id_4".to_string(), false, PolynomialSource::Permutation, PolynomialIndex::Id4),

        ];
        PolynomialManifest { manifest }
    };

    pub(crate) static ref ULTRA_MANIFEST_SIZE: usize = ULTRA_POLYNOMIAL_MANIFEST.len();
}

impl PolynomialManifest {
    pub(crate) fn new() -> Self {
        PolynomialManifest {
            manifest: Vec::new(),
        }
    }

    pub(crate) fn new_from_type(type_: ComposerType) -> Self {
        match type_ {
            ComposerType::Standard => STANDARD_POLYNOMIAL_MANIFEST.clone(),
            ComposerType::Turbo => TURBO_POLYNOMIAL_MANIFEST.clone(),
            ComposerType::Plookup => ULTRA_POLYNOMIAL_MANIFEST.clone(),
            _ => unimplemented!("no standardhonk..."),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.manifest.len()
    }
    pub(crate) fn get(&self, index: PolynomialIndex) -> &PolynomialDescriptor {
        &self.manifest[index as usize]
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
    Witness,
    Selector,
    Permutation,
    Other,
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
