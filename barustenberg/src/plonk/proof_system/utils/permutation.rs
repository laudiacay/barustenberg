use ark_ff::{FftField, Field};

use crate::{
    ecc::{
        conditionally_subtract_from_double_modulus, coset_generator,
        curves::external_coset_generator, tag_coset_generator,
    },
    numeric::bitop::Msb,
    polynomials::{evaluation_domain::EvaluationDomain, Polynomial},
    transcript::BarretenHasher,
};

#[derive(Debug, Clone)]
pub(crate) struct PermutationSubgroupElement {
    subgroup_index: u32,
    column_index: u8,
    is_public_input: bool,
    is_tag: bool,
}

pub(crate) fn compute_permutation_lagrange_base_single<H: BarretenHasher, Fr: Field + FftField>(
    output: &mut Polynomial<Fr>,
    permutation: &[u32],
    small_domain: &EvaluationDomain<Fr>,
) {
    let subgroup_elements: Vec<PermutationSubgroupElement> = permutation
        .iter()
        .map(|&permutation_element| {
            let index = permutation_element & 0xffffff;
            let column = permutation_element >> 30;
            PermutationSubgroupElement {
                subgroup_index: index,
                column_index: column as u8,
                is_public_input: false,
                is_tag: false,
            }
        })
        .collect();

    compute_permutation_lagrange_base_single_helper::<H, Fr>(
        output,
        &subgroup_elements,
        small_domain,
    );
}

pub(crate) fn compute_permutation_lagrange_base_single_helper<
    H: BarretenHasher,
    Fr: Field + FftField,
>(
    output: &mut Polynomial<Fr>,
    permutation: &[PermutationSubgroupElement],
    small_domain: &EvaluationDomain<Fr>,
) {
    if output.size() < permutation.len() {
        panic!("Permutation polynomial size is insufficient to store permutations.");
    }

    let roots = small_domain.get_round_roots()[small_domain.log2_size - 2];
    let root_size = small_domain.size >> 1;

    let log2_root_size = root_size.get_msb();

    for (i, perm_i) in permutation.iter().enumerate().take(small_domain.size) {
        let raw_idx = perm_i.subgroup_index as usize;
        let negative_idx = raw_idx >= root_size;
        let idx = raw_idx - ((negative_idx as usize) << log2_root_size);

        output.coefficients[i] =
            conditionally_subtract_from_double_modulus(&roots[idx], negative_idx as u64);

        if perm_i.is_public_input {
            output.coefficients[i] *= external_coset_generator::<Fr>();
        } else if perm_i.is_tag {
            output.coefficients[i] *= tag_coset_generator::<Fr>();
        } else {
            let column_index = perm_i.column_index;
            if column_index > 0 {
                output.coefficients[i] *= coset_generator::<Fr>(column_index - 1);
            }
        }
    }
}
