#[cfg(test)]
mod tests {
    use crate::polynomials::evaluation_domain::EvaluationDomain;
    use ark_bn254::Fr;
    use ark_ff::{Field, One};

    #[test]
    fn evaluation_domain_test() {
        let n = 256;
        let domain = EvaluationDomain::<Fr>::new(n, None);

        assert_eq!(domain.size, 256);
        assert_eq!(domain.log2_size, 8);
    }

    #[test]
    fn domain_roots_test() {
        let n = 256;
        let domain = EvaluationDomain::<Fr>::new(n, None);

        let result;
        let expected = Fr::one();
        result = domain.root.pow(&[n as u64]);

        assert!(result == expected);
    }

    #[test]
    fn evaluation_domain_roots_test() {
        let n = 16;
        let mut domain = EvaluationDomain::<Fr>::new(n, None);
        domain.compute_lookup_table();
        let root_table = domain.get_round_roots();
        let inverse_root_table = domain.get_inverse_round_roots();
        let roots = &root_table[root_table.len() - 1];
        let inverse_roots = &inverse_root_table[inverse_root_table.len() - 1];
        for i in 0..(n - 1) / 2 {
            assert_eq!(roots[i] * domain.root, roots[i + 1]);
            assert_eq!(inverse_roots[i] * domain.root_inverse, inverse_roots[i + 1]);
            assert_eq!(roots[i] * inverse_roots[i], Fr::one());
        }
    }
}
