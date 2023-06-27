#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{Field, One, UniformRand};

    use crate::polynomials::evaluation_domain::EvaluationDomain;
    use crate::polynomials::polynomial_arithmetic;

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

        assert_eq!(result, expected);
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

    #[test]
    fn fft_with_small_degree_test() {
        let n = 16;
        let mut rng = rand::thread_rng();
        let mut fft_transform = Vec::with_capacity(n);
        let mut poly = Vec::with_capacity(n);

        for _ in 0..n {
            let random_element = Fr::rand(&mut rng);
            poly.push(random_element.clone());
            fft_transform.push(random_element);
        }

        let mut domain = EvaluationDomain::new(n, None);
        domain.compute_lookup_table();
        domain.fft_inplace(&mut fft_transform);

        let mut work_root = Fr::one();
        for i in 0..n {
            let expected = polynomial_arithmetic::evaluate(&poly, &work_root, n);
            assert_eq!(fft_transform[i], expected);
            work_root *= domain.root;
        }
    }

    #[test]
    fn split_polynomial_fft_test() {
        let n = 256;
        let mut rng = rand::thread_rng();
        let mut fft_transform = Vec::with_capacity(n);
        let mut poly = Vec::with_capacity(n);

        for _ in 0..n {
            let random_element = Fr::rand(&mut rng);
            poly.push(random_element.clone());
            fft_transform.push(random_element);
        }

        let num_poly = 4;
        let n_poly = n / num_poly;
        let mut fft_transform_ = vec![vec![Fr::default(); n_poly]; num_poly];
        for i in 0..n {
            fft_transform_[i / n_poly][i % n_poly] = fft_transform[i];
        }

        let mut domain = EvaluationDomain::new(n, None);
        domain.compute_lookup_table();
        domain.fft_inplace(&mut fft_transform);
        domain.fft_vec_inplace(
            &mut fft_transform_
                .iter_mut()
                .map(|v| v.as_mut_slice())
                .collect::<Vec<_>>(),
        );

        let mut work_root = Fr::one();
        for i in 0..n {
            let expected = polynomial_arithmetic::evaluate(&poly, &work_root, n);
            assert_eq!(fft_transform[i], expected);
            assert_eq!(fft_transform_[i / n_poly][i % n_poly], fft_transform[i]);
            work_root *= domain.root;
        }
    }
}
