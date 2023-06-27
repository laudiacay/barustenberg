#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{Field, One, UniformRand, Zero};

    use crate::polynomials::evaluation_domain::EvaluationDomain;
    use crate::polynomials::polynomial_arithmetic;

    #[test]
    fn test_evaluation_domain() {
        let n = 256;
        let domain = EvaluationDomain::<Fr>::new(n, None);

        assert_eq!(domain.size, 256);
        assert_eq!(domain.log2_size, 8);
    }

    #[test]
    fn test_domain_roots() {
        let n = 256;
        let domain = EvaluationDomain::<Fr>::new(n, None);

        let result;
        let expected = Fr::one();
        result = domain.root.pow(&[n as u64]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluation_domain_roots() {
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
    fn test_fft_with_small_degree() {
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
    fn test_split_polynomial_fft() {
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

    #[test]
    fn test_basic_fft() {
        let n = 1 << 14;
        let mut rng = rand::thread_rng();
        let mut result = vec![Fr::default(); n];
        let mut expected = vec![Fr::default(); n];

        for i in 0..n {
            let random_element = Fr::rand(&mut rng);
            result[i] = random_element;
            expected[i] = random_element;
        }

        let mut domain = EvaluationDomain::<Fr>::new(n, None);
        domain.compute_lookup_table();
        domain.fft_inplace(&mut result[..]);
        domain.ifft_inplace(&mut result[..]);

        for i in 0..n {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_fft_ifft_consistency() {
        let n = 256;
        let mut rng = rand::thread_rng();
        let mut result = vec![Fr::default(); n];
        let mut expected = vec![Fr::default(); n];

        for i in 0..n {
            let random_element = Fr::rand(&mut rng);
            result[i] = random_element;
            expected[i] = random_element;
        }

        let mut domain = EvaluationDomain::<Fr>::new(n, None);
        domain.compute_lookup_table();
        domain.fft_inplace(&mut result);
        domain.ifft_inplace(&mut result);

        for i in 0..n {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_split_polynomial_fft_ifft_consistency() {
        let n = 256;
        let num_poly = 4;
        let mut rng = rand::thread_rng();
        let mut result = vec![vec![Fr::default(); n]; num_poly];
        let mut expected = vec![vec![Fr::default(); n]; num_poly];

        for j in 0..num_poly {
            for i in 0..n {
                let random_element = Fr::rand(&mut rng);
                result[j][i] = random_element;
                expected[j][i] = random_element;
            }
        }

        let mut domain = EvaluationDomain::<Fr>::new(num_poly * n, None);
        domain.compute_lookup_table();
        domain.fft_vec_inplace(
            &mut result
                .iter_mut()
                .map(|v| v.as_mut_slice())
                .collect::<Vec<_>>(),
        );
        domain.ifft_vec_inplace(
            &mut result
                .iter_mut()
                .map(|v| v.as_mut_slice())
                .collect::<Vec<_>>(),
        );

        for j in 0..num_poly {
            for i in 0..n {
                assert_eq!(result[j][i], expected[j][i]);
            }
        }
    }

    #[test]
    fn fft_coset_ifft_consistency() {
        let n = 256;
        let mut rng = rand::thread_rng();
        let mut result = vec![Fr::default(); n];
        let mut expected = vec![Fr::default(); n];

        for i in 0..n {
            let random_element = Fr::rand(&mut rng);
            result[i] = random_element;
            expected[i] = random_element;
        }

        let mut domain = EvaluationDomain::<Fr>::new(n, None);
        domain.compute_lookup_table();

        let t0 = domain.generator * domain.generator_inverse;
        assert_eq!(t0, Fr::one());

        domain.coset_fft_inplace(&mut result);
        domain.coset_ifft_inplace(&mut result);

        for i in 0..n {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_split_polynomial_fft_coset_ifft_consistency() {
        let n = 256;
        let num_poly = 4;
        let mut rng = rand::thread_rng();
        let mut result = vec![vec![Fr::default(); n]; num_poly];
        let mut expected = vec![vec![Fr::default(); n]; num_poly];

        for j in 0..num_poly {
            for i in 0..n {
                let random_element = Fr::rand(&mut rng);
                result[j][i] = random_element;
                expected[j][i] = random_element;
            }
        }

        let mut domain = EvaluationDomain::<Fr>::new(num_poly * n, None);
        domain.compute_lookup_table();
        domain.coset_fft_vec_inplace(
            &mut result
                .iter_mut()
                .map(|v| v.as_mut_slice())
                .collect::<Vec<_>>(),
        );
        domain.coset_ifft_vec_inplace(
            &mut result
                .iter_mut()
                .map(|v| v.as_mut_slice())
                .collect::<Vec<_>>(),
        );

        for j in 0..num_poly {
            for i in 0..n {
                assert_eq!(result[j][i], expected[j][i]);
            }
        }
    }

    #[test]
    fn test_fft_coset_ifft_cross_consistency() {
        let n = 2;
        let mut rng = rand::thread_rng();
        let mut expected = vec![Fr::default(); n];
        let mut poly_a = vec![Fr::default(); 4 * n];
        let mut poly_b = vec![Fr::default(); 4 * n];
        let mut poly_c = vec![Fr::default(); 4 * n];

        for i in 0..n {
            let random_element = Fr::rand(&mut rng);
            poly_a[i] = random_element;
            poly_b[i] = random_element;
            poly_c[i] = random_element;
            expected[i] = poly_a[i] + poly_c[i] + poly_b[i];
        }

        for i in n..4 * n {
            poly_a[i] = Fr::zero();
            poly_b[i] = Fr::zero();
            poly_c[i] = Fr::zero();
        }

        let mut small_domain = EvaluationDomain::<Fr>::new(n, None);
        let mut mid_domain = EvaluationDomain::<Fr>::new(2 * n, None);
        let mut large_domain = EvaluationDomain::<Fr>::new(4 * n, None);
        small_domain.compute_lookup_table();
        mid_domain.compute_lookup_table();
        large_domain.compute_lookup_table();
        small_domain.coset_fft_inplace(&mut poly_a);
        mid_domain.coset_fft_inplace(&mut poly_b);
        large_domain.coset_fft_inplace(&mut poly_c);

        for i in 0..n {
            poly_a[i] = poly_a[i] + poly_c[4 * i] + poly_b[2 * i];
        }

        small_domain.coset_ifft_inplace(&mut poly_a);

        for i in 0..n {
            assert_eq!(poly_a[i], expected[i]);
        }
    }
}
