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
        let mut fft_transform_ = fft_transform_
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect::<Vec<_>>();

        for i in 0..n {
            fft_transform_[i / n_poly][i % n_poly] = fft_transform[i];
        }

        let mut domain = EvaluationDomain::new(n, None);
        domain.compute_lookup_table();
        domain.fft_inplace(&mut fft_transform);
        domain.fft_vec_inplace(&mut fft_transform_);

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
        let mut result = result
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect::<Vec<_>>();
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
        domain.fft_vec_inplace(&mut result);
        domain.ifft_vec_inplace(&mut result);

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
        let mut result = result
            .iter_mut()
            .map(|v| v.as_mut_slice())
            .collect::<Vec<_>>();
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
        domain.coset_fft_vec_inplace(&mut result);
        domain.coset_ifft_vec_inplace(&mut result);

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

    #[test]
    fn test_compute_lagrange_polynomial_fft() {
        let n = 256;
        let mut small_domain = EvaluationDomain::<Fr>::new(n, None);
        let mut mid_domain = EvaluationDomain::<Fr>::new(2 * n, None);
        small_domain.compute_lookup_table();
        mid_domain.compute_lookup_table();

        let mut l_1_coefficients = vec![Fr::zero(); 2 * n];
        let mut scratch_memory = vec![Fr::zero(); 2 * n + 4];

        small_domain.compute_lagrange_polynomial_fft(&mut l_1_coefficients, &mid_domain);

        polynomial_arithmetic::copy_polynomial(
            &l_1_coefficients,
            &mut scratch_memory,
            2 * n,
            2 * n,
        );

        mid_domain.coset_ifft_inplace(&mut l_1_coefficients);

        let z = Fr::rand(&mut rand::thread_rng());
        let mut shifted_z = z * small_domain.root;
        shifted_z *= small_domain.root;

        let eval =
            polynomial_arithmetic::evaluate(&l_1_coefficients, &shifted_z, small_domain.size);
        small_domain.fft_inplace(&mut l_1_coefficients);

        let temp_slice = scratch_memory[..4].to_vec();
        scratch_memory[2 * n..2 * n + 4].copy_from_slice(&temp_slice);

        let l_n_minus_one_coefficients = &mut scratch_memory[4..];
        mid_domain.coset_ifft_inplace(l_n_minus_one_coefficients);

        let shifted_eval =
            polynomial_arithmetic::evaluate(l_n_minus_one_coefficients, &z, small_domain.size);
        assert_eq!(eval, shifted_eval);

        small_domain.fft_inplace(l_n_minus_one_coefficients);

        assert_eq!(l_1_coefficients[0], Fr::one());
        for i in 1..n {
            assert_eq!(l_1_coefficients[i], Fr::zero());
        }

        assert_eq!(l_n_minus_one_coefficients[n - 2], Fr::one());
        for i in 0..n {
            if i == n - 2 {
                continue;
            }
            assert_eq!(l_n_minus_one_coefficients[i], Fr::zero());
        }
    }

    #[test]
    fn test_compute_lagrange_polynomial_fft_large_domain() {
        let n = 256; // size of small_domain
        let M = 4; // size of large_domain == M * n
        let mut small_domain = EvaluationDomain::new(n, None);
        let mut large_domain = EvaluationDomain::new(M * n, None);
        small_domain.compute_lookup_table();
        large_domain.compute_lookup_table();

        let mut l_1_coefficients = vec![Fr::zero(); M * n];
        let mut scratch_memory = vec![Fr::zero(); M * n + M * 2];

        // Compute FFT on target domain
        small_domain.compute_lagrange_polynomial_fft(&mut l_1_coefficients, &large_domain);

        // Copy L_1 FFT into scratch space and shift it to get FFT of L_{n-1}
        polynomial_arithmetic::copy_polynomial(
            &l_1_coefficients,
            &mut scratch_memory,
            M * n,
            M * n,
        );

        // Manually 'shift' L_1 FFT in scratch memory by m*2
        let temp_slice = scratch_memory[..M * 2].to_vec();
        scratch_memory[M * n..M * n + M * 2].copy_from_slice(&temp_slice);

        let l_n_minus_one_coefficients = &mut scratch_memory[M * 2..];

        // Recover monomial forms of L_1 and L_{n-1} (from manually shifted L_1 FFT)
        large_domain.coset_ifft_inplace(&mut l_1_coefficients);
        large_domain.coset_ifft_inplace(l_n_minus_one_coefficients);

        // Compute shifted random eval point z*ω^2
        let z = Fr::rand(&mut rand::thread_rng());
        let shifted_z = z * small_domain.root * small_domain.root; // z*ω^2

        // Compute L_1(z_shifted) and L_{n-1}(z)
        let eval =
            polynomial_arithmetic::evaluate(&l_1_coefficients, &shifted_z, small_domain.size);
        let shifted_eval =
            polynomial_arithmetic::evaluate(l_n_minus_one_coefficients, &z, small_domain.size);

        // Check L_1(z_shifted) = L_{n-1}(z)
        assert_eq!(eval, shifted_eval);

        // Compute evaluation forms of L_1 and L_{n-1} and check that they have
        // a one in the right place and zeros elsewhere
        small_domain.fft_inplace(&mut l_1_coefficients);
        small_domain.fft_inplace(l_n_minus_one_coefficients);

        assert_eq!(l_1_coefficients[0], Fr::one());

        for i in 1..n {
            assert_eq!(l_1_coefficients[i], Fr::zero());
        }

        assert_eq!(l_n_minus_one_coefficients[n - 2], Fr::one());

        for i in 0..n {
            if i == (n - 2) {
                continue;
            }
            assert_eq!(l_n_minus_one_coefficients[i], Fr::zero());
        }
    }

    #[test]
    fn test_divide_by_pseudo_vanishing_polynomial() {
        let n = 256;
        let n_large = 4 * n;
        let mut a = vec![Fr::zero(); 4 * n];
        let mut b = vec![Fr::zero(); 4 * n];
        let mut c = vec![Fr::zero(); 4 * n];
        let mut rng = rand::thread_rng();

        for i in 0..n {
            a[i] = Fr::rand(&mut rng);
            b[i] = Fr::rand(&mut rng);
            c[i] = a[i] * b[i];
            c[i] = -c[i];
        }

        let mut small_domain = EvaluationDomain::new(n, None);
        let mut large_domain = EvaluationDomain::new(n_large, None);
        small_domain.compute_lookup_table();
        large_domain.compute_lookup_table();

        small_domain.ifft_inplace(&mut a);
        small_domain.ifft_inplace(&mut b);
        small_domain.ifft_inplace(&mut c);

        large_domain.coset_fft_inplace(&mut a);
        large_domain.coset_fft_inplace(&mut b);
        large_domain.coset_fft_inplace(&mut c);

        let mut result = vec![Fr::default(); n_large];
        for i in 0..large_domain.size {
            result[i] = a[i] * b[i] + c[i];
        }

        small_domain.divide_by_pseudo_vanishing_polynomial(
            &mut [&mut result[..]],
            &large_domain,
            1,
        );

        large_domain.coset_ifft_inplace(&mut result);

        for i in n + 1..large_domain.size {
            assert_eq!(result[i], Fr::zero());
        }
    }

    #[test]
    fn test_compute_kate_opening_coefficients() {
        let n = 256;
        let mut coeffs = vec![Fr::zero(); 2 * n];
        let mut w = vec![Fr::zero(); 2 * n];
        let mut rng = rand::thread_rng();

        for i in 0..n {
            coeffs[i] = Fr::rand(&mut rng);
            w[i] = coeffs[i];
        }

        let z = Fr::rand(&mut rng);

        let f = polynomial_arithmetic::compute_kate_opening_coefficients_inplace(&mut w, &z, n)
            .unwrap();

        let mut multiplicand = vec![Fr::zero(); 2 * n];
        multiplicand[0] = -z;
        multiplicand[1] = Fr::one();

        coeffs[0] -= f;

        let mut domain = EvaluationDomain::new(2 * n, None);
        domain.compute_lookup_table();
        domain.coset_fft_inplace(&mut coeffs);
        domain.coset_fft_inplace(&mut w);
        domain.coset_fft_inplace(&mut multiplicand);

        for i in 0..domain.size {
            let result = w[i] * multiplicand[i];
            assert_eq!(result, coeffs[i]);
        }
    }

    #[test]
    fn test_get_lagrange_evaluations() {
        let n = 16;

        let mut domain = EvaluationDomain::new(n, None);
        domain.compute_lookup_table();
        let mut rng = rand::thread_rng();
        let z = Fr::rand(&mut rng);

        let evals = domain.get_lagrange_evaluations(&z, Some(1));

        let mut vanishing_poly = vec![Fr::zero(); 2 * n];
        let mut l_1_poly = vec![Fr::zero(); n];
        let mut l_n_minus_1_poly = vec![Fr::zero(); n];

        l_1_poly[0] = Fr::one();
        l_n_minus_1_poly[n - 2] = Fr::one();

        let n_mont = Fr::from(n as u64);
        vanishing_poly[n - 1] = n_mont * domain.root;

        domain.ifft_inplace(&mut l_1_poly);
        domain.ifft_inplace(&mut l_n_minus_1_poly);
        domain.ifft_inplace(&mut vanishing_poly);

        let l_1_expected = polynomial_arithmetic::evaluate(&l_1_poly, &z, n);
        let l_n_minus_1_expected = polynomial_arithmetic::evaluate(&l_n_minus_1_poly, &z, n);
        let vanishing_poly_expected = polynomial_arithmetic::evaluate(&vanishing_poly, &z, n);

        assert_eq!(evals.l_start, l_1_expected);
        assert_eq!(evals.l_end, l_n_minus_1_expected);
        assert_eq!(evals.vanishing_poly, vanishing_poly_expected);
    }
}
