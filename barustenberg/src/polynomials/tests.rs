use ark_bn254::Fr;
use ark_ff::{Field, One, UniformRand, Zero};

use crate::polynomials::evaluation_domain::EvaluationDomain;
use crate::polynomials::polynomial_arithmetic;
use crate::polynomials::Polynomial;

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

    let expected = Fr::one();
    let result = domain.root.pow([n as u64]);

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
        poly.push(random_element);
        fft_transform.push(random_element);
    }

    let mut domain = EvaluationDomain::new(n, None);
    domain.compute_lookup_table();
    domain.fft_inplace(&mut fft_transform);

    let mut work_root = Fr::one();
    for transform in fft_transform.iter() {
        let expected = polynomial_arithmetic::evaluate(&poly, &work_root, n);
        assert_eq!(*transform, expected);
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
        poly.push(random_element);
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
    domain.fft_inplace(result.as_mut_slice());
    domain.ifft_inplace(result.as_mut_slice());

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

    let mut l_1_coefficients = Polynomial::new(2 * n);
    let mut scratch_memory = Polynomial::new(2 * n + 4);

    small_domain
        .compute_lagrange_polynomial_fft(&mut l_1_coefficients, &mid_domain)
        .unwrap();

    polynomial_arithmetic::copy_polynomial(&l_1_coefficients, &mut scratch_memory, 2 * n, 2 * n);

    mid_domain.coset_ifft_inplace(&mut l_1_coefficients.coefficients);

    let z = Fr::rand(&mut rand::thread_rng());
    let mut shifted_z = z * small_domain.root;
    shifted_z *= small_domain.root;

    let eval = polynomial_arithmetic::evaluate(
        &l_1_coefficients.coefficients,
        &shifted_z,
        small_domain.size,
    );
    small_domain.fft_inplace(&mut l_1_coefficients.coefficients);

    let temp_slice = scratch_memory.coefficients[..4].to_vec();
    scratch_memory[2 * n..2 * n + 4].copy_from_slice(&temp_slice);

    let l_n_minus_one_coefficients = &mut scratch_memory.coefficients[4..];
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
    let m = 4; // size of large_domain == M * n
    let mut small_domain = EvaluationDomain::new(n, None);
    let mut large_domain = EvaluationDomain::new(m * n, None);
    small_domain.compute_lookup_table();
    large_domain.compute_lookup_table();

    let mut l_1_coefficients = Polynomial::new(m * n);
    let mut scratch_memory = Polynomial::new(m * n + m * 2);

    // Compute FFT on target domain
    small_domain
        .compute_lagrange_polynomial_fft(&mut l_1_coefficients, &large_domain)
        .unwrap();

    // Copy L_1 FFT into scratch space and shift it to get FFT of L_{n-1}
    polynomial_arithmetic::copy_polynomial(&l_1_coefficients, &mut scratch_memory, m * n, m * n);

    // Manually 'shift' L_1 FFT in scratch memory by m*2
    let temp_slice = scratch_memory[..m * 2].to_vec();
    scratch_memory[m * n..m * n + m * 2].copy_from_slice(&temp_slice);

    let l_n_minus_one_coefficients = &mut scratch_memory.coefficients[m * 2..];

    // Recover monomial forms of L_1 and L_{n-1} (from manually shifted L_1 FFT)
    large_domain.coset_ifft_inplace(&mut l_1_coefficients.coefficients);
    large_domain.coset_ifft_inplace(l_n_minus_one_coefficients);

    // Compute shifted random eval point z*ω^2
    let z = Fr::rand(&mut rand::thread_rng());
    let shifted_z = z * small_domain.root * small_domain.root; // z*ω^2

    // Compute L_1(z_shifted) and L_{n-1}(z)
    let eval = polynomial_arithmetic::evaluate(
        &l_1_coefficients.coefficients,
        &shifted_z,
        small_domain.size,
    );
    let shifted_eval =
        polynomial_arithmetic::evaluate(l_n_minus_one_coefficients, &z, small_domain.size);

    // Check L_1(z_shifted) = L_{n-1}(z)
    assert_eq!(eval, shifted_eval);

    // Compute evaluation forms of L_1 and L_{n-1} and check that they have
    // a one in the right place and zeros elsewhere
    small_domain.fft_inplace(&mut l_1_coefficients.coefficients);
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

    small_domain
        .divide_by_pseudo_vanishing_polynomial(&mut [result.as_mut_slice()], &large_domain, 1)
        .unwrap();

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

    let f =
        polynomial_arithmetic::compute_kate_opening_coefficients_inplace(&mut w, &z, n).unwrap();

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

#[test]
fn test_barycentric_weight_evaluations() {
    let n = 16;

    let mut domain = EvaluationDomain::new(n, None);
    let mut rng = rand::thread_rng();

    let mut poly = vec![Fr::zero(); n];
    let mut barycentric_poly = vec![Fr::zero(); n];

    for i in 0..n / 2 {
        poly[i] = Fr::rand(&mut rng);
        barycentric_poly[i] = poly[i];
    }

    let evaluation_point = Fr::from(2_u64);

    let result = domain.compute_barycentric_evaluation(&barycentric_poly, n / 2, &evaluation_point);

    domain.compute_lookup_table();

    domain.ifft_inplace(&mut poly);

    let expected = polynomial_arithmetic::evaluate(&poly, &evaluation_point, n);

    assert_eq!(result, expected);
}

#[test]
fn test_divide_by_vanishing_polynomial() {
    let n = 16;

    let mut rng = rand::thread_rng();

    let mut a = vec![Fr::zero(); 2 * n];
    let mut b = vec![Fr::zero(); 2 * n];
    let mut c = vec![Fr::zero(); 2 * n];

    for i in 0..13 {
        a[i] = Fr::rand(&mut rng);
        b[i] = Fr::rand(&mut rng);
        c[i] = a[i] * b[i];
    }
    for i in 13..16 {
        a[i] = Fr::one();
        b[i] = Fr::from(2_u64);
        c[i] = Fr::from(3_u64);
    }

    let mut small_domain = EvaluationDomain::new(n, None);
    let mut large_domain = EvaluationDomain::new(2 * n, None);

    small_domain.compute_lookup_table();
    large_domain.compute_lookup_table();

    small_domain.ifft_inplace(&mut a);
    small_domain.ifft_inplace(&mut b);
    small_domain.ifft_inplace(&mut c);

    let z = Fr::rand(&mut rng);
    let a_eval = polynomial_arithmetic::evaluate(&a, &z, n);
    let b_eval = polynomial_arithmetic::evaluate(&b, &z, n);
    let c_eval = polynomial_arithmetic::evaluate(&c, &z, n);

    large_domain.coset_fft_inplace(&mut a);
    large_domain.coset_fft_inplace(&mut b);
    large_domain.coset_fft_inplace(&mut c);

    let mut r = vec![Fr::zero(); 2 * n];
    large_domain.mul(&a, &b, &mut r);
    large_domain.sub_inplace(&mut r, &c);

    let mut r_copy = r.clone();

    small_domain
        .divide_by_pseudo_vanishing_polynomial(&mut [r.as_mut_slice()], &large_domain, 3)
        .unwrap();
    large_domain.coset_ifft_inplace(&mut r);

    let r_eval = polynomial_arithmetic::evaluate(&r, &z, 2 * n);

    let z_h_eval = (z.pow(&[16]) - Fr::one())
        / ((z - small_domain.root_inverse)
            * (z - small_domain.root_inverse.square())
            * (z - small_domain.root_inverse * small_domain.root_inverse.square()));

    let lhs = a_eval * b_eval - c_eval;
    let rhs = r_eval * z_h_eval;
    assert_eq!(lhs, rhs);

    small_domain
        .divide_by_pseudo_vanishing_polynomial(&mut [r_copy.as_mut_slice()], &large_domain, 0)
        .unwrap();
    large_domain.coset_ifft_inplace(&mut r_copy);

    let r_eval = polynomial_arithmetic::evaluate(&r_copy, &z, 2 * n);
    let z_h_vanishing_eval = z.pow(&[16]) - Fr::one();
    let rhs = r_eval * z_h_vanishing_eval;
    assert_ne!(lhs, rhs);
}

#[test]
fn test_partial_fft_serial() {
    let n = 2;

    let mut rng = rand::thread_rng();

    let mut poly_eval = vec![Fr::zero(); 4 * n];
    let mut poly_partial_fft = vec![Fr::zero(); 4 * n];

    let mut large_domain = EvaluationDomain::new(4 * n, None);
    large_domain.compute_lookup_table();

    for i in 0..4 * n {
        poly_eval[i] = Fr::rand(&mut rng);
    }

    large_domain.partial_fft_serial(poly_eval.as_mut_slice(), &mut poly_partial_fft);

    let eval_point = Fr::rand(&mut rng);
    let expected = large_domain.compute_barycentric_evaluation(&poly_eval, 4 * n, &eval_point);

    let mut inner_poly_eval;
    let x_pow_4n = eval_point.pow([4 * n as u64]);
    let x_pow_4 = eval_point.pow([4]);
    let x_pow_3 = eval_point.pow([3]);
    let x_pow_2 = eval_point.pow([2]);
    let root = large_domain.root;
    let mut root_pow;
    let mut result = Fr::zero();

    for i in 0..n {
        inner_poly_eval = poly_partial_fft[i]
            + poly_partial_fft[n + i] * eval_point
            + poly_partial_fft[2 * n + i] * x_pow_2
            + poly_partial_fft[3 * n + i] * x_pow_3;
        root_pow = root.pow([4 * i as u64]);
        result += inner_poly_eval / (x_pow_4 - root_pow);
    }
    result *= x_pow_4n - Fr::one();
    result /= Fr::from(4 * n as u64);

    assert_eq!(result, expected);
}

#[test]
fn test_partial_fft_parallel() {
    let n = 2;

    let mut rng = rand::thread_rng();

    let mut poly_eval = vec![Fr::zero(); 4 * n];

    let mut large_domain = EvaluationDomain::new(4 * n, None);
    large_domain.compute_lookup_table();

    for i in 0..4 * n {
        poly_eval[i] = Fr::rand(&mut rng);
    }

    let eval_point = Fr::rand(&mut rng);
    let expected = large_domain.compute_barycentric_evaluation(&poly_eval, 4 * n, &eval_point);

    large_domain.partial_fft(poly_eval.as_mut_slice(), None, false);

    let mut inner_poly_eval;
    let x_pow_4n = eval_point.pow([4 * n as u64]);
    let x_pow_4 = eval_point.pow([4]);
    let x_pow_3 = eval_point.pow([3]);
    let x_pow_2 = eval_point.pow([2]);
    let root = large_domain.root;
    let mut root_pow;
    let mut result = Fr::zero();

    for i in 0..n {
        inner_poly_eval = poly_eval[i]
            + poly_eval[n + i] * eval_point
            + poly_eval[2 * n + i] * x_pow_2
            + poly_eval[3 * n + i] * x_pow_3;
        root_pow = root.pow(&[4 * i as u64]);
        result += inner_poly_eval / (x_pow_4 - root_pow);
    }
    result *= x_pow_4n - Fr::one();
    result /= Fr::from(4 * n as u64);

    assert_eq!(result, expected);
}

#[test]
fn test_partial_coset_fft_output() {
    let n = 64;

    let mut rng = rand::thread_rng();

    let mut poly_coset_fft = vec![Fr::zero(); 4 * n];
    let mut poly_coset_fft_copy = vec![Fr::zero(); 4 * n];

    let mut large_domain = EvaluationDomain::<Fr>::new(4 * n, None);
    large_domain.compute_lookup_table();
    let mut small_domain = EvaluationDomain::<Fr>::new(n, None);
    small_domain.compute_lookup_table();

    for i in 0..4 * n {
        poly_coset_fft[i] = Fr::rand(&mut rng);
        poly_coset_fft_copy[i] = poly_coset_fft[i];
    }

    large_domain.partial_fft(poly_coset_fft_copy.as_mut_slice(), None, false);

    let constant = large_domain.generator_inverse.pow(&[4]) * large_domain.four_inverse;
    large_domain.partial_fft(poly_coset_fft.as_mut_slice(), Some(constant), true);

    for i in 0..n {
        let current_root = small_domain.root_inverse.pow(&[i as u64]);
        let mut multiplicand = constant * current_root;
        for s in 0..4 {
            multiplicand *= large_domain.generator;
            assert_eq!(
                poly_coset_fft_copy[(3 - s) * n + i] * multiplicand,
                poly_coset_fft[(3 - s) * n + i]
            );
        }
    }
}

#[test]
fn test_partial_coset_fft() {
    let n = 64;

    let mut rng = rand::thread_rng();

    let mut poly_coset_fft = vec![Fr::zero(); 4 * n];

    let mut large_domain = EvaluationDomain::new(4 * n, None);
    large_domain.compute_lookup_table();
    let mut small_domain = EvaluationDomain::new(n, None);
    small_domain.compute_lookup_table();

    for i in 0..n {
        poly_coset_fft[i] = Fr::rand(&mut rng);
        poly_coset_fft[i + n] = Fr::zero();
        poly_coset_fft[i + 2 * n] = Fr::zero();
        poly_coset_fft[i + 3 * n] = Fr::zero();
    }

    large_domain.coset_fft_inplace(poly_coset_fft.as_mut_slice());

    let zeta = Fr::rand(&mut rng);
    let expected = polynomial_arithmetic::evaluate_from_fft(
        &poly_coset_fft,
        &large_domain,
        &zeta,
        &small_domain,
    );

    let constant = large_domain.generator_inverse.pow(&[4]) * large_domain.four_inverse;
    large_domain.partial_fft(poly_coset_fft.as_mut_slice(), Some(constant), true);

    let zeta_by_g_four = (zeta * large_domain.generator_inverse).pow(&[4]);
    let numerator = zeta_by_g_four.pow([n as u64]) - Fr::one();
    let mut result = Fr::zero();

    for i in 0..n {
        let current_root = small_domain.root_inverse.pow([i as u64]);
        let mut internal_term = Fr::zero();
        let mut multiplicand = Fr::one();
        let denominator = zeta_by_g_four * current_root - Fr::one();
        for s in 0..4 {
            internal_term += poly_coset_fft[s * n + i] * multiplicand;
            multiplicand *= zeta;
        }
        result += internal_term / denominator;
    }
    result *= numerator / Fr::from(n as u64);

    assert_eq!(result, expected);
}

#[test]
fn test_partial_coset_fft_evaluation() {
    let n = 64;

    let mut rng = rand::thread_rng();

    let mut poly_coset_fft = vec![Fr::zero(); 4 * n];

    let mut large_domain = EvaluationDomain::new(4 * n, None);
    large_domain.compute_lookup_table();
    let mut small_domain = EvaluationDomain::new(n, None);
    small_domain.compute_lookup_table();

    for i in 0..4 * n {
        poly_coset_fft[i] = Fr::rand(&mut rng);
    }

    let zeta = Fr::rand(&mut rng);
    let expected = large_domain.compute_barycentric_evaluation(
        &poly_coset_fft,
        4 * n,
        &(zeta * large_domain.generator_inverse),
    );

    let constant = large_domain.generator_inverse.pow([4]) * large_domain.four_inverse;
    large_domain.partial_fft(poly_coset_fft.as_mut_slice(), Some(constant), true);

    let zeta_by_g_four = (zeta * large_domain.generator_inverse).pow([4]);

    let mut result = Fr::zero();
    let mut multiplicand = Fr::one();
    for s in 0..4 {
        let local_eval = small_domain.compute_barycentric_evaluation(
            &poly_coset_fft[(s * n)..(s * n + n)],
            n,
            &zeta_by_g_four,
        );
        result += local_eval * multiplicand;
        multiplicand *= zeta;
    }

    assert_eq!(result, expected);
}

#[test]
fn test_linear_poly_product() {
    let n = 64;

    let mut rng = rand::thread_rng();

    let mut roots = vec![Fr::zero(); n];
    let mut expected = Fr::one();
    let z = Fr::rand(&mut rng);

    for i in 0..n {
        roots[i] = Fr::rand(&mut rng);
        expected *= z - roots[i];
    }

    let mut dest = vec![Fr::zero(); n + 1];
    polynomial_arithmetic::compute_linear_polynomial_product(&roots, &mut dest, n);
    let result = polynomial_arithmetic::evaluate(&dest, &z, n + 1);

    assert_eq!(result, expected);
}

#[test]
fn test_fft_linear_poly_product() {
    let n = 60;

    let mut rng = rand::thread_rng();

    let mut roots = vec![Fr::zero(); n];
    let mut expected = Fr::one();
    let z = Fr::rand(&mut rng);

    for i in 0..n {
        roots[i] = Fr::rand(&mut rng);
        expected *= z - roots[i];
    }

    let log2_n = n.next_power_of_two().trailing_zeros();
    let n = 1 << (log2_n + 1);

    let mut domain = EvaluationDomain::<Fr>::new(n, None);
    domain.compute_lookup_table();

    let mut dest = vec![Fr::zero(); n];
    domain.fft_linear_polynomial_product(&roots, &mut dest, n, false);
    let result = domain.compute_barycentric_evaluation(&dest, n, &z);

    let mut dest_coset = vec![Fr::zero(); n];
    let z_by_g = z * domain.generator_inverse;
    domain.fft_linear_polynomial_product(&roots, &mut dest_coset, n, true);
    let result1 = domain.compute_barycentric_evaluation(&dest_coset, n, &z_by_g);

    let mut coeffs = vec![Fr::zero(); n + 1];
    polynomial_arithmetic::compute_linear_polynomial_product(&roots, &mut coeffs, n);
    let result2 = polynomial_arithmetic::evaluate(&coeffs, &z, n + 1);

    assert_eq!(result, expected);
    assert_eq!(result1, expected);
    assert_eq!(result2, expected);
}

#[test]
fn test_compute_interpolation() {
    let n = 100;

    let mut rng = rand::thread_rng();

    let mut src = vec![Fr::zero(); n];
    let mut poly = vec![Fr::zero(); n];
    let mut x = vec![Fr::zero(); n];

    for i in 0..n {
        poly[i] = Fr::rand(&mut rng);
    }

    for i in 0..n {
        x[i] = Fr::rand(&mut rng);
        src[i] = polynomial_arithmetic::evaluate(&poly, &x[i], n);
    }

    let mut dest = vec![Fr::zero(); n];
    polynomial_arithmetic::compute_interpolation(&src, &mut dest, &mut x, n);

    for i in 0..n {
        assert_eq!(dest[i], poly[i]);
    }
}

#[test]
fn test_compute_efficient_interpolation() {
    let n = 250;

    let mut rng = rand::thread_rng();

    let mut src = vec![Fr::zero(); n];
    let mut poly = vec![Fr::zero(); n];
    let mut x = vec![Fr::zero(); n];

    for i in 0..n {
        poly[i] = Fr::rand(&mut rng);
    }

    for i in 0..n {
        x[i] = Fr::rand(&mut rng);
        src[i] = polynomial_arithmetic::evaluate(&poly, &x[i], n);
    }

    let mut dest = vec![Fr::zero(); n];
    polynomial_arithmetic::compute_efficient_interpolation(&src, &mut dest, &mut x, n).unwrap();

    for i in 0..n {
        assert_eq!(dest[i], poly[i]);
    }
}

#[test]
fn test_interpolation_constructor_single() {
    let root = vec![Fr::from(3)];
    let eval = vec![Fr::from(4)];

    let t = Polynomial::from_interpolations(&root, &eval).unwrap();

    assert_eq!(t.size(), 1);
    assert_eq!(t[0], eval[0]);
}

#[test]
fn test_interpolation_constructor() {
    let n = 32;

    let mut rng = rand::thread_rng();

    let mut roots = vec![Fr::zero(); n];
    let mut evaluations = vec![Fr::zero(); n];

    for i in 0..n {
        roots[i] = Fr::rand(&mut rng);
        evaluations[i] = Fr::rand(&mut rng);
    }

    let roots_copy = roots.clone();
    let evaluations_copy = evaluations.clone();

    let interpolated = Polynomial::from_interpolations(&roots, &evaluations).unwrap();

    assert_eq!(interpolated.size(), n);
    assert_eq!(roots, roots_copy);
    assert_eq!(evaluations, evaluations_copy);

    for i in 0..n {
        let eval = polynomial_arithmetic::evaluate(&interpolated.coefficients, &roots[i], n);
        assert_eq!(eval, evaluations[i]);
    }
}

#[test]
fn test_evaluate_mle() {
    fn test_case(n: usize) {
        let mut rng = rand::thread_rng();
        let m = n.next_power_of_two().trailing_zeros();
        assert_eq!(n, 1 << m);
        let mut poly = Polynomial::new(n);
        for i in 1..(n - 1) {
            poly[i] = Fr::rand(&mut rng);
        }
        poly[n - 1] = Fr::zero();

        assert!(poly[0].is_zero());

        // sample u = (u₀,…,uₘ₋₁)
        let mut u: Vec<Fr> = vec![Fr::zero(); m as usize];
        for l in 0..m {
            u[l as usize] = Fr::rand(&mut rng);
        }

        let mut lagrange_evals = vec![Fr::one(); n];
        for i in 0..n {
            let mut coef = Fr::one();
            for l in 0..m {
                let mask = 1 << l;
                if i & (mask as usize) == 0 {
                    coef *= Fr::one() - u[l as usize];
                } else {
                    coef *= u[l as usize];
                }
            }
            lagrange_evals[i] = coef;
        }

        // check eval by computing scalar product between
        // lagrange evaluations and coefficients
        let mut real_eval = Fr::zero();
        for i in 0..n {
            real_eval += poly[i] * lagrange_evals[i];
        }
        let computed_eval = poly.evaluate_mle(&u, false);
        assert_eq!(real_eval, computed_eval);

        // also check shifted eval
        let mut real_eval_shift = Fr::zero();
        for i in 1..n {
            real_eval_shift += poly[i] * lagrange_evals[i - 1];
        }
        let computed_eval_shift = poly.evaluate_mle(&u, true);
        assert_eq!(real_eval_shift, computed_eval_shift);
    }

    test_case(32);
    test_case(4);
    test_case(2);
}

#[test]
fn test_factor_roots() {
    fn test_case(num_zero_roots: usize, num_non_zero_roots: usize) {
        let num_roots = num_non_zero_roots + num_zero_roots;
        let n = 32;

        let mut poly = Polynomial::new(n);
        for i in num_zero_roots..n {
            poly[i] = Fr::rand(&mut rand::thread_rng());
        }

        // sample a root r, and compute p(r)/r^n for each non-zero root r
        let mut non_zero_roots: Vec<Fr> = vec![Fr::zero(); num_non_zero_roots];
        let mut non_zero_evaluations: Vec<Fr> = vec![Fr::zero(); num_non_zero_roots];
        for i in 0..num_non_zero_roots {
            let root = Fr::rand(&mut rand::thread_rng());
            non_zero_roots[i] = root;
            let root_pow = root.pow([num_zero_roots as u64]);
            non_zero_evaluations[i] = poly.evaluate(&root) / root_pow;
        }

        let mut roots: Vec<Fr> = vec![Fr::zero(); num_roots];
        for root in roots.iter_mut().take(num_zero_roots) {
            *root = Fr::zero();
        }

        roots[num_zero_roots..(num_non_zero_roots + num_zero_roots)]
            .copy_from_slice(&non_zero_roots[..num_non_zero_roots]);

        if num_non_zero_roots > 0 {
            let interpolated =
                Polynomial::from_interpolations(&non_zero_roots, &non_zero_evaluations).unwrap();
            assert_eq!(interpolated.size(), num_non_zero_roots);
            for i in 0..num_non_zero_roots {
                poly[num_zero_roots + i] -= interpolated[i];
            }
        }

        // Sanity check that all roots are actually roots
        for i in 0..num_roots {
            assert_eq!(poly.evaluate(&roots[i]), Fr::zero());
        }

        let mut quotient = poly.clone();
        quotient.factor_roots(&roots);

        // check that (t-r)q(t) == p(t)
        let t = Fr::rand(&mut rand::thread_rng());
        let roots_eval = polynomial_arithmetic::compute_linear_polynomial_product_evaluation(
            &roots, t, num_roots,
        );
        let q_t = quotient.evaluate(&t);
        let p_t = poly.evaluate(&t);
        assert_eq!(roots_eval * q_t, p_t);

        for i in (n - num_roots)..n {
            assert_eq!(quotient[i], Fr::zero());
        }
        if num_roots == 0 {
            assert_eq!(poly, quotient);
        }
        if num_roots == 1 {
            let mut quotient_single = poly.clone();
            quotient_single.factor_root(&roots[0]);
            assert_eq!(quotient_single, quotient);
        }
    }

    test_case(0, 0);
    test_case(0, 1);
    test_case(1, 0);
    test_case(1, 1);
    test_case(2, 0);
    test_case(0, 2);
    test_case(3, 6);
}

#[test]
fn test_default_construct_then_assign() {
    // construct an arbitrary but non-empty polynomial
    let num_coeffs = 64;
    let mut interesting_poly = Polynomial::new(num_coeffs);
    for coeff in &mut interesting_poly.coefficients {
        *coeff = Fr::rand(&mut rand::thread_rng());
    }

    // construct an empty poly via the default constructor
    let mut poly = Polynomial::new(0);

    assert!(poly.coefficients.is_empty());

    // fill the empty poly using the assignment operator
    poly = interesting_poly.clone();

    // coefficients and size should be equal in value
    for i in 0..num_coeffs {
        assert_eq!(poly[i], interesting_poly[i]);
    }
    assert_eq!(poly.size(), interesting_poly.size());
}
