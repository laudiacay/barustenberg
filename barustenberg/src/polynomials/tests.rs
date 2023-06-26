#[cfg(test)]
mod tests {
    use crate::polynomials::evaluation_domain::EvaluationDomain;
    use ark_bn254::Fr;

    #[test]
    fn evaluation_domain_test() {
        let n = 256;
        let domain = EvaluationDomain::<Fr>::new(n, None);

        assert_eq!(domain.size, 256);
        assert_eq!(domain.log2_size, 8);
    }
}
