use ark_bn254::{Bn254, Fq12, G1Affine, G2Affine};
use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_ff::Field;
use ark_ff::One;

use super::io;

#[test]
fn read_transcript_loads_well_formed_srs() {
    let degree = 100000;
    let mut monomials: Vec<G1Affine> = vec![G1Affine::default(); degree + 2];
    let mut g2_x = G2Affine::default();

    io::read_transcript(&mut monomials, &mut g2_x, degree, "./src/srs_db/ignition").unwrap();

    assert_eq!(G1Affine::generator(), monomials[0]);

    let mut p: Vec<G1Affine> = Vec::with_capacity(2);
    let mut q: Vec<G2Affine> = Vec::with_capacity(2);
    p.push(monomials[1]);
    p.push(G1Affine::generator());
    p[0].y.neg_in_place();
    q.push(G2Affine::generator());
    q.push(g2_x);

    let res = Bn254::multi_pairing(&p, &q).0;

    assert_eq!(res, Fq12::one());

    for i in 0..degree {
        assert!(monomials[i].is_on_curve());
    }
}
