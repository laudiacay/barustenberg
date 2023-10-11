use ark_bn254::{Bn254, Fq12, G1Affine, G2Affine};
use ark_ec::{AffineRepr, pairing::Pairing};
use ark_ff::{AdditiveGroup, One};

use super::io;

#[test]
fn read_transcript_loads_well_formed_srs() {
    let degree = 100000;
    let mut monomials: Vec<G1Affine> = vec![G1Affine::default(); degree + 2];
    let mut g2_x = G2Affine::default();

    io::read_transcript(&mut monomials, &mut g2_x, degree, "./src/srs_db/ignition").unwrap();

    assert_eq!(G1Affine::generator(), monomials[0]);

    let mut p: Vec<G1Affine> = vec![monomials[1], G1Affine::generator()];
    let q: Vec<G2Affine> = vec![G2Affine::generator(), g2_x];
    p[0].y.neg_in_place();

    let res = Bn254::multi_pairing(&p, &q).0;

    assert_eq!(res, Fq12::one());

    for mon in monomials.iter().take(degree) {
        assert!(mon.is_on_curve());
    }
}
