use ark_bn254::{G2Projective, Fq, Fq6Config};
use ark_bn254::{Fq2};
use ark_ec::bn::BnConfig;
use ark_ff::{Field, One, Fp6Config};

const LOOP_LENGTH: usize = 64;
const NEG_Z_LOOP_LENGTH: usize = 62;
const PRECOMPUTED_COEFFICIENTS_LENGTH: usize = 87;

const LOOP_BITS: [u8; LOOP_LENGTH] = [
    1, 0, 1, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 1, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 1, 0,
    0, 3, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 3, 0, 3, 0, 0, 1, 0, 0, 0, 3, 0, 0, 3, 0, 1, 0, 1, 0, 0, 0,
];

const NEG_Z_LOOP_BITS: [u8; NEG_Z_LOOP_LENGTH] = [
    0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
    0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
];

lazy_static::lazy_static! {
    static ref TWO_INV: Fq = Fq::one().double().inverse().unwrap();
}

#[derive(Debug, Default, Clone, Copy)]
struct EllCoeffs<QuadFP: ark_ff::Field> {
    o: QuadFP,
    vw: QuadFP,
    vv: QuadFP,
}

#[derive(Debug, Clone)]
pub(crate) struct MillerLines {
    lines: [EllCoeffs<Fq2>; PRECOMPUTED_COEFFICIENTS_LENGTH],
}

impl Default for MillerLines {
    fn default() -> Self {
        Self {
            lines: [EllCoeffs::default(); PRECOMPUTED_COEFFICIENTS_LENGTH],
        }
    }
}

fn doubling_step_for_flipped_miller_loop(current: &mut G2Projective, ell: &mut EllCoeffs<Fq2>) {
    let mut a = mul_by_fq(current.x, *TWO_INV);
    a *= &current.y;

    let mut b = current.y.square();
    let mut c = current.z.square();
    let mut d = &c + &c;
    d += &c;
    let mut e = d * Fq2::twist_coeff_b();
    let mut f = &e + &e;
    f += &e;

    let mut g = &b + &f;
    g = mul_by_fq(g, *TWO_INV);
    let mut h = &current.y + &current.z;
    h = h.square();
    let mut i = &b + &c;
    h -= &i;
    i = &e - &b;
    let mut j = current.x.square();
    let mut ee = e.square();
    let mut k = &b - &f;
    current.x = a * k;

    k = &ee + &ee;
    k += &ee;

    let mut c = g.square();
    current.y = c - k;
    current.z = b * h;

    ell.o = Fq6Config::mul_fp2_by_nonresidue(i);

    ell.vw = -h;
    ell.vv = &j + &j;
    ell.vv += &j;
}

fn mixed_addition_step_for_flipped_miller_loop(
    base: &G2Projective,
    q: &mut G2Projective,
    line: &mut EllCoeffs<Fq2>,
) {
    let mut d = &base.x * &q.z;
    d = &q.x - &d;

    let mut e = &base.y * &q.z;
    e = &q.y - &e;

    let mut f = d.square();
    let mut g = e.square();
    let mut h = &d * &f;
    let mut i = &q.x * &f;

    let mut j = &q.z * &g;
    j += &h;
    j -= &i;
    j -= &i;

    q.x = &d * &j;
    i -= &j;
    i *= &e;
    let mut j = &q.y * &h;
    q.y = &i - &j;
    q.z *= &h;

    h = &e * &base.x;
    i = &d * &base.y;

    h -= &i;
    line.o = Fq6Config::mul_fp2_by_nonresidue(h);

    line.vv = -e;
    line.vw = d;
}

fn mul_by_q(a: &G2Projective) -> G2Projective {
    let t0 = a.x.frobenius_map();
    let t1 = a.y.frobenius_map();

    G2Projective {
        x: ark_bn254::Config::TWIST_MUL_BY_Q_X * &t0,
        y: ark_bn254::Config::TWIST_MUL_BY_Q_Y * &t1,
        z: a.z.frobenius_map(),
    }
}

fn mul_by_fq(f: Fq2, a: Fq) -> Fq2 {
    let mut c0 = f.c0;
    let mut c1 = f.c1;

    c0 *= a;
    c1 *= a;

    Fq2 { c0, c1 }
}

pub(crate) fn precompute_miller_lines(q: G2Projective, lines: &mut MillerLines) {
    let q_neg = G2Projective::new(q.x, -q.y, Fq2::one());
    let mut work_point = q.clone();

    let mut it: usize = 0;
    for i in 0..LOOP_LENGTH {
        doubling_step_for_flipped_miller_loop(&mut work_point, &mut lines.lines[it]);
        it += 1;
        match LOOP_BITS[i] {
            1 => {
                mixed_addition_step_for_flipped_miller_loop(
                    &q,
                    &mut work_point,
                    &mut lines.lines[it],
                );
                it += 1;
            }
            3 => {
                mixed_addition_step_for_flipped_miller_loop(
                    &q_neg,
                    &mut work_point,
                    &mut lines.lines[it],
                );
                it += 1;
            }
            _ => {}
        }
    }

    let mut q1 = mul_by_q(&q);
    let mut q2 = mul_by_q(&q1);
    q2 = -q2;
    mixed_addition_step_for_flipped_miller_loop(&q1, &mut work_point, &mut lines.lines[it]);
    it += 1;
    mixed_addition_step_for_flipped_miller_loop(&q2, &mut work_point, &mut lines.lines[it]);
}
