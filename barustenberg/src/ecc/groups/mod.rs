use self::{
    affine_element::{AffineElement, AffineElementImpl},
    element::{Element, ElementImpl},
};

use super::fields::field::Field;

pub(crate) mod affine_element;
pub(crate) mod element;
pub(crate) mod wnaf;
#[cfg(test)]
mod wnaf_test;

pub trait GroupParams<Fq: Field> {
    const USE_ENDOMORPHISM: bool;
    const has_a: bool;
    const one_x: Fq;
    const one_y: Fq;
    const a: Fq;
    const b: Fq;
}

pub trait Group<CoordinateField: Field, SubgroupField: Field, Params: GroupParams<CoordinateField>>
{
    // coordinate_field: CoordinateField,
    // subgroup_field: SubgroupField,
    // element: element::Element<CoordinateField, SubgroupField, GroupParams>,
    // fq: CoordinateField,
    // fr: SubgroupField,
    // Affine = AffineElementImpl<CoordinateField, SubgroupField, GroupParams>;
    fn derive_generators<const N: usize>(
    ) -> [dyn AffineElement<CoordinateField, SubgroupField, Params>; N] {
        let mut generators = [AffineElement::default(); N];
        let mut count = 0;
        let mut seed = 0;

        while count < N {
            seed += 1;
            let candidate = AffineElement::hash_to_curve(seed);
            if candidate.on_curve() && !candidate.is_point_at_infinity() {
                generators[count] = candidate;
                count += 1;
            }
        }

        generators
    }

    fn conditional_negate_affine(
        src: &dyn AffineElement<CoordinateField, SubgroupField, Params>,
        dest: &mut dyn AffineElement<CoordinateField, SubgroupField, Params>,
        predicate: u64,
    ) {
        // Implement conditional_negate_affine logic here
    }
    const one: Element<CoordinateField, SubgroupField, Params> = ElementImpl {
        x: GroupParams::one_x,
        y: GroupParams::one_y,
        z: CoordinateField::one(),
    };

    const point_at_infinity: Element<CoordinateField, SubgroupField, Params> =
        Self::one.set_infinity();

    const affine_one: AffineElementImpl = AffineElementImpl {
        x: GroupParams::one_x,
        y: GroupParams::one_y,
    };

    const affine_point_at_infinity: AffineElementImpl<CoordinateField, SubgroupField, Params> =
        Self::affine_one.set_infinity();

    const curve_a: CoordinateField = GroupParams::a;
    const curve_b: CoordinateField = GroupParams::b;
}

pub struct GroupImpl<Fq: Field, Fr: Field, Params: GroupParams<Fq>> {}
impl<Fq: Field, Fr: Field, Params: GroupParams<Fq>> Group<Fq, Fr, Params>
    for GroupImpl<Fq, Fr, Params>
{
}
