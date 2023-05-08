pub(crate) mod affine_element;
pub(crate) mod element;
pub(crate) mod wnaf;
#[cfg(test)]
mod wnaf_test;

use crate::field::Fq;
use crate::field::Fr;

pub struct GroupParams {
    pub USE_ENDOMORPHISM: bool,
    pub has_a: bool,
    pub one_x: Fq,
    pub one_y: Fq,
    pub a: Fq,
    pub b: Fq,
}

pub struct Group<CoordinateField, SubgroupField, GroupParams> {
    coordinate_field: CoordinateField,
    subgroup_field: SubgroupField,
    element: element::Element<CoordinateField, SubgroupField, GroupParams>,
    affine_element: affine_element::AffineElement<CoordinateField, SubgroupField, GroupParams>,
    fq: CoordinateField,
    fr: SubgroupField,
}

impl<CoordinateField, SubgroupField, GroupParams>
    Group<CoordinateField, SubgroupField, GroupParams>
{
    pub fn derive_generators<const N: usize>(
    ) -> [affine_element::AffineElement<CoordinateField, SubgroupField, GroupParams>; N]
    where
        CoordinateField: Default + Clone + PartialEq,
        SubgroupField: Default + Clone + PartialEq,
        GroupParams: Default + Clone,
    {
        let mut generators = [affine_element::AffineElement::default(); N];
        let mut count = 0;
        let mut seed = 0;

        while count < N {
            seed += 1;
            let candidate = affine_element::AffineElement::hash_to_curve(seed);
            if candidate.on_curve() && !candidate.is_point_at_infinity() {
                generators[count] = candidate;
                count += 1;
            }
        }

        generators
    }

    pub fn conditional_negate_affine(
        src: &affine_element::AffineElement<CoordinateField, SubgroupField, GroupParams>,
        dest: &mut affine_element::AffineElement<CoordinateField, SubgroupField, GroupParams>,
        predicate: u64,
    ) {
        // Implement conditional_negate_affine logic here
    }
}

impl<CoordinateField, SubgroupField, GroupParams> Group<CoordinateField, SubgroupField, GroupParams>
where
    CoordinateField: Default + Clone + PartialEq,
    SubgroupField: Default + Clone + PartialEq,
    GroupParams: Default + Clone,
{
    pub const one: element::Element<CoordinateField, SubgroupField, GroupParams> =
        element::Element {
            x: GroupParams::one_x,
            y: GroupParams::one_y,
            z: CoordinateField::one(),
        };

    pub const point_at_infinity: element::Element<CoordinateField, SubgroupField, GroupParams> =
        Self::one.set_infinity();

    pub const affine_one: affine_element::AffineElement<
        CoordinateField,
        SubgroupField,
        GroupParams,
    > = affine_element::AffineElement {
        x: GroupParams::one_x,
        y: GroupParams::one_y,
    };

    pub const affine_point_at_infinity: affine_element::AffineElement<
        CoordinateField,
        SubgroupField,
        GroupParams,
    > = Self::affine_one.set_infinity();

    pub const curve_a: CoordinateField = GroupParams::a;
    pub const curve_b: CoordinateField = GroupParams::b;
}
