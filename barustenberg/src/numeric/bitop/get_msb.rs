// from http://supertech.csail.mit.edu/papers/debruijn.pdf
// from
/*
constexpr uint8_t MultiplyDeBruijnBitPosition[32] = {
    0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31
}; */

lazy_static::lazy_static! {
    static ref MULTIPLY_DE_BRUIJN_BIT_POSITION: [u8; 32] = [
        0, 9,  1,  10, 13, 21, 2,  29, 11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7,  19, 27, 23, 6,  26, 5,  4, 31
    ];

    static ref DE_BRUIJN_SEQUENCE: [u8; 64] = [
        0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61,
        54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11, 4,  62,
        46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63 ];
}

fn get_msb32(in_val: u32) -> u32 {
    let v = in_val | (in_val >> 1);
    let v = v | (v >> 2);
    let v = v | (v >> 4);
    let v = v | (v >> 8);
    let v = v | (v >> 16);

    MULTIPLY_DE_BRUIJN_BIT_POSITION[((v.wrapping_mul(0x07C4ACDDu32)) >> 27) as usize] as u32
}

pub(crate) fn get_msb64(in_val: u64) -> u64 {
    let t = in_val | (in_val >> 1);
    let t = t | (t >> 2);
    let t = t | (t >> 4);
    let t = t | (t >> 8);
    let t = t | (t >> 16);
    let t = t | (t >> 32);

    DE_BRUIJN_SEQUENCE[((t.wrapping_mul(0x03F79D71B4CB0A89u64)) >> 58) as usize] as u64
}

pub(crate) trait Msb {
    fn get_msb(self) -> Self;
}

impl Msb for u32 {
    fn get_msb(self) -> Self {
        get_msb32(self)
    }
}

impl Msb for u64 {
    fn get_msb(self) -> Self {
        get_msb64(self)
    }
}

impl Msb for i32 {
    fn get_msb(self) -> Self {
        get_msb32(self as u32) as i32
    }
}

impl Msb for i64 {
    fn get_msb(self) -> Self {
        get_msb64(self as u64) as i64
    }
}

impl Msb for usize {
    fn get_msb(self) -> Self {
        get_msb64(self as u64) as usize
    }
}

// add test
#[cfg(test)]
mod test {
    use super::*;

    /*TEST(bitop, get_msb_uint64_0_value)
    {
        uint64_t a = 0b00000000000000000000000000000000;
        EXPECT_EQ(numeric::get_msb(a), 0U);
    } */
    #[test]
    fn get_msb_u64_0_value() {
        let a: u64 = 0b00000000000000000000000000000000;
        assert_eq!(a.get_msb(), 0u64);
    }

    /*
        TEST(bitop, get_msb_uint32_0)
    {
        uint32_t a = 0b00000000000000000000000000000001;
        EXPECT_EQ(numeric::get_msb(a), 0U);
    } */
    #[test]
    fn get_msb_u32_0_value() {
        let a: u32 = 0b00000000000000000000000000000001;
        assert_eq!(a.get_msb(), 0u32);
    }

    /*
        TEST(bitop, get_msb_uint32_31)
    {
        uint32_t a = 0b10000000000000000000000000000001;
        EXPECT_EQ(numeric::get_msb(a), 31U);
    } */
    #[test]
    fn get_msb_uint32_31() {
        let a: u32 = 0b10000000000000000000000000000001;
        assert_eq!(a.get_msb(), 31u32);
    }
    /*
        TEST(bitop, get_msb_uint64_63)
    {
        uint64_t a = 0b1000000000000000000000000000000100000000000000000000000000000000;
        EXPECT_EQ(numeric::get_msb(a), 63U);
    }
         */
    #[test]
    fn get_msb_u64_63() {
        let a: u64 = 0b1000000000000000000000000000000100000000000000000000000000000000;
        assert_eq!(a.get_msb(), 63u64);
    }

    /*TEST(bitop, get_msb_size_t_7)
    {
        size_t a = 0x80;
        auto r = numeric::get_msb(a);
        EXPECT_EQ(r, 7U);
    } */
    #[test]
    fn get_msb_size_t_7() {
        let a: usize = 0x80;
        assert_eq!(a.get_msb(), 7usize);
    }
}
