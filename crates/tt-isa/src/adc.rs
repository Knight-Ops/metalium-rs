//! Checked XY and ZW address-counter operations. Blackhole Z/W live counters
//! and cursor anchors are thirteen bits; unpacker input addressing sees only
//! the low eight bits. Counters are private to the issuing Tensix thread. The
//! unsupported cross-thread override is deliberately absent from this API.
use crate::isa::{generated::encode, EncodeError, Instruction};

/// One or more counter owners. A zero target mask cannot be constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Targets(u8);

impl Targets {
    pub const UNPACKER0: Self = Self(1);
    pub const UNPACKER1: Self = Self(2);
    pub const PACKERS: Self = Self(4);
    pub const ALL: Self = Self(7);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Increments for the input (0) and output/end (1) channels. Every field
/// must fit three bits; encode checks this rather than truncating it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Xy {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

/// Coordinates whose cursor and live counter an ADDRCRXY updates. Unselected
/// coordinates retain both values, even if their increment is nonzero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coordinates {
    pub x0: bool,
    pub y0: bool,
    pub x1: bool,
    pub y1: bool,
}

impl Coordinates {
    pub const ALL: Self = Self {
        x0: true,
        y0: true,
        x1: true,
        y1: true,
    };
    pub const ROWS: Self = Self {
        x0: false,
        y0: true,
        x1: false,
        y1: true,
    };
}

/// Increment live counters, leaving their cursor (`_Cr`) anchors unchanged.
pub const fn increment(targets: Targets, by: Xy) -> Result<Instruction, EncodeError> {
    encode::Incadcxy::ZERO
        .u0((targets.0 & 1 != 0) as u32)
        .u1((targets.0 & 2 != 0) as u32)
        .pk((targets.0 & 4 != 0) as u32)
        .x0_inc(by.x0)
        .y0_inc(by.y0)
        .x1_inc(by.x1)
        .y1_inc(by.y1)
        .encode()
}

/// Advance selected cursor anchors and restore their live counters to those
/// anchors. A zero increment restores a cursor without advancing it. An empty
/// coordinate mask is a silicon no-op, but the pinned simulator refuses it
/// (`ttsim-divergence.md`, row 74); portable programs select at least one.
/// Drain affected unpack/pack work before changing counters used by it.
pub const fn advance_cursor(
    targets: Targets,
    coordinates: Coordinates,
    by: Xy,
) -> Result<Instruction, EncodeError> {
    encode::Addrcrxy::ZERO
        .u0((targets.0 & 1 != 0) as u32)
        .u1((targets.0 & 2 != 0) as u32)
        .pk((targets.0 & 4 != 0) as u32)
        .x0(coordinates.x0 as u32)
        .y0(coordinates.y0 as u32)
        .x1(coordinates.x1 as u32)
        .y1(coordinates.y1 as u32)
        .x0_inc(by.x0)
        .y0_inc(by.y0)
        .x1_inc(by.x1)
        .y1_inc(by.y1)
        .encode()
}

/// Increments for the input (0) and output/end (1) channels. Every field
/// must fit three bits; encode checks this rather than truncating it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Zw {
    pub z0: u32,
    pub w0: u32,
    pub z1: u32,
    pub w1: u32,
}

/// Coordinates whose cursor and live counter an ADDRCRZW updates. Unselected
/// coordinates retain both values, even if their increment is nonzero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZwCoordinates {
    pub z0: bool,
    pub w0: bool,
    pub z1: bool,
    pub w1: bool,
}

impl ZwCoordinates {
    pub const ALL: Self = Self {
        z0: true,
        w0: true,
        z1: true,
        w1: true,
    };
    pub const ROWS: Self = Self {
        z0: false,
        w0: true,
        z1: false,
        w1: true,
    };
}

/// Increment live counters, leaving their cursor (`_Cr`) anchors unchanged.
pub const fn increment_zw(targets: Targets, by: Zw) -> Result<Instruction, EncodeError> {
    encode::Incadczw::ZERO
        .u0((targets.0 & 1 != 0) as u32)
        .u1((targets.0 & 2 != 0) as u32)
        .pk((targets.0 & 4 != 0) as u32)
        .z0_inc(by.z0)
        .w0_inc(by.w0)
        .z1_inc(by.z1)
        .w1_inc(by.w1)
        .encode()
}

/// Advance selected cursor anchors and restore their live counters to those
/// anchors. A zero increment restores a cursor without advancing it. An empty
/// coordinate mask is a silicon no-op but the pinned simulator refuses it
/// (`ttsim-divergence.md`). Drain affected unpack/pack work before changing
/// counters used by it.
pub const fn advance_cursor_zw(
    targets: Targets,
    coordinates: ZwCoordinates,
    by: Zw,
) -> Result<Instruction, EncodeError> {
    encode::Addrcrzw::ZERO
        .u0((targets.0 & 1 != 0) as u32)
        .u1((targets.0 & 2 != 0) as u32)
        .pk((targets.0 & 4 != 0) as u32)
        .z0(coordinates.z0 as u32)
        .w0(coordinates.w0 as u32)
        .z1(coordinates.z1 as u32)
        .w1(coordinates.w1 as u32)
        .z0_inc(by.z0)
        .w0_inc(by.w0)
        .z1_inc(by.z1)
        .w1_inc(by.w1)
        .encode()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::generated::defs;

    #[test]
    fn fields_are_checked_and_cross_thread_override_is_absent() {
        let by = Xy {
            x0: 7,
            y0: 6,
            x1: 5,
            y1: 4,
        };
        for bits in 1..8 {
            let targets = Targets(bits);
            for i in [
                increment(targets, by).unwrap(),
                advance_cursor(targets, Coordinates::ALL, by).unwrap(),
            ] {
                let field = |name| {
                    i.def()
                        .fields()
                        .iter()
                        .find(|f| f.name() == name)
                        .unwrap()
                        .extract(i.word())
                };
                assert_eq!(field("U0"), u32::from(bits & 1 != 0));
                assert_eq!(field("U1"), u32::from(bits & 2 != 0));
                assert_eq!(field("PK"), u32::from(bits & 4 != 0));
                assert_eq!(field("ThreadOverride"), 0);
                assert_eq!(field("X0Inc"), 7);
                assert_eq!(field("Y0Inc"), 6);
                assert_eq!(field("X1Inc"), 5);
                assert_eq!(field("Y1Inc"), 4);
            }
        }
        for by in [
            Xy {
                x0: 8,
                ..Xy::default()
            },
            Xy {
                y0: 8,
                ..Xy::default()
            },
            Xy {
                x1: 8,
                ..Xy::default()
            },
            Xy {
                y1: 8,
                ..Xy::default()
            },
        ] {
            assert!(increment(Targets::ALL, by).is_err());
            assert!(advance_cursor(Targets::ALL, Coordinates::ALL, by).is_err());
        }
        assert_eq!(
            increment(Targets::ALL, Xy::default()).unwrap().def(),
            &defs::INCADCXY
        );
        let i = advance_cursor(Targets::ALL, Coordinates::ROWS, Xy::default()).unwrap();
        for f in i.def().fields() {
            if ["X0", "X1", "Y0", "Y1"].contains(&f.name()) {
                assert_eq!(f.extract(i.word()), u32::from(f.name().starts_with('Y')));
            }
        }
    }
    #[test]
    fn zw_fields_are_checked_and_cross_thread_override_is_absent() {
        let by = Zw {
            z0: 7,
            w0: 6,
            z1: 5,
            w1: 4,
        };
        for bits in 1..8 {
            let targets = Targets(bits);
            for i in [
                increment_zw(targets, by).unwrap(),
                advance_cursor_zw(targets, ZwCoordinates::ALL, by).unwrap(),
            ] {
                let field = |name| {
                    i.def()
                        .fields()
                        .iter()
                        .find(|f| f.name() == name)
                        .unwrap()
                        .extract(i.word())
                };
                assert_eq!(field("U0"), u32::from(bits & 1 != 0));
                assert_eq!(field("U1"), u32::from(bits & 2 != 0));
                assert_eq!(field("PK"), u32::from(bits & 4 != 0));
                assert_eq!(field("ThreadOverride"), 0);
                assert_eq!(field("Z0Inc"), 7);
                assert_eq!(field("W0Inc"), 6);
                assert_eq!(field("Z1Inc"), 5);
                assert_eq!(field("W1Inc"), 4);
            }
        }
        for by in [
            Zw {
                z0: 8,
                ..Zw::default()
            },
            Zw {
                w0: 8,
                ..Zw::default()
            },
            Zw {
                z1: 8,
                ..Zw::default()
            },
            Zw {
                w1: 8,
                ..Zw::default()
            },
        ] {
            assert!(increment_zw(Targets::ALL, by).is_err());
            assert!(advance_cursor_zw(Targets::ALL, ZwCoordinates::ALL, by).is_err());
        }
        assert_eq!(
            increment_zw(Targets::ALL, Zw::default()).unwrap().def(),
            &defs::INCADCZW
        );
        let i = advance_cursor_zw(Targets::ALL, ZwCoordinates::ROWS, Zw::default()).unwrap();
        for f in i.def().fields() {
            if ["Z0", "Z1", "W0", "W1"].contains(&f.name()) {
                assert_eq!(f.extract(i.word()), u32::from(f.name().starts_with('W')));
            }
        }
    }
    #[test]
    fn zw_all_increments_and_coordinate_masks_encode_without_override() {
        for bits in 1..8 {
            for n in 0..8 {
                let by = Zw {
                    z0: n,
                    w0: n,
                    z1: n,
                    w1: n,
                };
                for mask in 0..16 {
                    let coords = ZwCoordinates {
                        z0: mask & 1 != 0,
                        w0: mask & 2 != 0,
                        z1: mask & 4 != 0,
                        w1: mask & 8 != 0,
                    };
                    let i = advance_cursor_zw(Targets(bits), coords, by).unwrap();
                    for (name, value) in [
                        ("Z0", mask & 1),
                        ("W0", (mask >> 1) & 1),
                        ("Z1", (mask >> 2) & 1),
                        ("W1", (mask >> 3) & 1),
                        ("Z0Inc", n),
                        ("W0Inc", n),
                        ("Z1Inc", n),
                        ("W1Inc", n),
                        ("ThreadOverride", 0),
                    ] {
                        let field = i.def().fields().iter().find(|f| f.name() == name).unwrap();
                        assert_eq!(field.extract(i.word()), value);
                    }
                    assert!(increment_zw(Targets(bits), by).is_ok());
                }
            }
        }
    }
}
