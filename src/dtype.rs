use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
/// An element type supported by the safe mirtal API.
pub enum DType {
    /// Boolean values.
    Bool = 0,
    /// Unsigned 32-bit integers.
    Uint32 = 1,
    /// Signed 32-bit integers.
    Int32 = 2,
    /// IEEE 754 half-precision values.
    Float16 = 3,
    /// Brain floating-point values.
    Bfloat16 = 4,
    /// IEEE 754 single-precision values.
    Float32 = 5,
    /// Unsigned 8-bit integers.
    Uint8 = 6,
}

impl TryFrom<u8> for DType {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Bool),
            1 => Ok(Self::Uint32),
            2 => Ok(Self::Int32),
            3 => Ok(Self::Float16),
            4 => Ok(Self::Bfloat16),
            5 => Ok(Self::Float32),
            6 => Ok(Self::Uint8),
            _ => Err(Error::UnsupportedDtype(value)),
        }
    }
}
