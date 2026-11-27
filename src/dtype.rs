use mirtal_sys::ffi;

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// An element type supported by the safe mirtal API.
pub enum DType {
    /// Boolean values.
    Bool,
    /// Unsigned 32-bit integers.
    Uint32,
    /// Signed 32-bit integers.
    Int32,
    /// IEEE 754 half-precision values.
    Float16,
    /// Brain floating-point values.
    Bfloat16,
    /// IEEE 754 single-precision values.
    Float32,
    /// Unsigned 8-bit integers.
    Uint8,
}

impl DType {
    /// Bytes per element.
    #[must_use]
    pub const fn size(self) -> u64 {
        match self {
            Self::Bool | Self::Uint8 => 1,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
        }
    }
}

impl From<DType> for ffi::DType {
    fn from(value: DType) -> Self {
        match value {
            DType::Bool => Self::Bool,
            DType::Uint32 => Self::Uint32,
            DType::Int32 => Self::Int32,
            DType::Float16 => Self::Float16,
            DType::Bfloat16 => Self::Bfloat16,
            DType::Float32 => Self::Float32,
            DType::Uint8 => Self::Uint8,
        }
    }
}

impl TryFrom<ffi::DType> for DType {
    type Error = Error;

    /// The bridge enum can carry a value C++ wrote that no variant names.
    fn try_from(value: ffi::DType) -> Result<Self> {
        match value {
            ffi::DType::Bool => Ok(Self::Bool),
            ffi::DType::Uint32 => Ok(Self::Uint32),
            ffi::DType::Int32 => Ok(Self::Int32),
            ffi::DType::Float16 => Ok(Self::Float16),
            ffi::DType::Bfloat16 => Ok(Self::Bfloat16),
            ffi::DType::Float32 => Ok(Self::Float32),
            ffi::DType::Uint8 => Ok(Self::Uint8),
            _ => Err(Error::UnsupportedDtype(value.repr)),
        }
    }
}
