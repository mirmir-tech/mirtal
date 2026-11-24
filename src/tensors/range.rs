//! A named row-major array stored at an offset of a file, valid by
//! construction: only its place in a particular file remains to check.

use crate::{DType, Error, Result, TensorRangeError};

/// A named row-major array stored at `offset` of a file.
///
/// Containers other than safetensors (for example GGUF) describe each tensor
/// by its file offset, element type and shape. Opaque block encodings are
/// described as `Uint8` rows; the bytes reach memory unchanged and are read
/// only when the array is evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRange {
    name: String,
    offset: u64,
    dtype: DType,
    shape: Vec<u64>,
    bytes: u64,
}

impl TensorRange {
    /// A range of `shape` elements of `dtype` from byte `offset`; every
    /// dimension is positive and fits `i32`, the byte span fits `u64`, and
    /// the name has no control characters.
    pub fn new(
        name: impl Into<String>,
        offset: u64,
        dtype: DType,
        shape: Vec<u64>,
    ) -> Result<Self> {
        let name = name.into();
        let invalid = |reason| Err(Error::TensorRange { name: name.clone(), reason });
        if name.chars().any(char::is_control) {
            return invalid(TensorRangeError::ControlCharacter);
        }
        if shape.is_empty() {
            return invalid(TensorRangeError::Scalar);
        }
        let mut bytes = dtype.size();
        for &dimension in &shape {
            if dimension == 0 {
                return invalid(TensorRangeError::EmptyDimension);
            }
            if i32::try_from(dimension).is_err() {
                return invalid(TensorRangeError::DimensionTooLarge(dimension));
            }
            let Some(total) = bytes.checked_mul(dimension) else {
                return invalid(TensorRangeError::SizeOverflow);
            };
            bytes = total;
        }
        if offset.checked_add(bytes).is_none() {
            return invalid(TensorRangeError::SizeOverflow);
        }
        Ok(Self { name, offset, dtype, shape, bytes })
    }

    /// Name under which the array is returned by [`crate::TensorFile::get`].
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Absolute byte offset of the first element in the file.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Element type of the stored values.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Row-major shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Bytes the range spans.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Exclusive end offset, which [`Self::new`] keeps within `u64`.
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.offset + self.bytes
    }
}
