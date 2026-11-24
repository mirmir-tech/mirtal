mod range;
#[cfg(test)]
mod tests;

use std::{collections::HashSet, path::Path};

use cxx::UniquePtr;
use mirtal_sys::io::ffi;
pub use range::TensorRange;

use crate::{Array, Error, Result, Stream, TensorRangeError};

/// A lazily loaded map of named arrays from a tensor file.
pub struct TensorFile {
    raw: UniquePtr<ffi::TensorMap>,
}

impl TensorFile {
    /// Opens a safetensors file and binds its arrays to `stream`.
    pub fn load(path: &Path, stream: &Stream) -> Result<Self> {
        let path = path.to_str().ok_or_else(|| Error::InvalidPath(path.to_path_buf()))?;
        let raw = ffi::load_safetensors(path, stream.native()?)?;
        if raw.is_null() {
            return Err(Error::NullHandle("safetensors map"));
        }
        Ok(Self { raw })
    }

    /// Binds `ranges` of the file at `path` as lazily read arrays.
    ///
    /// Like [`TensorFile::load`], reads evaluate on a CPU stream. Each read
    /// copies its range from the file into a new buffer once, on
    /// evaluation; nothing is memory-mapped.
    ///
    /// Names must be unique and every range must end within the file.
    pub fn load_ranges(path: &Path, ranges: &[TensorRange], stream: &Stream) -> Result<Self> {
        let file = std::fs::metadata(path)?.len();
        let mut names = HashSet::with_capacity(ranges.len());
        for range in ranges {
            let invalid = |reason| Error::TensorRange { name: range.name().to_owned(), reason };
            if !names.insert(range.name()) {
                return Err(invalid(TensorRangeError::Duplicate));
            }
            if range.end() > file {
                return Err(invalid(TensorRangeError::PastEndOfFile { end: range.end(), file }));
            }
        }
        let utf8 = path.to_str().ok_or_else(|| Error::InvalidPath(path.to_path_buf()))?;
        let ranges = ranges
            .iter()
            .map(|range| ffi::TensorRange {
                name: range.name().to_owned(),
                offset: range.offset(),
                dtype: range.dtype().into(),
                shape: range.shape().to_vec(),
            })
            .collect::<Vec<_>>();
        let raw = ffi::load_tensor_ranges(utf8, &ranges, stream.native()?)?;
        if raw.is_null() {
            return Err(Error::NullHandle("tensor range map"));
        }
        Ok(Self { raw })
    }

    /// Returns the number of named arrays in the file.
    pub fn len(&self) -> usize {
        self.raw.as_ref().map_or(0, ffi::tensor_map_len)
    }

    #[must_use]
    /// Returns whether the file contains no arrays.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluates every array loaded from the file.
    pub fn evaluate(&self) -> Result<()> {
        Ok(ffi::tensor_map_eval(self.native()?)?)
    }

    /// Returns whether the file contains an array named `name`.
    pub fn contains(&self, name: &str) -> Result<bool> {
        Ok(ffi::tensor_map_contains(self.native()?, name))
    }

    /// Returns the array named `name`.
    pub fn get(&self, name: &str) -> Result<Array> {
        Array::from_raw(ffi::tensor_map_get(self.native()?, name)?, "safetensor")
    }

    /// Removes the array named `name` and returns it, so the file no longer
    /// keeps it alive: a caller that replaces it (by a cast, say) frees it
    /// once the replacement is evaluated.
    pub fn take(&mut self, name: &str) -> Result<Array> {
        let tensors = self.raw.as_mut().ok_or(Error::NullHandle("safetensors map"))?;
        Array::from_raw(ffi::tensor_map_take(tensors, name)?, "safetensor")
    }

    fn native(&self) -> Result<&ffi::TensorMap> {
        self.raw.as_ref().ok_or(Error::NullHandle("safetensors map"))
    }
}

impl std::fmt::Debug for TensorFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorFile")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
