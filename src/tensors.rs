use std::path::Path;

use cxx::UniquePtr;
use mirtal_sys::io::ffi;

use crate::{Array, Error, Result, Stream};

/// A lazily loaded map of arrays from a safetensors file.
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
