use cxx::UniquePtr;
use mirtal_sys::ffi;

use super::MetalSource;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy)]
/// Checked source used to construct a complete Metal library.
pub struct MetalLibraryDescriptor {
    /// Human-readable library name.
    pub name: &'static str,
    /// Complete Metal translation unit validated at build time.
    pub source: MetalSource,
}

/// A compiled Metal library that can export checked functions.
pub struct MetalLibrary {
    raw: UniquePtr<ffi::MetalLibrary>,
    name: &'static str,
}

impl MetalLibrary {
    /// Compiles a complete, previously validated Metal translation unit.
    pub fn new(descriptor: MetalLibraryDescriptor) -> Result<Self> {
        if descriptor.name.is_empty() || descriptor.source.code().is_empty() {
            return Err(Error::InvalidKernel(format!(
                "Metal library from {} has an empty name or source",
                descriptor.source.origin()
            )));
        }
        let raw = ffi::new_metal_library(descriptor.name, descriptor.source.code())?;
        if raw.is_null() {
            return Err(Error::NullHandle("Metal library"));
        }
        Ok(Self { raw, name: descriptor.name })
    }

    pub(crate) fn native(&self) -> Result<&ffi::MetalLibrary> {
        self.raw.as_ref().ok_or(Error::NullHandle("Metal library"))
    }
}

impl std::fmt::Debug for MetalLibrary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalLibrary")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}
