use std::{
    fmt,
    hash::{Hash, Hasher},
};

use mirtal_sys::memory::ffi;

use crate::{Array, Result};

#[derive(Clone)]
/// A retained allocation belonging to an already available array.
///
/// Equality and hashing identify shared storage, including slices. Keeping this
/// handle alive retains the buffer, not its computation graph. It can inhibit
/// donation, so use it for retained-memory accounting, not the execution loop.
pub struct Allocation {
    _owner: cxx::SharedPtr<ffi::Allocation>,
    identity: usize,
    bytes: usize,
}

impl Array {
    /// Returns the backing allocation without evaluating or synchronizing.
    ///
    /// Returns `None` for unavailable arrays or arrays without storage. Evaluate
    /// and synchronize explicitly before taking a complete memory inventory.
    /// The allocation size includes the whole backing buffer, not just a view.
    pub fn allocation(&self) -> Result<Option<Allocation>> {
        let owner = ffi::array_allocation(self.native()?)?;
        let Some(native) = owner.as_ref() else {
            return Ok(None);
        };
        let identity = ffi::allocation_identity(native);
        let bytes = ffi::allocation_bytes(native)?;
        Ok(Some(Allocation { _owner: owner, identity, bytes }))
    }
}

impl Allocation {
    #[must_use]
    /// Returns the allocator-reported size of the retained buffer in bytes.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

impl PartialEq for Allocation {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}
impl Eq for Allocation {}
impl Hash for Allocation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}
impl fmt::Debug for Allocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Allocation")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
