use std::{cell::Cell, marker::PhantomData};

use cxx::UniquePtr;
use mirtal_sys::ffi;

use crate::{Array, Device, Element, Error, Graph, Result, element};

/// An explicit MLX execution stream.
///
/// Streams are intentionally not [`Sync`]; all graph construction and host
/// synchronization is tied to a concrete stream value.
pub struct Stream {
    raw: UniquePtr<ffi::Stream>,
    not_sync: PhantomData<Cell<()>>,
}

impl Stream {
    pub(crate) fn new(device: Device) -> Result<Self> {
        let raw = ffi::new_stream(device.native_kind(), device.native_index()?)?;
        if raw.is_null() {
            return Err(Error::NullHandle("stream"));
        }
        Ok(Self { raw, not_sync: PhantomData })
    }

    #[must_use]
    /// Returns the stable native identifier of this stream.
    pub fn id(&self) -> u64 {
        self.raw.as_ref().map_or(0, ffi::stream_id)
    }

    #[must_use]
    /// Returns a graph-operation facade bound to this stream.
    pub const fn graph(&self) -> Graph<'_> {
        Graph::new(self)
    }

    /// Schedules evaluation of `array`.
    pub fn eval(&self, array: &Array) -> Result<()> {
        Ok(ffi::array_eval(array.native()?)?)
    }

    /// Waits until all work submitted to this stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        Ok(ffi::synchronize(self.native()?)?)
    }

    /// Evaluates `array` and copies all elements to host memory.
    pub fn read<T: Element>(&self, array: &Array) -> Result<Vec<T>> {
        element::copy(array, self)
    }

    /// Evaluates a scalar array and copies its `u32` value to the host.
    pub fn read_scalar_u32(&self, array: &Array) -> Result<u32> {
        Ok(ffi::item_u32(array.native()?, self.native()?)?)
    }

    pub(crate) fn native(&self) -> Result<&ffi::Stream> {
        self.raw.as_ref().ok_or(Error::NullHandle("stream"))
    }
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Stream").field("id", &self.id()).finish_non_exhaustive()
    }
}
