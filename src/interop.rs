//! Transitional zero-copy interoperability with external MLX C++ owners.
//!
//! New integrations should use the safe `mirtal` API. These functions exist so
//! runtimes can migrate an established native ABI one owned handle at a time.

use crate::{Array, Compiled, Error, MetalKernel, MetalLibrary, Result, Stream};

/// Returns the address of the native `mirtal::Array` wrapper.
///
/// The address is non-owning and remains valid only while `array` or one of its
/// clones retains the same native wrapper.
pub fn array_native_handle(array: &Array) -> Result<usize> {
    let address = mirtal_sys::ffi::array_native_handle(array.native()?);
    if address == 0 {
        return Err(Error::NullHandle("native array"));
    }
    Ok(address)
}

/// Takes ownership of a heap-allocated native `mirtal::Array` wrapper.
///
/// No tensor buffer is copied; only the existing `mx::array` handle is adopted
/// into the RAII owner used by [`Array`].
///
/// # Safety
///
/// `address` must point to a live `mirtal::Array` allocated with `new`. Its
/// ownership must not have been transferred before, and the caller must not
/// access or delete the wrapper after this call.
pub unsafe fn array_from_owned_native_handle(address: usize) -> Result<Array> {
    Array::from_raw(mirtal_sys::ffi::array_from_owned_native_handle(address)?, "owned native array")
}

/// Returns a non-owning address of the `mlx::core::Stream` value.
///
/// The address remains valid only while `stream` is alive. It is intended for a
/// transitional native adapter that immediately copies the MLX stream value.
pub fn stream_native_value(stream: &Stream) -> Result<usize> {
    let address = mirtal_sys::ffi::stream_native_value(stream.native()?);
    if address == 0 {
        return Err(Error::NullHandle("native stream value"));
    }
    Ok(address)
}

/// Returns a non-owning address of a compiled MLX graph.
///
/// The address remains valid only while `compiled` is alive. This is intended
/// for native adapters that still compose a larger graph in C++ while mirtal
/// owns and caches the reusable compiled subgraph.
pub fn compiled_native_handle<const INPUTS: usize, const OUTPUTS: usize>(
    compiled: &Compiled<INPUTS, OUTPUTS>,
) -> Result<usize> {
    let address = mirtal_sys::ffi::compiled_native_handle(compiled.native()?);
    if address == 0 {
        return Err(Error::NullHandle("native compiled graph"));
    }
    Ok(address)
}

/// Returns a non-owning address of a checked Metal kernel.
///
/// The address remains valid only while `kernel` is alive. It supports gradual
/// migration of native graph composition without duplicating kernel source or
/// constructing MLX custom kernels outside mirtal.
pub fn metal_kernel_native_handle<const INPUTS: usize, const OUTPUTS: usize>(
    kernel: &MetalKernel<INPUTS, OUTPUTS>,
) -> Result<usize> {
    let address = mirtal_sys::ffi::metal_kernel_native_handle(kernel.native()?);
    if address == 0 {
        return Err(Error::NullHandle("native Metal kernel"));
    }
    Ok(address)
}

/// Returns a non-owning address of a checked full Metal library.
pub fn metal_library_native_handle(library: &MetalLibrary) -> Result<usize> {
    let address = mirtal_sys::ffi::metal_library_native_handle(library.native()?);
    if address == 0 {
        return Err(Error::NullHandle("native Metal library"));
    }
    Ok(address)
}
