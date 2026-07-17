use super::{aliasing, ffi};

// SAFETY: MLX arrays are immutable shared graph handles.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ffi::Array {}
// SAFETY: Concurrent graph composition does not mutate an MLX array handle.
unsafe impl Sync for ffi::Array {}
// SAFETY: Stream use remains serialized by the non-Sync safe wrapper.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ffi::Stream {}
// SAFETY: Compiled calls remain serialized through the owning stream.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ffi::Compiled {}
// SAFETY: Kernel descriptors are immutable after construction.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ffi::MetalKernel {}
// SAFETY: Immutable descriptors may target independent explicit streams.
unsafe impl Sync for ffi::MetalKernel {}
// SAFETY: Prepared aliasing plans mutate only through a unique borrow or the
// caller's mutex; their retained MLX arrays are immutable shared graph handles.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for aliasing::ffi::AliasingPlan {}
// SAFETY: Prepared launches mutate only through a unique borrow. Their copied
// kernel descriptor is immutable and retained MLX arrays are shared handles.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ffi::PreparedMetal {}
