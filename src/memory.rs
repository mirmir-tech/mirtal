//! Process-wide MLX memory accounting and cache controls.

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A snapshot of MLX allocator statistics, in bytes.
pub struct MemoryStats {
    /// Bytes currently held by live arrays.
    pub active: usize,
    /// Bytes retained in the allocator cache.
    pub cached: usize,
    /// Peak active allocation in bytes.
    pub peak: usize,
    /// Current process memory limit in bytes.
    pub limit: usize,
    /// Platform-recommended working-set limit in bytes.
    pub recommended: usize,
}

/// Reads a consistent snapshot of the process-wide MLX allocator counters.
pub fn stats() -> Result<MemoryStats> {
    Ok(MemoryStats {
        active: mirtal_sys::ffi::active_memory()?,
        cached: mirtal_sys::ffi::cache_memory()?,
        peak: mirtal_sys::ffi::peak_memory()?,
        limit: mirtal_sys::ffi::memory_limit()?,
        recommended: mirtal_sys::ffi::recommended_memory()?,
    })
}

/// Configures the wired-memory limit recommended by the current platform.
///
/// Returns whether MLX changed the limit.
pub fn configure_recommended_wired_limit() -> Result<bool> {
    Ok(mirtal_sys::ffi::configure_recommended_wired_limit()?)
}

/// Releases reusable allocations held by the process-wide MLX cache.
pub fn clear_cache() -> Result<()> {
    Ok(mirtal_sys::ffi::clear_memory_cache()?)
}
