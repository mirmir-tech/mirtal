#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mirtal/bridge/memory.h");
        type Array = crate::ffi::Array;
        type Allocation;
        fn array_allocation(array: &Array) -> Result<SharedPtr<Allocation>>;
        fn allocation_identity(allocation: &Allocation) -> usize;
        fn allocation_bytes(allocation: &Allocation) -> Result<usize>;
    }
}

// SAFETY: An allocation retains an immutable shared buffer owner. It never
// accesses tensor contents or mutates arrays; MLX allocator release is locked.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for ffi::Allocation {}
// SAFETY: Identity/size queries only read the retained buffer's immutable metadata.
unsafe impl Sync for ffi::Allocation {}
