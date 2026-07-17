#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mirtal/bridge.h");

        type Array = crate::ffi::Array;

        fn array_read_f32(array: &Array, output: &mut [f32]) -> Result<()>;
        fn array_read_u32_scalar(array: &Array) -> Result<u32>;
    }
}
