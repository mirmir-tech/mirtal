#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mirtal/bridge.h");

        type Array = crate::ffi::Array;

        fn export_graph_dot(array: &Array, path: &str) -> Result<()>;
    }
}
