#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mirtal/bridge.h");

        type Array = crate::ffi::Array;
        type Stream = crate::ffi::Stream;
        type TensorMap;

        fn load_safetensors(path: &str, stream: &Stream) -> Result<UniquePtr<TensorMap>>;
        fn tensor_map_len(tensors: &TensorMap) -> usize;
        fn tensor_map_eval(tensors: &TensorMap) -> Result<()>;
        fn tensor_map_contains(tensors: &TensorMap, name: &str) -> bool;
        fn tensor_map_get(tensors: &TensorMap, name: &str) -> Result<SharedPtr<Array>>;
    }
}
