#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    /// One named row-major array read lazily from `offset` of a file.
    struct TensorRange {
        name: String,
        offset: u64,
        dtype: DType,
        shape: Vec<u64>,
    }
    unsafe extern "C++" {
        include!("mirtal/bridge.h");

        type Array = crate::ffi::Array;
        type Stream = crate::ffi::Stream;
        type DType = crate::ffi::DType;
        type TensorMap;

        fn load_safetensors(path: &str, stream: &Stream) -> Result<UniquePtr<TensorMap>>;
        fn load_tensor_ranges(
            path: &str,
            ranges: &[TensorRange],
            stream: &Stream,
        ) -> Result<UniquePtr<TensorMap>>;
        fn tensor_map_len(tensors: &TensorMap) -> usize;
        fn tensor_map_eval(tensors: &TensorMap) -> Result<()>;
        fn tensor_map_contains(tensors: &TensorMap, name: &str) -> bool;
        fn tensor_map_get(tensors: &TensorMap, name: &str) -> Result<SharedPtr<Array>>;
        fn tensor_map_take(tensors: Pin<&mut TensorMap>, name: &str) -> Result<SharedPtr<Array>>;
    }
}
