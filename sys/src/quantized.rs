#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    struct QuantizationOptions {
        group_size: i32,
        bits: i32,
        transpose: bool,
        sorted_indices: bool,
    }
    struct NativeGatherIndices {
        lhs: SharedPtr<Array>,
        rhs: SharedPtr<Array>,
    }
    unsafe extern "C++" {
        include!("mirtal/bridge.h");
        type Array = crate::ffi::Array;
        type Arrays = crate::ffi::Arrays;
        type Stream = crate::ffi::Stream;
        fn quantize(
            input: &Array,
            group_size: i32,
            bits: i32,
            stream: &Stream,
        ) -> Result<UniquePtr<Arrays>>;
        fn quantize_mxfp8(input: &Array, stream: &Stream) -> Result<UniquePtr<Arrays>>;
        fn quantize_mxfp4(input: &Array, stream: &Stream) -> Result<UniquePtr<Arrays>>;
        fn mxfp4_matmul(
            input: &Array,
            weight: &Array,
            scales: &Array,
            transpose: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn gather_mxfp4(
            input: &Array,
            weight: &Array,
            scales: &Array,
            indices: &NativeGatherIndices,
            transpose: bool,
            sorted_indices: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn mxfp8_matmul(
            input: &Array,
            weight: &Array,
            scales: &Array,
            transpose: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn dequantize_mxfp8(
            weight: &Array,
            scales: &Array,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn quantized_matmul(
            input: &Array,
            weight: &Array,
            scales: &Array,
            biases: &Array,
            options: &QuantizationOptions,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn gather_qmm(
            input: &Array,
            weight: &Array,
            scales: &Array,
            biases: &Array,
            indices: &NativeGatherIndices,
            options: &QuantizationOptions,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn dequantize(
            weight: &Array,
            scales: &Array,
            biases: &Array,
            options: &QuantizationOptions,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
    }
}
