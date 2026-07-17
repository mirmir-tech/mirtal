#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    struct QuantizationOptions {
        group_size: i32,
        bits: i32,
        transpose: bool,
        sorted_indices: bool,
    }
    struct NativeRopeOptions {
        dimensions: i32,
        traditional: bool,
        has_base: bool,
        base: f32,
        scale: f32,
        offset: i32,
    }
    struct NativeAttentionOptions {
        scale: f32,
        mask_kind: u8,
    }
    unsafe extern "C++" {
        include!("mirtal/bridge.h");
        type Array;
        type Arrays;
        type Compiled;
        type MetalKernel;
        type PreparedMetal;
        type MetalLibrary;
        type MetalLaunch;
        type Stream;
        fn version() -> Result<String>;
        fn clear_memory_cache() -> Result<()>;
        fn configure_recommended_wired_limit() -> Result<bool>;
        fn active_memory() -> Result<usize>;
        fn cache_memory() -> Result<usize>;
        fn peak_memory() -> Result<usize>;
        fn memory_limit() -> Result<usize>;
        fn recommended_memory() -> Result<usize>;
        fn new_stream(kind: u8, index: i32) -> Result<UniquePtr<Stream>>;
        fn stream_native_value(stream: &Stream) -> usize;
        fn stream_id(stream: &Stream) -> u64;
        fn synchronize(stream: &Stream) -> Result<()>;
        fn array_from_f32(data: &[f32], shape: &[i32]) -> Result<SharedPtr<Array>>;
        fn array_from_u32(data: &[u32], shape: &[i32]) -> Result<SharedPtr<Array>>;
        fn array_from_owned_native_handle(address: usize) -> Result<SharedPtr<Array>>;
        fn array_native_handle(array: &Array) -> usize;
        fn array_shape(array: &Array) -> Result<Vec<i32>>;
        fn array_dtype(array: &Array) -> Result<u8>;
        fn array_len(array: &Array) -> usize;
        fn array_eval(array: &Array) -> Result<()>;
        fn array_copy_f32(array: &Array, stream: &Stream, output: &mut [f32]) -> Result<()>;
        fn array_copy_u32(array: &Array, stream: &Stream, output: &mut [u32]) -> Result<()>;

        fn add(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn add_scalar(input: &Array, value: f32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn multiply(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn multiply_scalar(input: &Array, value: f32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn divide(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn power_scalar(input: &Array, exponent: f32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn rms_norm(
            input: &Array,
            weight: &Array,
            eps: f32,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn rms_norm_unit(input: &Array, eps: f32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn astype(input: &Array, dtype: u8, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn reshape(input: &Array, shape: &[i32], stream: &Stream) -> Result<SharedPtr<Array>>;
        fn transpose(input: &Array, axes: &[i32], stream: &Stream) -> Result<SharedPtr<Array>>;
        fn expand_dims(input: &Array, axes: &[i32], stream: &Stream) -> Result<SharedPtr<Array>>;
        fn squeeze_axis(input: &Array, axis: i32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn sigmoid(input: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn sigmoid_multiply(
            gate: &Array,
            input: &Array,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn silu(input: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn tanh(input: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;

        fn quantize(
            input: &Array,
            group_size: i32,
            bits: i32,
            stream: &Stream,
        ) -> Result<UniquePtr<Arrays>>;
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
            rhs_indices: &Array,
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
        fn item_u32(input: &Array, stream: &Stream) -> Result<u32>;
        fn sdpa(
            queries: &Array,
            keys: &Array,
            values: &Array,
            options: &NativeAttentionOptions,
            mask: SharedPtr<Array>,
            sinks: SharedPtr<Array>,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn rope(
            input: &Array,
            options: &NativeRopeOptions,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn rope_with_frequencies(
            input: &Array,
            dimensions: i32,
            traditional: bool,
            frequencies: &Array,
            offset: i32,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn new_arrays() -> UniquePtr<Arrays>;
        fn arrays_push(arrays: Pin<&mut Arrays>, array: &Array);
        fn arrays_len(arrays: &Arrays) -> usize;
        fn arrays_get(arrays: &Arrays, index: usize) -> Result<SharedPtr<Array>>;
        fn arrays_stream(arrays: &Arrays) -> Result<&Stream>;
        fn new_compiled(
            callback: Box<GraphCallback>,
            shapeless: bool,
            stream: &Stream,
        ) -> Result<UniquePtr<Compiled>>;
        fn compiled_call(compiled: &Compiled, inputs: &Arrays) -> Result<UniquePtr<Arrays>>;
        fn compiled_native_handle(compiled: &Compiled) -> usize;
        fn new_metal_kernel(
            name: &str,
            input_names: &str,
            output_names: &str,
            source: &str,
            header: &str,
            row_contiguous: bool,
            atomic_outputs: bool,
        ) -> Result<UniquePtr<MetalKernel>>;
        fn metal_kernel_native_handle(kernel: &MetalKernel) -> usize;
        fn new_metal_library(name: &str, source: &str) -> Result<UniquePtr<MetalLibrary>>;
        fn metal_library_native_handle(library: &MetalLibrary) -> usize;
        fn new_metal_launch(
            grid_x: i32,
            grid_y: i32,
            grid_z: i32,
            group_x: i32,
            group_y: i32,
            group_z: i32,
            verbose: bool,
        ) -> UniquePtr<MetalLaunch>;
        fn metal_launch_add_output(launch: Pin<&mut MetalLaunch>, shape: &[i32], dtype: u8);
        fn metal_launch_add_template_int(launch: Pin<&mut MetalLaunch>, name: &str, value: i32);
        fn metal_launch_add_template_bool(launch: Pin<&mut MetalLaunch>, name: &str, value: bool);
        fn metal_launch_add_template_dtype(launch: Pin<&mut MetalLaunch>, name: &str, value: u8);
        fn metal_launch_set_init(launch: Pin<&mut MetalLaunch>, value: f32);
        fn metal_dispatch(
            kernel: &MetalKernel,
            inputs: &Arrays,
            launch: &MetalLaunch,
            stream: &Stream,
        ) -> Result<UniquePtr<Arrays>>;
        fn new_prepared_metal(
            kernel: &MetalKernel,
            launch: &MetalLaunch,
        ) -> Result<UniquePtr<PreparedMetal>>;
        fn prepared_metal_set_input(
            prepared: Pin<&mut PreparedMetal>,
            index: usize,
            input: &Array,
        ) -> Result<()>;
        fn prepared_metal_dispatch(
            prepared: Pin<&mut PreparedMetal>,
            stream: &Stream,
        ) -> Result<UniquePtr<Arrays>>;
    }

    extern "Rust" {
        type GraphCallback;
        fn invoke_graph(
            callback: &GraphCallback,
            inputs: UniquePtr<Arrays>,
        ) -> Result<UniquePtr<Arrays>>;
    }
}

pub mod aliasing;
mod callback;
pub mod graph;
pub mod io;
pub mod ops;
pub mod read;
mod traits;

pub(crate) use callback::invoke_graph;
pub use callback::{GraphCallback, graph_callback};
