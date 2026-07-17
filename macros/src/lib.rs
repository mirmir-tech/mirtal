#![deny(missing_docs)]

//! Build-time checked procedural macros re-exported by `mirtal`.

mod compiled;
mod library;
mod metal;

use proc_macro::TokenStream;

#[proc_macro_attribute]
/// Generates a typed factory for an explicitly streamed compiled graph.
///
/// Apply the attribute to a function whose first argument is `mirtal::Graph`
/// and whose second argument and result are fixed-size arrays. For a function
/// named `swiglu`, the macro generates `compile_swiglu(&Stream)` returning a
/// `mirtal::Compiled<INPUTS, OUTPUTS>`.
///
/// Use `#[mirtal::compiled(shapeless)]` when the compiled graph may accept
/// input shapes other than the shapes used during its initial trace. Omitting
/// the option keeps MLX's default shape-specialized behavior.
///
/// # Example
///
/// ```rust,ignore
/// use mirtal::{Array, Device, Graph, Result};
///
/// #[mirtal::compiled(shapeless)]
/// fn swiglu(graph: Graph<'_>, [gate, value]: [Array; 2]) -> Result<[Array; 1]> {
///     let activated = graph.silu(&gate)?;
///     Ok([graph.multiply(&activated, &value)?])
/// }
///
/// fn main() -> Result<()> {
///     let stream = Device::gpu(0).new_stream()?;
///     let gate = Array::from_slice(&[0.0_f32], [1])?;
///     let value = Array::from_slice(&[1.0_f32], [1])?;
///
///     let compiled = compile_swiglu(&stream)?;
///     let [output] = compiled.call(&stream, [&gate, &value])?;
///     let values = stream.read::<f32>(&output)?;
///     assert_eq!(values, vec![0.0]);
///     Ok(())
/// }
/// ```
pub fn compiled(attributes: TokenStream, item: TokenStream) -> TokenStream {
    compiled::expand_macro(attributes, item)
}

#[proc_macro]
/// Declares a fixed-arity Metal kernel with a checked Rust-side contract.
///
/// The macro generates a factory function returning a typed
/// `mirtal::MetalKernel<INPUTS, OUTPUTS>`. `source` and `header` accept either
/// `inline "..."` or `file "path/to/source.metal"`; file paths are relative to
/// `CARGO_MANIFEST_DIR`. During Rust compilation, Apple's Metal compiler checks
/// the generated function and reports invalid MSL at the macro invocation.
///
/// Buffer types may be `bool`, `u32`, `i32`, `f16`, `bf16`, `f32`, the generic
/// floating-point constraint `float`, or a declared dtype template. Templates
/// use `dtype`, `int`, or `bool` defaults and must be supplied explicitly in
/// `mirtal::Dispatch` when launching the kernel.
///
/// # Example
///
/// ```rust,ignore
/// use mirtal::{Array, DType, Device, Dispatch, OutputSpec, Result, Shape};
///
/// mirtal::metal_kernel! {
///     fn double {
///         name: "double_f32",
///         templates: [],
///         inputs: [input: f32],
///         outputs: [output: f32],
///         source: inline r"
///             uint index = thread_position_in_grid.x;
///             output[index] = input[index] * 2.0f;
///         ",
///         header: inline "",
///         row_contiguous: true,
///         atomic_outputs: false,
///     }
/// }
///
/// fn main() -> Result<()> {
///     let stream = Device::gpu(0).new_stream()?;
///     let input = Array::from_slice(&[2.0_f32, 3.0], [2])?;
///     let outputs = [OutputSpec::new(Shape::new([2])?, DType::Float32)];
///     let launch = Dispatch::new([2, 1, 1], [2, 1, 1]);
///     let [output] = double()?.dispatch(&stream, [&input], &outputs, &launch)?;
///
///     assert_eq!(stream.read::<f32>(&output)?, vec![4.0, 6.0]);
///     Ok(())
/// }
/// ```
pub fn metal_kernel(input: TokenStream) -> TokenStream {
    metal::expand(input)
}

#[proc_macro]
/// Declares and validates a complete Metal translation unit.
///
/// The macro generates a factory returning `mirtal::MetalLibrary`. Unlike
/// `metal_kernel!`, the source contains complete function signatures and may
/// export several kernels. Select a checked function with
/// `mirtal::MetalLibrary::export`, then use the aliasing dispatch API for
/// kernels that mutate buffers supplied by the caller.
///
/// `source` accepts `inline "..."` or `file "path/to/library.metal"`. File
/// paths are relative to `CARGO_MANIFEST_DIR`, and the complete translation unit
/// is validated with Apple's Metal compiler during the Rust build.
///
/// # Example
///
/// ```rust,ignore
/// use mirtal::{AliasingDispatch, Array, Device, Result};
///
/// mirtal::metal_library! {
///     fn copy_library {
///         name: "copy_library",
///         source: inline r"
///             #include <metal_stdlib>
///             using namespace metal;
///
///             kernel void copy_f32(
///                 const device float* input [[buffer(0)]],
///                 device float* output [[buffer(1)]],
///                 uint index [[thread_position_in_grid]]) {
///                 output[index] = input[index];
///             }
///         ",
///     }
/// }
///
/// fn main() -> Result<()> {
///     let stream = Device::gpu(0).new_stream()?;
///     let input = Array::from_slice(&[3.0_f32, 7.0], [2])?;
///     let output = Array::from_slice(&[0.0_f32, 0.0], [2])?;
///     let launch = AliasingDispatch::new([1])
///         .grid([2, 1, 1])
///         .threadgroup([2, 1, 1]);
///
///     let [copied] = copy_library()?
///         .export("copy_f32")?
///         .dispatch_aliasing_array(&stream, &[&input, &output], &launch)?;
///     assert_eq!(stream.read::<f32>(&copied)?, vec![3.0, 7.0]);
///     Ok(())
/// }
/// ```
pub fn metal_library(input: TokenStream) -> TokenStream {
    library::expand(input)
}
