#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

mod array;
mod attention;
mod compile;
mod device;
mod dtype;
mod element;
mod error;
mod graph;
pub mod interop;
/// Process-wide MLX memory statistics and cache controls.
pub mod memory;
mod metal;
mod ops;
mod quantized;
mod rope;
mod shape;
mod stream;
mod tensors;

pub use array::Array;
pub use attention::{AttentionMask, ScaledDotProductAttention};
pub use compile::{CompileOptions, Compiled};
pub use device::{Device, DeviceKind};
pub use dtype::DType;
pub use element::Element;
pub use error::{Error, Result};
pub use graph::Graph;
pub use metal::{
    AliasingDispatch, DTypeConstraint, Dispatch, KernelDescriptor, MetalFunction, MetalKernel,
    MetalLibrary, MetalLibraryDescriptor, MetalSource, OutputSpec, PreparedAliasing,
    PreparedMetalKernel, StrideBinding, TemplateArg, TemplateKind, TemplateParameter,
    TemplateValue,
};
pub use mirtal_macros::{compiled, metal_kernel, metal_library};
pub use quantized::{
    GatherQmmOptions, MxFp8, MxFp8Arrays, Quantization, Quantized, QuantizedArrays,
    SymmetricQuantization,
};
pub use rope::{FrequencyRopeOptions, RopeOptions};
pub use shape::Shape;
pub use stream::Stream;
pub use tensors::TensorFile;

/// Returns the version of the linked native MLX library.
pub fn version() -> Result<String> {
    Ok(mirtal_sys::ffi::version()?)
}

#[macro_export]
/// Creates a validated [`Shape`] from a comma-separated list of dimensions.
macro_rules! shape {
    ($($dimension:expr),+ $(,)?) => {
        $crate::Shape::new([$($dimension),+])
    };
}
