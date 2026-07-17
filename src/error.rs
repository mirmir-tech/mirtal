use thiserror::Error;

/// Result type returned by mirtal operations.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
/// An error produced by validation, MLX, or the native boundary.
pub enum Error {
    /// The native MLX operation raised an exception.
    #[error("native MLX operation failed: {0}")]
    Native(#[from] cxx::Exception),
    /// A native constructor unexpectedly returned a null handle.
    #[error("MLX returned a null {0} handle")]
    NullHandle(&'static str),
    /// Host data length does not match the requested shape.
    #[error("shape {shape:?} contains {elements} elements, data contains {data}")]
    Shape {
        /// Requested dimensions.
        shape: Vec<usize>,
        /// Element count implied by `shape`.
        elements: usize,
        /// Number of supplied host elements.
        data: usize,
    },
    /// Multiplying shape dimensions overflowed `usize`.
    #[error("shape element count overflowed usize")]
    ShapeOverflow,
    /// A value could not be represented by the native integer type.
    #[error("integer conversion failed: {0}")]
    Integer(#[from] std::num::TryFromIntError),
    /// A path could not be represented as UTF-8 for the native API.
    #[error("path is not valid UTF-8: {0}")]
    InvalidPath(std::path::PathBuf),
    /// MLX returned a dtype code unknown to mirtal.
    #[error("MLX returned unsupported dtype code {0}")]
    UnsupportedDtype(u8),
    /// An array's dtype did not satisfy an operation's contract.
    #[error("array dtype is {actual:?}, expected {expected:?}")]
    DtypeMismatch {
        /// Dtype found on the array.
        actual: crate::DType,
        /// Dtype required by the operation.
        expected: crate::DType,
    },
    /// A native result contained an unexpected number of arrays.
    #[error("{operation} expected {expected} arrays, received {actual}")]
    Arity {
        /// Operation whose arity was checked.
        operation: &'static str,
        /// Required number of arrays.
        expected: usize,
        /// Observed number of arrays.
        actual: usize,
    },
    /// A compiled graph was called on a different stream.
    #[error("compiled graph belongs to stream {expected}, received stream {actual}")]
    StreamMismatch {
        /// Identifier of the creating stream.
        expected: u64,
        /// Identifier of the supplied stream.
        actual: u64,
    },
    /// A Metal kernel or library descriptor is invalid.
    #[error("invalid Metal kernel descriptor: {0}")]
    InvalidKernel(String),
    /// Metal launch geometry or bindings are invalid.
    #[error("invalid Metal dispatch: {0}")]
    InvalidDispatch(String),
    /// An affine quantization configuration is invalid.
    #[error("invalid affine quantization: {0}")]
    InvalidQuantization(String),
    /// A rotary-position-encoding configuration is invalid.
    #[error("invalid RoPE configuration: {0}")]
    InvalidRope(String),
    /// An attention configuration is invalid.
    #[error("invalid attention configuration: {0}")]
    InvalidAttention(String),
    /// A generic graph operation received invalid arguments.
    #[error("invalid graph operation: {0}")]
    InvalidOperation(String),
}
