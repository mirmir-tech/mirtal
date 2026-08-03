use mirtal_sys::ffi;

use crate::{Array, DType, Result, Shape, Stream};

#[derive(Clone, Copy)]
/// Lazy array operations bound to an explicit execution stream.
pub struct Graph<'stream> {
    stream: GraphStream<'stream>,
}

#[derive(Clone, Copy)]
enum GraphStream<'stream> {
    Public(&'stream Stream),
    Native(&'stream ffi::Stream),
}

impl<'stream> Graph<'stream> {
    pub(crate) const fn new(stream: &'stream Stream) -> Self {
        Self { stream: GraphStream::Public(stream) }
    }

    pub(crate) const fn from_native(stream: &'stream ffi::Stream) -> Self {
        Self { stream: GraphStream::Native(stream) }
    }

    /// Adds two arrays with MLX broadcasting rules.
    pub fn add(self, left: &Array, right: &Array) -> Result<Array> {
        Array::from_raw(ffi::add(left.native()?, right.native()?, self.native()?)?, "add")
    }

    /// Adds a scalar to every element of an array.
    pub fn add_scalar(self, input: &Array, value: f32) -> Result<Array> {
        Array::from_raw(ffi::add_scalar(input.native()?, value, self.native()?)?, "add_scalar")
    }

    /// Multiplies two arrays with MLX broadcasting rules.
    pub fn multiply(self, left: &Array, right: &Array) -> Result<Array> {
        Array::from_raw(ffi::multiply(left.native()?, right.native()?, self.native()?)?, "multiply")
    }

    /// Multiplies every array element by a scalar.
    pub fn multiply_scalar(self, input: &Array, value: f32) -> Result<Array> {
        Array::from_raw(
            ffi::multiply_scalar(input.native()?, value, self.native()?)?,
            "multiply_scalar",
        )
    }

    /// Divides two arrays with MLX broadcasting rules.
    pub fn divide(self, left: &Array, right: &Array) -> Result<Array> {
        Array::from_raw(ffi::divide(left.native()?, right.native()?, self.native()?)?, "divide")
    }

    /// Raises every array element to a scalar power.
    pub fn power_scalar(self, input: &Array, exponent: f32) -> Result<Array> {
        Array::from_raw(
            ffi::power_scalar(input.native()?, exponent, self.native()?)?,
            "power_scalar",
        )
    }

    /// Applies RMS normalization with a learned weight.
    pub fn rms_norm(self, input: &Array, weight: &Array, eps: f32) -> Result<Array> {
        Array::from_raw(
            ffi::rms_norm(input.native()?, weight.native()?, eps, self.native()?)?,
            "rms_norm",
        )
    }

    /// Applies RMS normalization with an implicit unit weight.
    pub fn rms_norm_unit(self, input: &Array, eps: f32) -> Result<Array> {
        Array::from_raw(ffi::rms_norm_unit(input.native()?, eps, self.native()?)?, "rms_norm_unit")
    }

    /// Casts an array to `dtype`.
    pub fn astype(self, input: &Array, dtype: DType) -> Result<Array> {
        Array::from_raw(ffi::astype(input.native()?, dtype as u8, self.native()?)?, "astype")
    }

    /// Converts E4M3 bytes into the requested floating-point type.
    pub fn from_fp8(self, input: &Array, dtype: DType) -> Result<Array> {
        Array::from_raw(
            ffi::from_fp8(input.native()?, dtype as u8, self.native()?)?,
            "FP8 conversion",
        )
    }

    /// Converts floating-point values into E4M3 bytes.
    pub fn to_fp8(self, input: &Array) -> Result<Array> {
        Array::from_raw(ffi::to_fp8(input.native()?, self.native()?)?, "FP8 conversion")
    }

    /// Reinterprets the storage of `input` with an equally sized element type.
    pub fn view_dtype(self, input: &Array, dtype: DType) -> Result<Array> {
        Array::from_raw(
            ffi::view_dtype(input.native()?, dtype as u8, self.native()?)?,
            "dtype view",
        )
    }

    /// Materializes `input` into row-contiguous device storage.
    pub fn contiguous(self, input: &Array) -> Result<Array> {
        Array::from_raw(ffi::contiguous(input.native()?, self.native()?)?, "contiguous")
    }

    /// Returns a view of `input` with a new shape.
    pub fn reshape(self, input: &Array, shape: &Shape) -> Result<Array> {
        Array::from_raw(ffi::reshape(input.native()?, &shape.native()?, self.native()?)?, "reshape")
    }

    /// Permutes array axes in the specified order.
    pub fn transpose(self, input: &Array, axes: &[usize]) -> Result<Array> {
        let axes = axes
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Array::from_raw(ffi::transpose(input.native()?, &axes, self.native()?)?, "transpose")
    }

    /// Inserts size-one axes at the specified positions.
    pub fn expand_dims(self, input: &Array, axes: &[i32]) -> Result<Array> {
        Array::from_raw(ffi::expand_dims(input.native()?, axes, self.native()?)?, "expand_dims")
    }

    /// Removes one size-one axis.
    pub fn squeeze_axis(self, input: &Array, axis: i32) -> Result<Array> {
        Array::from_raw(ffi::squeeze_axis(input.native()?, axis, self.native()?)?, "squeeze_axis")
    }

    /// Applies the logistic sigmoid elementwise.
    pub fn sigmoid(self, input: &Array) -> Result<Array> {
        Array::from_raw(ffi::sigmoid(input.native()?, self.native()?)?, "sigmoid")
    }

    /// Computes `sigmoid(gate) * input` as one graph operation.
    pub fn sigmoid_multiply(self, gate: &Array, input: &Array) -> Result<Array> {
        Array::from_raw(
            ffi::sigmoid_multiply(gate.native()?, input.native()?, self.native()?)?,
            "sigmoid_multiply",
        )
    }

    /// Applies the `SiLU` activation elementwise.
    pub fn silu(self, input: &Array) -> Result<Array> {
        Array::from_raw(ffi::silu(input.native()?, self.native()?)?, "silu")
    }

    /// Applies the hyperbolic tangent elementwise.
    pub fn tanh(self, input: &Array) -> Result<Array> {
        Array::from_raw(ffi::tanh(input.native()?, self.native()?)?, "tanh")
    }

    pub(crate) fn native(self) -> Result<&'stream ffi::Stream> {
        match self.stream {
            GraphStream::Public(stream) => stream.native(),
            GraphStream::Native(stream) => Ok(stream),
        }
    }
}

impl std::fmt::Debug for Graph<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Graph").finish_non_exhaustive()
    }
}
