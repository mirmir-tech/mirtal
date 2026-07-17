use mirtal_sys::{ffi, ops::ffi as ops};

use crate::{Array, DType, Error, Graph, Result, Shape};

impl Graph<'_> {
    /// Creates evenly spaced values in the half-open interval `start..stop`.
    pub fn arange(self, start: f32, stop: f32, stride: f32, dtype: DType) -> Result<Array> {
        if !start.is_finite() || !stop.is_finite() || !stride.is_finite() || stride == 0.0 {
            return Err(invalid("arange requires finite bounds and a non-zero step"));
        }
        Array::from_raw(ops::arange(start, stop, stride, dtype as u8, self.native()?)?, "arange")
    }

    /// Creates an array of `shape` filled with `value`.
    pub fn full(self, shape: &Shape, value: f32, dtype: DType) -> Result<Array> {
        Array::from_raw(ops::full(&shape.native()?, value, dtype as u8, self.native()?)?, "full")
    }

    /// Joins arrays along an existing axis.
    pub fn concatenate(self, inputs: &[&Array], axis: i32) -> Result<Array> {
        let native = native_inputs(inputs, "concatenate")?;
        Array::from_raw(
            ops::concatenate(
                native.as_ref().ok_or(Error::NullHandle("concatenate inputs"))?,
                axis,
                self.native()?,
            )?,
            "concatenate",
        )
    }

    /// Joins arrays along a newly inserted axis.
    pub fn stack(self, inputs: &[&Array], axis: i32) -> Result<Array> {
        let native = native_inputs(inputs, "stack")?;
        Array::from_raw(
            ops::stack(
                native.as_ref().ok_or(Error::NullHandle("stack inputs"))?,
                axis,
                self.native()?,
            )?,
            "stack",
        )
    }

    /// Repeats elements of an array along `axis`.
    pub fn repeat(self, input: &Array, repeats: i32, axis: i32) -> Result<Array> {
        if repeats <= 0 {
            return Err(invalid("repeat count must be positive"));
        }
        Array::from_raw(ops::repeat(input.native()?, repeats, axis, self.native()?)?, "repeat")
    }

    /// Applies a one-dimensional grouped convolution.
    pub fn conv1d(
        self,
        input: &Array,
        weight: &Array,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<Array> {
        if stride <= 0 || padding < 0 || dilation <= 0 || groups <= 0 {
            return Err(invalid("conv1d parameters are invalid"));
        }
        Array::from_raw(
            ops::conv1d(
                input.native()?,
                weight.native()?,
                stride,
                padding,
                dilation,
                groups,
                self.native()?,
            )?,
            "conv1d",
        )
    }
}

fn native_inputs(inputs: &[&Array], operation: &str) -> Result<cxx::UniquePtr<ffi::Arrays>> {
    if inputs.is_empty() {
        return Err(invalid(format!("{operation} requires at least one input")));
    }
    let mut native = ffi::new_arrays();
    for input in inputs {
        ffi::arrays_push(native.pin_mut(), input.native()?);
    }
    Ok(native)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidOperation(message.into())
}
