mod prepared;

use std::borrow::Cow;

use cxx::UniquePtr;
use mirtal_sys::{aliasing, ffi};
pub use prepared::PreparedAliasing;

use super::MetalLibrary;
use crate::{Array, DType, Error, Result, Stream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Binds an input array stride to a Metal constant.
pub struct StrideBinding {
    /// Zero-based input-buffer index.
    pub input: u32,
    /// Axis whose element stride is passed to the kernel.
    pub axis: u32,
}

#[derive(Debug, Clone)]
/// Mutable-buffer bindings and geometry for an aliasing Metal dispatch.
pub struct AliasingDispatch {
    /// Input indices reused as mutable output buffers.
    pub output_aliases: Vec<u32>,
    /// Scalar `u32` values passed after buffer arguments.
    pub constants: Vec<u32>,
    /// Input indices whose strides are passed to the kernel.
    pub stride_inputs: Vec<u32>,
    /// Axes paired with `stride_inputs`.
    pub stride_axes: Vec<u32>,
    /// Number of threads in the launch grid.
    pub grid: [usize; 3],
    /// Threads per threadgroup.
    pub threadgroup: [usize; 3],
}

impl AliasingDispatch {
    #[must_use]
    /// Creates a dispatch whose outputs alias the listed inputs.
    pub fn new(output_aliases: impl IntoIterator<Item = u32>) -> Self {
        Self {
            output_aliases: output_aliases.into_iter().collect(),
            constants: Vec::new(),
            stride_inputs: Vec::new(),
            stride_axes: Vec::new(),
            grid: [1, 1, 1],
            threadgroup: [1, 1, 1],
        }
    }

    #[must_use]
    /// Appends scalar constants in kernel argument order.
    pub fn constants(mut self, values: impl IntoIterator<Item = u32>) -> Self {
        self.constants.extend(values);
        self
    }

    #[must_use]
    /// Appends input-stride bindings in kernel argument order.
    pub fn strides(mut self, values: impl IntoIterator<Item = StrideBinding>) -> Self {
        for binding in values {
            self.stride_inputs.push(binding.input);
            self.stride_axes.push(binding.axis);
        }
        self
    }

    #[must_use]
    /// Sets the three-dimensional grid size.
    pub const fn grid(mut self, value: [usize; 3]) -> Self {
        self.grid = value;
        self
    }

    #[must_use]
    /// Sets the three-dimensional threadgroup size.
    pub const fn threadgroup(mut self, value: [usize; 3]) -> Self {
        self.threadgroup = value;
        self
    }

    /// Replaces constants and launch geometry without changing binding arity.
    pub fn rebind(
        &mut self,
        constants: &[u32],
        grid: [usize; 3],
        threadgroup: [usize; 3],
    ) -> Result<()> {
        if constants.len() != self.constants.len() {
            return Err(Error::InvalidDispatch("rebound Metal constants changed arity".into()));
        }
        self.constants.copy_from_slice(constants);
        self.grid = grid;
        self.threadgroup = threadgroup;
        Ok(())
    }
}

/// A named function exported from a checked [`MetalLibrary`].
pub struct MetalFunction<'library, 'name> {
    pub(super) library: &'library MetalLibrary,
    pub(super) name: Cow<'name, str>,
}

impl MetalLibrary {
    /// Selects a non-empty named function from this library.
    pub fn export<'name>(
        &self,
        name: impl Into<Cow<'name, str>>,
    ) -> Result<MetalFunction<'_, 'name>> {
        let name = name.into();
        if name.is_empty() {
            return Err(Error::InvalidKernel("Metal export name is empty".into()));
        }
        Ok(MetalFunction { library: self, name })
    }
}

impl MetalFunction<'_, '_> {
    /// Appends mirtal's conventional suffix for a floating-point dtype.
    pub fn with_dtype_suffix(mut self, dtype: DType) -> Result<Self> {
        let suffix = match dtype {
            DType::Float32 => "f32",
            DType::Float16 => "f16",
            DType::Bfloat16 => "bf16",
            _ => return Err(Error::InvalidKernel(format!("no Metal suffix for {dtype:?}"))),
        };
        self.name = format!("{}_{suffix}", self.name).into();
        Ok(self)
    }

    /// Dispatches a function whose outputs alias mutable input buffers.
    pub fn dispatch_aliasing(
        &self,
        stream: &Stream,
        inputs: &[&Array],
        dispatch: &AliasingDispatch,
    ) -> Result<Vec<Array>> {
        let outputs = self.dispatch_native(stream, inputs, dispatch)?;
        let outputs = outputs.as_ref().ok_or(Error::NullHandle("aliasing outputs"))?;
        (0..ffi::arrays_len(outputs))
            .map(|index| Array::from_raw(ffi::arrays_get(outputs, index)?, "aliasing output"))
            .collect()
    }

    /// Dispatches an aliasing function and validates a fixed output arity.
    pub fn dispatch_aliasing_array<const OUTPUTS: usize>(
        &self,
        stream: &Stream,
        inputs: &[&Array],
        dispatch: &AliasingDispatch,
    ) -> Result<[Array; OUTPUTS]> {
        let outputs = self.dispatch_native(stream, inputs, dispatch)?;
        output_array(&outputs)
    }

    fn dispatch_native(
        &self,
        stream: &Stream,
        inputs: &[&Array],
        dispatch: &AliasingDispatch,
    ) -> Result<UniquePtr<ffi::Arrays>> {
        validate(inputs.len(), dispatch)?;
        let mut native = ffi::new_arrays();
        for input in inputs {
            ffi::arrays_push(native.pin_mut(), input.native()?);
        }
        let grid = native_grid(dispatch.grid)?;
        let group = native_grid(dispatch.threadgroup)?;
        Ok(aliasing::ffi::aliasing_dispatch(
            native.as_ref().ok_or(Error::NullHandle("aliasing inputs"))?,
            &dispatch.output_aliases,
            &dispatch.constants,
            &dispatch.stride_inputs,
            &dispatch.stride_axes,
            self.name.as_ref(),
            self.library.native()?,
            grid[0],
            grid[1],
            grid[2],
            group[0],
            group[1],
            group[2],
            stream.native()?,
        )?)
    }
}

pub(super) fn validate(input_arity: usize, dispatch: &AliasingDispatch) -> Result<()> {
    let dimensions = dispatch.grid.iter().chain(&dispatch.threadgroup);
    if input_arity == 0
        || dispatch.output_aliases.is_empty()
        || dispatch
            .output_aliases
            .iter()
            .any(|index| usize::try_from(*index).map_or(true, |index| index >= input_arity))
        || dispatch
            .stride_inputs
            .iter()
            .any(|index| usize::try_from(*index).map_or(true, |index| index >= input_arity))
        || dispatch.stride_inputs.len() != dispatch.stride_axes.len()
        || dimensions.into_iter().any(|value| *value == 0)
    {
        return Err(Error::InvalidDispatch("invalid aliasing Metal dispatch".into()));
    }
    Ok(())
}

pub(super) fn native_grid(values: [usize; 3]) -> Result<[u32; 3]> {
    Ok([u32::try_from(values[0])?, u32::try_from(values[1])?, u32::try_from(values[2])?])
}

pub(super) fn output_array<const OUTPUTS: usize>(
    values: &UniquePtr<ffi::Arrays>,
) -> Result<[Array; OUTPUTS]> {
    let values = values.as_ref().ok_or(Error::NullHandle("aliasing outputs"))?;
    let actual = ffi::arrays_len(values);
    if actual != OUTPUTS {
        return Err(Error::Arity {
            operation: "aliasing outputs",
            expected: OUTPUTS,
            actual,
        });
    }
    let mut output = std::array::from_fn(|_| None);
    for (index, value) in output.iter_mut().enumerate() {
        *value = Some(Array::from_raw(ffi::arrays_get(values, index)?, "aliasing output")?);
    }
    Ok(output.map(|value| {
        let Some(value) = value else {
            unreachable!("validated aliasing output arity")
        };
        value
    }))
}
