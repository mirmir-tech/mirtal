use cxx::UniquePtr;
use mirtal_sys::aliasing;

use super::{AliasingDispatch, MetalFunction, native_grid, output_array, validate};
use crate::{Array, Error, Result, Stream};

/// A cached aliasing Metal plan with fixed input and output arities.
///
/// The plan is mutable session-local state and must not be shared concurrently.
pub struct PreparedAliasing<const INPUTS: usize, const OUTPUTS: usize> {
    raw: UniquePtr<aliasing::ffi::AliasingPlan>,
    dispatch: AliasingDispatch,
}

impl MetalFunction<'_, '_> {
    /// Prepares this function for repeated aliasing dispatches.
    pub fn prepare_aliasing<const INPUTS: usize, const OUTPUTS: usize>(
        &self,
        dispatch: AliasingDispatch,
    ) -> Result<PreparedAliasing<INPUTS, OUTPUTS>> {
        validate(INPUTS, &dispatch)?;
        if dispatch.output_aliases.len() != OUTPUTS {
            return Err(Error::Arity {
                operation: "prepared aliasing outputs",
                expected: OUTPUTS,
                actual: dispatch.output_aliases.len(),
            });
        }
        let raw = aliasing::ffi::new_aliasing_plan(
            INPUTS,
            dispatch.constants.len(),
            &dispatch.output_aliases,
            &dispatch.stride_inputs,
            &dispatch.stride_axes,
            self.name.as_ref(),
            self.library.native()?,
        )?;
        if raw.is_null() {
            return Err(Error::NullHandle("prepared aliasing plan"));
        }
        Ok(PreparedAliasing { raw, dispatch })
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> PreparedAliasing<INPUTS, OUTPUTS> {
    /// Replaces constants and launch geometry while preserving binding arity.
    pub fn rebind(
        &mut self,
        constants: &[u32],
        grid: [usize; 3],
        threadgroup: [usize; 3],
    ) -> Result<()> {
        self.dispatch.rebind(constants, grid, threadgroup)
    }

    /// Dispatches the prepared plan and updates its cached input bindings.
    pub fn dispatch(
        &mut self,
        stream: &Stream,
        inputs: [&Array; INPUTS],
    ) -> Result<[Array; OUTPUTS]> {
        let grid = native_grid(self.dispatch.grid)?;
        let group = native_grid(self.dispatch.threadgroup)?;
        for (index, input) in inputs.into_iter().enumerate() {
            aliasing::ffi::aliasing_plan_set_input(self.raw.pin_mut(), index, input.native()?)?;
        }
        let outputs = aliasing::ffi::aliasing_plan_dispatch(
            self.raw.pin_mut(),
            &self.dispatch.constants,
            grid[0],
            grid[1],
            grid[2],
            group[0],
            group[1],
            group[2],
            stream.native()?,
        )?;
        output_array(&outputs)
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for PreparedAliasing<INPUTS, OUTPUTS>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedAliasing")
            .field("inputs", &INPUTS)
            .field("outputs", &OUTPUTS)
            .field("dispatch", &self.dispatch)
            .finish_non_exhaustive()
    }
}
