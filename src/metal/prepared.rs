use cxx::UniquePtr;
use mirtal_sys::ffi;

use super::{
    DTypeConstraint, Dispatch, MetalKernel, OutputSpec, TemplateArg, native_launch, output_array,
    validate_inputs,
};
use crate::{Array, Error, Result, Stream};

/// A cached ordinary Metal launch with fixed output and dispatch specifications.
///
/// MLX still creates its lazy custom primitive for each call.
pub struct PreparedMetalKernel<const INPUTS: usize, const OUTPUTS: usize> {
    raw: UniquePtr<ffi::PreparedMetal>,
    name: &'static str,
    input_dtypes: [DTypeConstraint; INPUTS],
    templates: Vec<TemplateArg>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> MetalKernel<INPUTS, OUTPUTS> {
    /// Caches invariant output and launch specifications for repeated dispatches.
    pub fn prepare(
        &self,
        outputs: &[OutputSpec; OUTPUTS],
        dispatch: &Dispatch,
    ) -> Result<PreparedMetalKernel<INPUTS, OUTPUTS>> {
        self.validate_plan(outputs, dispatch)?;
        let launch = native_launch(outputs, dispatch)?;
        let raw = ffi::new_prepared_metal(
            self.native()?,
            launch.as_ref().ok_or(Error::NullHandle("Metal launch"))?,
        )?;
        if raw.is_null() {
            return Err(Error::NullHandle("prepared Metal kernel"));
        }
        Ok(PreparedMetalKernel {
            raw,
            name: self.name,
            input_dtypes: self.input_dtypes,
            templates: dispatch.templates.clone(),
        })
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> PreparedMetalKernel<INPUTS, OUTPUTS> {
    /// Dispatches with new inputs using the cached launch specification.
    pub fn dispatch(
        &mut self,
        stream: &Stream,
        inputs: [&Array; INPUTS],
    ) -> Result<[Array; OUTPUTS]> {
        validate_inputs(self.name, self.input_dtypes, &self.templates, inputs)?;
        for (index, input) in inputs.into_iter().enumerate() {
            ffi::prepared_metal_set_input(self.raw.pin_mut(), index, input.native()?)?;
        }
        let outputs = ffi::prepared_metal_dispatch(self.raw.pin_mut(), stream.native()?)?;
        output_array(&outputs)
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for PreparedMetalKernel<INPUTS, OUTPUTS>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedMetalKernel")
            .field("name", &self.name)
            .field("inputs", &INPUTS)
            .field("outputs", &OUTPUTS)
            .finish_non_exhaustive()
    }
}
