mod aliasing;
mod contract;
mod launch;
mod library;
mod prepared;
mod source;

pub use contract::{DTypeConstraint, KernelDescriptor, TemplateKind, TemplateParameter};
use cxx::UniquePtr;
pub use launch::{Dispatch, OutputSpec, TemplateArg, TemplateValue};
pub use library::{MetalLibrary, MetalLibraryDescriptor};
use mirtal_sys::ffi;
pub use prepared::PreparedMetalKernel;
pub use source::MetalSource;

use crate::{Array, Error, Graph, Result, Stream};

/// A checked MLX custom Metal kernel with statically known buffer arities.
pub struct MetalKernel<const INPUTS: usize, const OUTPUTS: usize> {
    raw: UniquePtr<ffi::MetalKernel>,
    name: &'static str,
    input_dtypes: [DTypeConstraint; INPUTS],
    output_dtypes: [DTypeConstraint; OUTPUTS],
    templates: &'static [TemplateParameter],
}

impl<const INPUTS: usize, const OUTPUTS: usize> MetalKernel<INPUTS, OUTPUTS> {
    /// Constructs a kernel from a source and buffer contract validated at build time.
    pub fn new(descriptor: KernelDescriptor<INPUTS, OUTPUTS>) -> Result<Self> {
        contract::validate_descriptor(&descriptor)?;
        let input_names = descriptor.input_names.join("\x1f");
        let output_names = descriptor.output_names.join("\x1f");
        let raw = ffi::new_metal_kernel(
            descriptor.name,
            &input_names,
            &output_names,
            descriptor.source.code(),
            descriptor.header.code(),
            descriptor.row_contiguous,
            descriptor.atomic_outputs,
        )?;
        if raw.is_null() {
            return Err(Error::NullHandle("Metal kernel"));
        }
        Ok(Self {
            raw,
            name: descriptor.name,
            input_dtypes: descriptor.input_dtypes,
            output_dtypes: descriptor.output_dtypes,
            templates: descriptor.templates,
        })
    }

    /// Launches the kernel on an explicit stream.
    pub fn dispatch(
        &self,
        stream: &Stream,
        inputs: [&Array; INPUTS],
        outputs: &[OutputSpec; OUTPUTS],
        dispatch: &Dispatch,
    ) -> Result<[Array; OUTPUTS]> {
        self.dispatch_native(stream.native()?, inputs, outputs, dispatch)
    }

    /// Adds a lazy kernel launch to an existing graph.
    pub fn dispatch_graph(
        &self,
        graph: Graph<'_>,
        inputs: [&Array; INPUTS],
        outputs: &[OutputSpec; OUTPUTS],
        dispatch: &Dispatch,
    ) -> Result<[Array; OUTPUTS]> {
        self.dispatch_native(graph.native()?, inputs, outputs, dispatch)
    }

    fn dispatch_native(
        &self,
        stream: &ffi::Stream,
        inputs: [&Array; INPUTS],
        outputs: &[OutputSpec; OUTPUTS],
        dispatch: &Dispatch,
    ) -> Result<[Array; OUTPUTS]> {
        self.validate_dispatch(inputs, outputs, dispatch)?;
        let mut native_inputs = ffi::new_arrays();
        for input in inputs {
            ffi::arrays_push(native_inputs.pin_mut(), input.native()?);
        }
        let launch = native_launch(outputs, dispatch)?;
        let values = ffi::metal_dispatch(
            self.raw.as_ref().ok_or(Error::NullHandle("Metal kernel"))?,
            native_inputs.as_ref().ok_or(Error::NullHandle("Metal inputs"))?,
            launch.as_ref().ok_or(Error::NullHandle("Metal launch"))?,
            stream,
        )?;
        output_array(&values)
    }

    fn validate_dispatch(
        &self,
        inputs: [&Array; INPUTS],
        outputs: &[OutputSpec; OUTPUTS],
        dispatch: &Dispatch,
    ) -> Result<()> {
        self.validate_plan(outputs, dispatch)?;
        validate_inputs(self.name, self.input_dtypes, &dispatch.templates, inputs)?;
        Ok(())
    }

    fn validate_plan(&self, outputs: &[OutputSpec; OUTPUTS], dispatch: &Dispatch) -> Result<()> {
        contract::validate_templates(self.name, self.templates, &dispatch.templates)?;
        contract::validate_outputs(self.name, outputs, self.output_dtypes, &dispatch.templates)
    }

    pub(crate) fn native(&self) -> Result<&ffi::MetalKernel> {
        self.raw.as_ref().ok_or(Error::NullHandle("Metal kernel"))
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug for MetalKernel<INPUTS, OUTPUTS> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalKernel")
            .field("name", &self.name)
            .field("inputs", &INPUTS)
            .field("outputs", &OUTPUTS)
            .finish_non_exhaustive()
    }
}

fn native_dimensions(values: [usize; 3]) -> Result<[i32; 3]> {
    if values.contains(&0) {
        return Err(Error::InvalidDispatch(
            "grid and threadgroup dimensions must be positive".into(),
        ));
    }
    Ok([i32::try_from(values[0])?, i32::try_from(values[1])?, i32::try_from(values[2])?])
}

fn add_templates(launch: &mut UniquePtr<ffi::MetalLaunch>, templates: &[TemplateArg]) {
    for template in templates {
        match template.value {
            TemplateValue::Int(value) => {
                ffi::metal_launch_add_template_int(launch.pin_mut(), template.name, value);
            },
            TemplateValue::Bool(value) => {
                ffi::metal_launch_add_template_bool(launch.pin_mut(), template.name, value);
            },
            TemplateValue::DType(value) => {
                ffi::metal_launch_add_template_dtype(launch.pin_mut(), template.name, value as u8);
            },
        }
    }
}

fn native_launch<const OUTPUTS: usize>(
    outputs: &[OutputSpec; OUTPUTS],
    dispatch: &Dispatch,
) -> Result<UniquePtr<ffi::MetalLaunch>> {
    let [grid_x, grid_y, grid_z] = native_dimensions(dispatch.grid)?;
    let [group_x, group_y, group_z] = native_dimensions(dispatch.threadgroup)?;
    let mut launch =
        ffi::new_metal_launch(grid_x, grid_y, grid_z, group_x, group_y, group_z, dispatch.verbose);
    for output in outputs {
        ffi::metal_launch_add_output(launch.pin_mut(), &output.shape.native()?, output.dtype as u8);
    }
    add_templates(&mut launch, &dispatch.templates);
    if let Some(value) = dispatch.init_value {
        ffi::metal_launch_set_init(launch.pin_mut(), value);
    }
    Ok(launch)
}

fn validate_inputs<const INPUTS: usize>(
    name: &str,
    constraints: [DTypeConstraint; INPUTS],
    templates: &[TemplateArg],
    inputs: [&Array; INPUTS],
) -> Result<()> {
    for (index, (input, constraint)) in inputs.into_iter().zip(constraints).enumerate() {
        contract::validate_dtype(
            name,
            contract::TensorLabel::Input(index),
            input.dtype()?,
            constraint,
            templates,
        )?;
    }
    Ok(())
}

fn output_array<const OUTPUTS: usize>(values: &UniquePtr<ffi::Arrays>) -> Result<[Array; OUTPUTS]> {
    let values = values.as_ref().ok_or(Error::NullHandle("Metal outputs"))?;
    let actual = ffi::arrays_len(values);
    if actual != OUTPUTS {
        return Err(Error::Arity {
            operation: "Metal outputs",
            expected: OUTPUTS,
            actual,
        });
    }
    let mut output = std::array::from_fn(|_| None);
    for (index, value) in output.iter_mut().enumerate() {
        *value = Some(Array::from_raw(ffi::arrays_get(values, index)?, "Metal output")?);
    }
    Ok(output.map(|value| {
        let Some(value) = value else {
            unreachable!("validated Metal output arity")
        };
        value
    }))
}
pub use aliasing::{AliasingDispatch, MetalFunction, PreparedAliasing, StrideBinding};
