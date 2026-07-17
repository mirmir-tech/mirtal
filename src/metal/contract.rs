use super::{OutputSpec, TemplateArg, TemplateValue};
use crate::{DType, Error, Result};

#[derive(Debug, Clone, Copy)]
pub(super) enum TensorLabel {
    Input(usize),
    Output(usize),
}

impl std::fmt::Display for TensorLabel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(index) => write!(formatter, "input {index}"),
            Self::Output(index) => write!(formatter, "output {index}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A dtype requirement declared for a Metal kernel buffer.
pub enum DTypeConstraint {
    /// Requires one exact dtype.
    Exact(DType),
    /// Accepts any floating-point dtype supported by mirtal.
    Float,
    /// Uses the dtype supplied by the named template argument.
    Template(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The value category accepted by a Metal template parameter.
pub enum TemplateKind {
    /// A signed integer value.
    Int,
    /// A Boolean value.
    Bool,
    /// A mirtal dtype value.
    DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Declares one named template parameter accepted by a Metal kernel.
pub struct TemplateParameter {
    /// Parameter name used by the Metal source.
    pub name: &'static str,
    /// Required value category.
    pub kind: TemplateKind,
}

#[derive(Debug, Clone, Copy)]
/// Complete checked contract used to construct a typed Metal kernel.
pub struct KernelDescriptor<const INPUTS: usize, const OUTPUTS: usize> {
    /// Metal function name.
    pub name: &'static str,
    /// Names of input buffer parameters in binding order.
    pub input_names: [&'static str; INPUTS],
    /// Dtype requirements for input buffers.
    pub input_dtypes: [DTypeConstraint; INPUTS],
    /// Names of output buffer parameters in binding order.
    pub output_names: [&'static str; OUTPUTS],
    /// Dtype requirements for output buffers.
    pub output_dtypes: [DTypeConstraint; OUTPUTS],
    /// Template parameters required at dispatch time.
    pub templates: &'static [TemplateParameter],
    /// Checked Metal kernel source.
    pub source: super::MetalSource,
    /// Optional checked source prepended as a header.
    pub header: super::MetalSource,
    /// Declares that all array buffers must be row-contiguous.
    pub row_contiguous: bool,
    /// Declares that output buffers are accessed atomically.
    pub atomic_outputs: bool,
}

pub(super) fn validate_descriptor<const INPUTS: usize, const OUTPUTS: usize>(
    descriptor: &KernelDescriptor<INPUTS, OUTPUTS>,
) -> Result<()> {
    let invalid_name = descriptor
        .input_names
        .iter()
        .chain(&descriptor.output_names)
        .any(|name| name.is_empty() || name.contains('\x1f'));
    if descriptor.name.is_empty() || descriptor.source.code().is_empty() || invalid_name {
        return Err(Error::InvalidKernel(format!(
            "{} from {} contains an empty or reserved field",
            descriptor.name,
            descriptor.source.origin()
        )));
    }
    Ok(())
}

pub(super) fn validate_templates(
    kernel: &str,
    parameters: &[TemplateParameter],
    arguments: &[TemplateArg],
) -> Result<()> {
    if parameters.len() != arguments.len() {
        return Err(Error::InvalidDispatch(format!(
            "{kernel} expects {} template arguments, received {}",
            parameters.len(),
            arguments.len()
        )));
    }
    for parameter in parameters {
        let matches = arguments.iter().filter(|argument| argument.name == parameter.name).count();
        if matches != 1 {
            return Err(Error::InvalidDispatch(format!(
                "{kernel} requires exactly one `{}` template argument",
                parameter.name
            )));
        }
        let argument = arguments
            .iter()
            .find(|argument| argument.name == parameter.name)
            .ok_or_else(|| Error::InvalidDispatch("validated template disappeared".into()))?;
        if template_kind(argument.value) != parameter.kind {
            return Err(Error::InvalidDispatch(format!(
                "{kernel} template `{}` has the wrong kind",
                parameter.name
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_dtype(
    kernel: &str,
    label: TensorLabel,
    actual: DType,
    constraint: DTypeConstraint,
    templates: &[TemplateArg],
) -> Result<()> {
    let expected = match constraint {
        DTypeConstraint::Exact(expected) => expected,
        DTypeConstraint::Float => {
            if matches!(actual, DType::Float16 | DType::Bfloat16 | DType::Float32) {
                return Ok(());
            }
            return Err(Error::InvalidDispatch(format!(
                "{kernel} {label} dtype {actual:?} is not floating point"
            )));
        },
        DTypeConstraint::Template(name) => template_dtype(name, templates)?,
    };
    if actual != expected {
        return Err(Error::InvalidDispatch(format!(
            "{kernel} {label} dtype {actual:?} does not match declared {expected:?}"
        )));
    }
    Ok(())
}

pub(super) fn validate_outputs<const OUTPUTS: usize>(
    kernel: &str,
    outputs: &[OutputSpec; OUTPUTS],
    constraints: [DTypeConstraint; OUTPUTS],
    templates: &[TemplateArg],
) -> Result<()> {
    for (index, (output, constraint)) in outputs.iter().zip(constraints).enumerate() {
        validate_dtype(kernel, TensorLabel::Output(index), output.dtype, constraint, templates)?;
    }
    Ok(())
}

fn template_dtype(name: &str, templates: &[TemplateArg]) -> Result<DType> {
    templates
        .iter()
        .find_map(|argument| match argument {
            TemplateArg {
                name: actual,
                value: TemplateValue::DType(dtype),
            } if *actual == name => Some(*dtype),
            _ => None,
        })
        .ok_or_else(|| Error::InvalidDispatch(format!("missing dtype template `{name}`")))
}

const fn template_kind(value: TemplateValue) -> TemplateKind {
    match value {
        TemplateValue::Int(_) => TemplateKind::Int,
        TemplateValue::Bool(_) => TemplateKind::Bool,
        TemplateValue::DType(_) => TemplateKind::DType,
    }
}
