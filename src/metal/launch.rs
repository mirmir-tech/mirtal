use crate::{DType, Shape};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Shape and dtype requested for one Metal kernel output.
pub struct OutputSpec {
    /// Output array shape.
    pub shape: Shape,
    /// Output array dtype.
    pub dtype: DType,
}

impl OutputSpec {
    #[must_use]
    /// Creates an output specification.
    pub const fn new(shape: Shape, dtype: DType) -> Self {
        Self { shape, dtype }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A named Metal function-template argument.
pub struct TemplateArg {
    /// Template parameter name.
    pub name: &'static str,
    /// Value supplied for the parameter.
    pub value: TemplateValue,
}

impl TemplateArg {
    #[must_use]
    /// Creates a signed-integer template argument.
    pub const fn int(name: &'static str, value: i32) -> Self {
        Self { name, value: TemplateValue::Int(value) }
    }

    #[must_use]
    /// Creates a Boolean template argument.
    pub const fn bool(name: &'static str, value: bool) -> Self {
        Self { name, value: TemplateValue::Bool(value) }
    }

    #[must_use]
    /// Creates a dtype template argument.
    pub const fn dtype(name: &'static str, value: DType) -> Self {
        Self { name, value: TemplateValue::DType(value) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A value accepted by a Metal function-template parameter.
pub enum TemplateValue {
    /// Signed integer template value.
    Int(i32),
    /// Boolean template value.
    Bool(bool),
    /// Dtype template value.
    DType(DType),
}

#[derive(Debug, Clone)]
/// Geometry and template values for one Metal kernel launch.
pub struct Dispatch {
    /// Number of threads in the three-dimensional launch grid.
    pub grid: [usize; 3],
    /// Threads per threadgroup in each dimension.
    pub threadgroup: [usize; 3],
    /// Named function-template arguments.
    pub templates: Vec<TemplateArg>,
    /// Optional initial value used to allocate output buffers.
    pub init_value: Option<f32>,
    /// Enables verbose MLX Metal diagnostics.
    pub verbose: bool,
}

impl Dispatch {
    #[must_use]
    /// Creates a launch with no template arguments or output initializer.
    pub const fn new(grid: [usize; 3], threadgroup: [usize; 3]) -> Self {
        Self {
            grid,
            threadgroup,
            templates: Vec::new(),
            init_value: None,
            verbose: false,
        }
    }

    #[must_use]
    /// Appends function-template arguments to the launch.
    pub fn templates(mut self, templates: impl IntoIterator<Item = TemplateArg>) -> Self {
        self.templates.extend(templates);
        self
    }

    #[must_use]
    /// Sets the value used to initialize output buffers.
    pub const fn init_value(mut self, value: f32) -> Self {
        self.init_value = Some(value);
        self
    }

    #[must_use]
    /// Enables or disables verbose MLX Metal diagnostics.
    pub const fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}
