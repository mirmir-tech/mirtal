use mirtal_sys::ffi;

use crate::{Array, Error, Graph, Result, Shape};

mod batched;

#[derive(Debug, Clone, Copy)]
/// Parameters for rotary position encoding with generated frequencies.
pub struct RopeOptions<Offset = usize> {
    /// Number of trailing feature dimensions to rotate; must be positive and even.
    pub dimensions: usize,
    /// Whether to use the traditional pairing layout.
    pub traditional: bool,
    /// Optional frequency base; uses the MLX default when absent.
    pub base: Option<f32>,
    /// Positive multiplier applied to token positions.
    pub scale: f32,
    /// Starting position, or a device array of per-batch positions for batched `RoPE`.
    pub offset: Offset,
}

#[derive(Debug, Clone, Copy)]
/// Parameters for rotary position encoding with supplied frequencies.
pub struct FrequencyRopeOptions<Offset = usize> {
    /// Number of trailing feature dimensions to rotate; must be positive and even.
    pub dimensions: usize,
    /// Whether to use the traditional pairing layout.
    pub traditional: bool,
    /// Starting position, or a device array of per-batch positions for batched `RoPE`.
    pub offset: Offset,
}

impl Graph<'_> {
    /// Applies rotary position encoding using generated frequencies.
    pub fn rope(self, input: &Array, options: RopeOptions) -> Result<Array> {
        let dimensions = dimensions(options.dimensions)?;
        positive(options.scale, "scale")?;
        if let Some(base) = options.base {
            positive(base, "base")?;
        }
        let native = ffi::NativeRopeOptions {
            dimensions,
            traditional: options.traditional,
            has_base: options.base.is_some(),
            base: options.base.unwrap_or_default(),
            scale: options.scale,
            offset: i32::try_from(options.offset)?,
        };
        let normalized = single_token_layout(self, input)?;
        let input = normalized.as_ref().map_or(input, |(array, _)| array);
        let output = Array::from_raw(ffi::rope(input.native()?, &native, self.native()?)?, "RoPE")?;
        restore_layout(self, output, normalized)
    }

    /// Applies rotary position encoding using a device-resident frequency array.
    pub fn rope_with_frequencies(
        self,
        input: &Array,
        frequencies: &Array,
        options: FrequencyRopeOptions,
    ) -> Result<Array> {
        let normalized = single_token_layout(self, input)?;
        let input = normalized.as_ref().map_or(input, |(array, _)| array);
        let output = Array::from_raw(
            ffi::rope_with_frequencies(
                input.native()?,
                dimensions(options.dimensions)?,
                options.traditional,
                frequencies.native()?,
                i32::try_from(options.offset)?,
                self.native()?,
            )?,
            "frequency RoPE",
        )?;
        restore_layout(self, output, normalized)
    }
}

fn single_token_layout(graph: Graph<'_>, input: &Array) -> Result<Option<(Array, Shape)>> {
    let shape = input.shape()?;
    let dims = shape.dimensions();
    if dims.len() < 3 || dims[0] <= 1 || dims[dims.len() - 2] != 1 {
        return Ok(None);
    }
    // Scalar offsets are shared by all leading axes. Fold them into heads so
    // MLX 0.31.2's single-token kernel (fixed upstream in #3498) visits every row.
    // This is a lazy layout change, without host reads or per-row launches.
    let heads = dims[..dims.len() - 2]
        .iter()
        .try_fold(1_usize, |count, dim| count.checked_mul(*dim).ok_or(Error::ShapeOverflow))?;
    let normalized = Shape::new([1, heads, 1, dims[dims.len() - 1]])?;
    Ok(Some((graph.reshape(input, &normalized)?, shape)))
}

fn restore_layout(
    graph: Graph<'_>,
    output: Array,
    normalized: Option<(Array, Shape)>,
) -> Result<Array> {
    match normalized {
        Some((_, shape)) => graph.reshape(&output, &shape),
        None => Ok(output),
    }
}

fn dimensions(value: usize) -> Result<i32> {
    if value == 0 || !value.is_multiple_of(2) {
        return Err(invalid("dimensions must be positive and even"));
    }
    Ok(i32::try_from(value)?)
}

fn positive(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("{name} must be finite and positive")));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRope(message.into())
}
