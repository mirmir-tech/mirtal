use mirtal_sys::ffi;

use crate::{Array, Error, Graph, Result};

#[derive(Debug, Clone, Copy)]
/// Parameters for rotary position encoding with generated frequencies.
pub struct RopeOptions {
    /// Number of trailing feature dimensions to rotate; must be positive and even.
    pub dimensions: usize,
    /// Whether to use the traditional pairing layout.
    pub traditional: bool,
    /// Optional frequency base; uses the MLX default when absent.
    pub base: Option<f32>,
    /// Positive multiplier applied to token positions.
    pub scale: f32,
    /// Starting token-position offset.
    pub offset: usize,
}

#[derive(Debug, Clone, Copy)]
/// Parameters for rotary position encoding with supplied frequencies.
pub struct FrequencyRopeOptions {
    /// Number of trailing feature dimensions to rotate; must be positive and even.
    pub dimensions: usize,
    /// Whether to use the traditional pairing layout.
    pub traditional: bool,
    /// Starting token-position offset.
    pub offset: usize,
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
        Array::from_raw(ffi::rope(input.native()?, &native, self.native()?)?, "RoPE")
    }

    /// Applies rotary position encoding using a device-resident frequency array.
    pub fn rope_with_frequencies(
        self,
        input: &Array,
        frequencies: &Array,
        options: FrequencyRopeOptions,
    ) -> Result<Array> {
        Array::from_raw(
            ffi::rope_with_frequencies(
                input.native()?,
                dimensions(options.dimensions)?,
                options.traditional,
                frequencies.native()?,
                i32::try_from(options.offset)?,
                self.native()?,
            )?,
            "frequency RoPE",
        )
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
