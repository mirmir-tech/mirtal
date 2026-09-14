use cxx::SharedPtr;
use mirtal_sys::ffi;

use super::{FrequencyRopeOptions, RopeOptions, dimensions, positive};
use crate::{Array, Graph, Result};

impl Graph<'_> {
    /// Applies `RoPE` using integer device offsets, one per leading batch row.
    ///
    /// A scalar or one-element offset array broadcasts to every row. Offsets
    /// remain on device; unlike scalar `RoPE`, batch axes must not be folded.
    pub fn rope_batched(self, input: &Array, options: RopeOptions<&Array>) -> Result<Array> {
        positive(options.scale, "scale")?;
        if let Some(base) = options.base {
            positive(base, "base")?;
        }
        let native = ffi::NativeRopeOptions {
            dimensions: dimensions(options.dimensions)?,
            traditional: options.traditional,
            has_base: options.base.is_some(),
            base: options.base.unwrap_or_default(),
            scale: options.scale,
            offset: 0,
        };
        Array::from_raw(
            ffi::rope_batched(
                input.native()?,
                &native,
                options.offset.native()?,
                SharedPtr::null(),
                self.native()?,
            )?,
            "batched RoPE",
        )
    }

    /// Applies `RoPE` with device frequencies and integer per-batch offsets.
    ///
    /// Offset broadcasting follows [`Graph::rope_batched`]. No input is read
    /// back or synchronized by this operation.
    pub fn rope_with_frequencies_batched(
        self,
        input: &Array,
        frequencies: &Array,
        options: FrequencyRopeOptions<&Array>,
    ) -> Result<Array> {
        let native = ffi::NativeRopeOptions {
            dimensions: dimensions(options.dimensions)?,
            traditional: options.traditional,
            has_base: false,
            base: 0.0,
            scale: 1.0,
            offset: 0,
        };
        Array::from_raw(
            ffi::rope_batched(
                input.native()?,
                &native,
                options.offset.native()?,
                frequencies.raw_clone(),
                self.native()?,
            )?,
            "batched frequency RoPE",
        )
    }
}
