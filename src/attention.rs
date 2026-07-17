use cxx::SharedPtr;
use mirtal_sys::ffi;

use crate::{Array, Error, Graph, Result};

#[derive(Debug, Default, Clone, Copy)]
/// Mask policy for scaled dot-product attention.
pub enum AttentionMask<'array> {
    #[default]
    /// Applies no attention mask.
    None,
    /// Applies MLX's causal attention mask.
    Causal,
    /// Uses a device-resident tensor as the attention mask.
    Array(&'array Array),
}

#[derive(Debug, Clone, Copy)]
/// Options for scaled dot-product attention.
pub struct ScaledDotProductAttention<'array> {
    /// Multiplier applied to query-key scores.
    pub scale: f32,
    /// Mask policy applied before softmax.
    pub mask: AttentionMask<'array>,
    /// Optional device-resident attention-sink values.
    pub sinks: Option<&'array Array>,
}

impl Graph<'_> {
    /// Computes scaled dot-product attention on this graph's stream.
    pub fn scaled_dot_product_attention(
        self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        options: ScaledDotProductAttention<'_>,
    ) -> Result<Array> {
        if !options.scale.is_finite() || options.scale <= 0.0 {
            return Err(Error::InvalidAttention("scale must be finite and positive".into()));
        }
        let (mask_kind, mask) = match options.mask {
            AttentionMask::None => (0, SharedPtr::null()),
            AttentionMask::Causal => (1, SharedPtr::null()),
            AttentionMask::Array(mask) => (2, mask.raw_clone()),
        };
        let sinks = options.sinks.map_or_else(SharedPtr::null, Array::raw_clone);
        let native = ffi::NativeAttentionOptions { scale: options.scale, mask_kind };
        Array::from_raw(
            ffi::sdpa(
                queries.native()?,
                keys.native()?,
                values.native()?,
                &native,
                mask,
                sinks,
                self.native()?,
            )?,
            "scaled dot-product attention",
        )
    }
}
