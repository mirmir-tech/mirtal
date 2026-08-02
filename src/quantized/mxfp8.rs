use mirtal_sys::ffi;

use super::array;
use crate::{Array, Error, Graph, Result};

#[derive(Debug, Clone, Copy)]
/// Borrowed arrays that form an OCP MXFP8 tensor.
pub struct MxFp8<'array> {
    /// Four packed E4M3 values per 32-bit word.
    pub weight: &'array Array,
    /// One E8M0 scale exponent per 32-value block.
    pub scales: &'array Array,
}

#[derive(Debug, Clone)]
/// Owned arrays that form an OCP MXFP8 tensor.
pub struct MxFp8Arrays {
    /// Four packed E4M3 values per 32-bit word.
    pub weight: Array,
    /// One E8M0 scale exponent per 32-value block.
    pub scales: Array,
}

impl MxFp8Arrays {
    #[must_use]
    /// Borrows both component arrays as an [`MxFp8`] value.
    pub const fn as_ref(&self) -> MxFp8<'_> {
        MxFp8 {
            weight: &self.weight,
            scales: &self.scales,
        }
    }
}

impl Graph<'_> {
    /// Quantizes complete 32-value blocks into OCP MXFP8 storage.
    pub fn quantize_mxfp8(self, input: &Array) -> Result<MxFp8Arrays> {
        let outputs = ffi::quantize_mxfp8(input.native()?, self.native()?)?;
        let outputs = outputs.as_ref().ok_or(Error::NullHandle("MXFP8 arrays"))?;
        let actual = ffi::arrays_len(outputs);
        if actual != 2 {
            return Err(Error::Arity {
                operation: "quantize MXFP8",
                expected: 2,
                actual,
            });
        }
        Ok(MxFp8Arrays {
            weight: array(outputs, 0, "MXFP8 weight")?,
            scales: array(outputs, 1, "MXFP8 scales")?,
        })
    }

    /// Multiplies an array by an OCP MXFP8 matrix.
    pub fn mxfp8_matmul(
        self,
        input: &Array,
        quantized: MxFp8<'_>,
        transpose: bool,
    ) -> Result<Array> {
        Array::from_raw(
            ffi::mxfp8_matmul(
                input.native()?,
                quantized.weight.native()?,
                quantized.scales.native()?,
                transpose,
                self.native()?,
            )?,
            "MXFP8 matmul",
        )
    }

    /// Reconstructs BF16 values from OCP MXFP8 components.
    pub fn dequantize_mxfp8(self, quantized: MxFp8<'_>) -> Result<Array> {
        Array::from_raw(
            ffi::dequantize_mxfp8(
                quantized.weight.native()?,
                quantized.scales.native()?,
                self.native()?,
            )?,
            "dequantize MXFP8",
        )
    }
}
