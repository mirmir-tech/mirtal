use mirtal_sys::ffi;

use super::{GatherQmmOptions, array};
use crate::{Array, Error, Graph, Result};

#[derive(Debug, Clone, Copy)]
/// Borrowed arrays that form an OCP MXFP4 tensor.
pub struct MxFp4<'array> {
    /// Eight packed E2M1 values per 32-bit word.
    pub weight: &'array Array,
    /// One E8M0 scale exponent per 32-value block.
    pub scales: &'array Array,
}

#[derive(Debug, Clone)]
/// Owned arrays that form an OCP MXFP4 tensor.
pub struct MxFp4Arrays {
    /// Eight packed E2M1 values per 32-bit word.
    pub weight: Array,
    /// One E8M0 scale exponent per 32-value block.
    pub scales: Array,
}

impl MxFp4Arrays {
    #[must_use]
    /// Borrows both component arrays as an [`MxFp4`] value.
    pub const fn as_ref(&self) -> MxFp4<'_> {
        MxFp4 {
            weight: &self.weight,
            scales: &self.scales,
        }
    }
}

impl Graph<'_> {
    /// Quantizes complete 32-value blocks into OCP MXFP4 storage.
    pub fn quantize_mxfp4(self, input: &Array) -> Result<MxFp4Arrays> {
        let outputs = ffi::quantize_mxfp4(input.native()?, self.native()?)?;
        let outputs = outputs.as_ref().ok_or(Error::NullHandle("MXFP4 arrays"))?;
        let actual = ffi::arrays_len(outputs);
        if actual != 2 {
            return Err(Error::Arity {
                operation: "quantize MXFP4",
                expected: 2,
                actual,
            });
        }
        Ok(MxFp4Arrays {
            weight: array(outputs, 0, "MXFP4 weight")?,
            scales: array(outputs, 1, "MXFP4 scales")?,
        })
    }

    /// Multiplies an array by an OCP MXFP4 matrix.
    pub fn mxfp4_matmul(
        self,
        input: &Array,
        quantized: MxFp4<'_>,
        transpose: bool,
    ) -> Result<Array> {
        Array::from_raw(
            ffi::mxfp4_matmul(
                input.native()?,
                quantized.weight.native()?,
                quantized.scales.native()?,
                transpose,
                self.native()?,
            )?,
            "MXFP4 matmul",
        )
    }

    /// Selects OCP MXFP4 matrices by index and multiplies them by `input`.
    pub fn gather_mxfp4(
        self,
        input: &Array,
        quantized: MxFp4<'_>,
        rhs_indices: &Array,
        options: GatherQmmOptions,
    ) -> Result<Array> {
        Array::from_raw(
            ffi::gather_mxfp4(
                input.native()?,
                quantized.weight.native()?,
                quantized.scales.native()?,
                rhs_indices.native()?,
                options.transpose,
                options.sorted_indices,
                self.native()?,
            )?,
            "gather MXFP4",
        )
    }
}
