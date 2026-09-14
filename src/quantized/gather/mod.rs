use cxx::SharedPtr;
use mirtal_sys::quantized::ffi;

use super::{GatherQmmOptions, MxFp4, Quantized, native_options};
use crate::{Array, Graph, Result};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, Default)]
/// Device-resident matrix selectors for the two `GatherQMM` operands.
///
/// Indices address flattened leading (batch) dimensions, not matrix rows.
/// The two index shapes broadcast together. An omitted selector follows MLX's
/// implicit batch indexing; omitting both reduces to ordinary quantized matmul.
/// Values must be valid matrix indices. This API does not read them on the host.
pub struct GatherIndices<'array> {
    /// Selects matrices from the left-hand input.
    pub lhs: Option<&'array Array>,
    /// Selects matrices from the quantized right-hand bank.
    pub rhs: Option<&'array Array>,
}

impl Graph<'_> {
    /// Selects quantized right-hand matrices; preserves implicit left indexing.
    pub fn gather_qmm(
        self,
        input: &Array,
        quantized: Quantized<'_>,
        rhs_indices: &Array,
        options: GatherQmmOptions,
    ) -> Result<Array> {
        self.gather_qmm_with_indices(
            input,
            quantized,
            GatherIndices { lhs: None, rhs: Some(rhs_indices) },
            options,
        )
    }

    /// Selects matrices independently from both affine-quantized operands.
    ///
    /// Remains lazy and uses this graph's explicit stream. The sorted hint is
    /// forwarded unchanged; MLX may use a different kernel with explicit LHS
    /// indices even when the RHS indices are sorted.
    pub fn gather_qmm_with_indices(
        self,
        input: &Array,
        quantized: Quantized<'_>,
        indices: GatherIndices<'_>,
        options: GatherQmmOptions,
    ) -> Result<Array> {
        let native_options =
            native_options(quantized.format, options.transpose, options.sorted_indices);
        Array::from_raw(
            ffi::gather_qmm(
                input.native()?,
                quantized.weight.native()?,
                quantized.scales.native()?,
                quantized.biases.native()?,
                &indices.native(),
                &native_options,
                self.native()?,
            )?,
            "gather qmm",
        )
    }

    /// Selects MXFP4 right-hand matrices; preserves implicit left indexing.
    pub fn gather_mxfp4(
        self,
        input: &Array,
        quantized: MxFp4<'_>,
        rhs_indices: &Array,
        options: GatherQmmOptions,
    ) -> Result<Array> {
        self.gather_mxfp4_with_indices(
            input,
            quantized,
            GatherIndices { lhs: None, rhs: Some(rhs_indices) },
            options,
        )
    }

    /// Selects matrices independently from both MXFP4 operands.
    ///
    /// Has the same indexing and execution contract as
    /// [`Self::gather_qmm_with_indices`]. Scales retain OCP MXFP4 semantics.
    pub fn gather_mxfp4_with_indices(
        self,
        input: &Array,
        quantized: MxFp4<'_>,
        indices: GatherIndices<'_>,
        options: GatherQmmOptions,
    ) -> Result<Array> {
        Array::from_raw(
            ffi::gather_mxfp4(
                input.native()?,
                quantized.weight.native()?,
                quantized.scales.native()?,
                &indices.native(),
                options.transpose,
                options.sorted_indices,
                self.native()?,
            )?,
            "gather MXFP4",
        )
    }
}

impl GatherIndices<'_> {
    fn native(self) -> ffi::NativeGatherIndices {
        ffi::NativeGatherIndices {
            lhs: self.lhs.map_or_else(SharedPtr::null, Array::raw_clone),
            rhs: self.rhs.map_or_else(SharedPtr::null, Array::raw_clone),
        }
    }
}
