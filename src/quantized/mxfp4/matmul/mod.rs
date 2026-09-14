mod shape;
#[cfg(test)]
mod tests;

use std::num::NonZeroUsize;

use crate::{
    Array, DType, Dispatch, Graph, MetalKernel, MxFp4, OutputSpec, Result, Shape, Stream,
    TemplateArg,
};

crate::metal_kernel! {
    fn project {
        name: "mirtal_mxfp4_fp32_partials",
        templates: [T: dtype = bf16, O: dtype = f32, M: int = 16, N: int = 32, K: int = 64, PART: int = 32, BM: int = 16],
        inputs: [input: T, weight: u32, scales: u8], outputs: [output: O],
        source: file "src/quantized/mxfp4/matmul/project.metal",
        header: inline "#include <metal_simdgroup_matrix>",
        row_contiguous: true, atomic_outputs: false,
    }
}

crate::metal_kernel! {
    fn reduce {
        name: "mirtal_mxfp4_reduce_fp32",
        templates: [T: dtype = bf16, ELEMENTS: int = 512, PARTS: int = 2],
        inputs: [partials: f32], outputs: [output: T],
        source: file "src/quantized/mxfp4/matmul/reduce.metal",
        header: inline "", row_contiguous: true, atomic_outputs: false,
    }
}

/// Output rows processed by each MXFP4 SIMD threadgroup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MxFp4RowTile {
    /// Sixteen rows with two SIMD groups (the default).
    #[default]
    Rows16,
    /// Thirty-two rows with four SIMD groups and greater weight reuse.
    Rows32,
}

impl MxFp4RowTile {
    const fn rows(self) -> usize {
        match self {
            Self::Rows16 => 16,
            Self::Rows32 => 32,
        }
    }
}

#[derive(Debug)]
/// Reusable ordinary MXFP4 matrix multiplication with FP32 partial sums.
///
/// This explicit alternative does not change native MLX selection. It computes
/// `input @ weight.T` on Metal and returns BF16. Input and decoded weight values
/// are BF16; matrix accumulators, split-K storage and final reduction are FP32.
/// Output bias belongs after this operation. No host reads or waits are added.
pub struct MxFp4Matmul {
    row_tile: MxFp4RowTile,
    projection: MetalKernel<3, 1>,
    reduction: MetalKernel<1, 1>,
}

impl MxFp4Matmul {
    /// Creates checked kernels. Reuse this object instead of recreating it per call.
    pub fn new() -> Result<Self> {
        Ok(Self {
            row_tile: MxFp4RowTile::default(),
            projection: project()?,
            reduction: reduce()?,
        })
    }

    /// Selects a row tile explicitly; no autotuning or native default is changed.
    /// The FP32 accumulation order and split-K buffer size remain unchanged.
    #[must_use]
    pub const fn with_row_tile(mut self, tile: MxFp4RowTile) -> Self {
        self.row_tile = tile;
        self
    }

    /// Multiplies BF16 `[..., K]` by a packed U32 `[N, K/8]` matrix.
    ///
    /// Input rank must be at least two. Scales must be U8 `[N, K/32]`;
    /// dimensions must be positive and K divisible
    /// by 32. `partitions` must divide K/32. One partition writes BF16 directly;
    /// otherwise a temporary FP32 `[partitions, M, N]` array retains partial sums.
    /// Selection and memory budgeting are explicit caller decisions.
    pub fn matmul(
        &self,
        input: &Array,
        weights: MxFp4<'_>,
        partitions: NonZeroUsize,
        stream: &Stream,
    ) -> Result<Array> {
        self.matmul_graph(stream.graph(), input, weights, partitions)
    }

    /// Adds the projection to an explicit graph, including a compiled callback.
    ///
    /// Uses the same contract as [`Self::matmul`]. Compile with default,
    /// shape-specialized options: dispatch geometry depends on input dimensions.
    /// Pass input, weights and scales as graph arguments so repeated calls do not
    /// capture or retain a particular checkpoint's arrays.
    pub fn matmul_graph(
        &self,
        graph: Graph<'_>,
        input: &Array,
        weights: MxFp4<'_>,
        partitions: NonZeroUsize,
    ) -> Result<Array> {
        let layout = shape::Layout::new(input, weights, partitions.get())?;
        let split = partitions.get();
        let rows = self.row_tile.rows();
        let threads = rows * 4;
        let dtype = if split == 1 {
            DType::Bfloat16
        } else {
            DType::Float32
        };
        let partial_shape = if split == 1 {
            layout.output.clone()
        } else {
            Shape::new([split, layout.rows, layout.columns])?
        };
        let [partials] = self.projection.dispatch_graph(
            graph,
            [input, weights.weight, weights.scales],
            &[OutputSpec::new(partial_shape, dtype)],
            &Dispatch::new(
                [layout.columns.div_ceil(32) * threads, layout.rows.div_ceil(rows), split],
                [threads, 1, 1],
            )
            .templates([
                TemplateArg::dtype("T", DType::Bfloat16),
                TemplateArg::dtype("O", dtype),
                integer("M", layout.rows)?,
                integer("N", layout.columns)?,
                integer("K", layout.inner)?,
                integer("PART", layout.inner / split)?,
                integer("BM", rows)?,
            ]),
        )?;
        if split == 1 {
            return Ok(partials);
        }
        let elements = layout.output.elements()?;
        let [output] = self.reduction.dispatch_graph(
            graph,
            [&partials],
            &[OutputSpec::new(layout.output, DType::Bfloat16)],
            &Dispatch::new([elements, 1, 1], [256, 1, 1]).templates([
                TemplateArg::dtype("T", DType::Bfloat16),
                integer("ELEMENTS", elements)?,
                integer("PARTS", split)?,
            ]),
        )?;
        Ok(output)
    }
}

fn integer(name: &'static str, value: usize) -> Result<TemplateArg> {
    Ok(TemplateArg::int(name, i32::try_from(value)?))
}
