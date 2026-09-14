use super::{Array, DType, MxFp4, Result, Shape};
use crate::Error;

pub(super) struct Layout {
    pub rows: usize,
    pub columns: usize,
    pub inner: usize,
    pub output: Shape,
}

impl Layout {
    pub fn new(input: &Array, weights: MxFp4<'_>, partitions: usize) -> Result<Self> {
        for (array, expected) in [
            (input, DType::Bfloat16),
            (weights.weight, DType::Uint32),
            (weights.scales, DType::Uint8),
        ] {
            let actual = array.dtype()?;
            if actual != expected {
                return Err(Error::DtypeMismatch { actual, expected });
            }
        }
        let xs = input.shape()?;
        let ws = weights.weight.shape()?;
        let ss = weights.scales.shape()?;
        let Some((&inner, prefix)) = xs.dimensions().split_last() else {
            return Err(invalid());
        };
        let [columns, packed] = ws.dimensions() else {
            return Err(invalid());
        };
        if prefix.is_empty()
            || inner == 0
            || *columns == 0
            || !inner.is_multiple_of(32)
            || *packed != inner / 8
            || ss.dimensions() != [*columns, inner / 32]
            || partitions == 0
            || !(inner / 32).is_multiple_of(partitions)
        {
            return Err(invalid());
        }
        let rows = Shape::new(prefix.iter().copied())?.elements()?;
        if rows == 0 {
            return Err(invalid());
        }
        let output = Shape::new(prefix.iter().copied().chain([*columns]))?;
        Ok(Self { rows, columns: *columns, inner, output })
    }
}

fn invalid() -> Error {
    Error::InvalidOperation("MXFP4 FP32 matmul requires positive [..., K], [N, K/8], [N, K/32] and partitions dividing K/32".into())
}
