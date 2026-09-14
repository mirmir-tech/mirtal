use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use super::*;
use crate::{CompileOptions, Error};

#[test]
fn compiled_projection_reuses_trace_and_rebinds_arrays() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    for (parts, tile) in [1, 2]
        .into_iter()
        .flat_map(|parts| [MxFp4RowTile::Rows16, MxFp4RowTile::Rows32].map(|tile| (parts, tile)))
    {
        let count = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&count);
        let plan = MxFp4Matmul::new()?.with_row_tile(tile);
        let partitions = NonZeroUsize::new(parts).ok_or(Error::ShapeOverflow)?;
        let compiled = stream.compile(CompileOptions::default(), move |graph, [x, w, s]| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok([plan.matmul_graph(graph, &x, MxFp4 { weight: &w, scales: &s }, partitions)?])
        })?;
        for (run, rows, value, code, exponent) in [
            (0, 3, 0.25, 0x2222_2222u32, 127u32),
            (1, 3, 0.5, 0xaaaa_aaaau32, 128u32),
            (2, 33, 0.125, 0x2222_2222u32, 127u32),
            (3, 3, 0.25, 0x2222_2222u32, 127u32),
        ] {
            let input = graph.astype(
                &Array::from_slice(&vec![value; rows * 64], [rows, 64])?,
                DType::Bfloat16,
            )?;
            let weight = Array::from_slice(&[code; 40], [5, 8])?;
            let scales =
                graph.astype(&Array::from_slice(&[exponent; 10], [5, 2])?, DType::Uint8)?;
            let [output] = compiled.call(&stream, [&input, &weight, &scales])?;
            let expected = if run == 1 {
                -64.0
            } else {
                64.0 * value
            };
            assert_eq!(
                stream.read::<f32>(&graph.astype(&output, DType::Float32)?)?,
                vec![expected; rows * 5]
            );
            assert_eq!(
                count.load(Ordering::Relaxed),
                if run < 2 {
                    1
                } else {
                    2
                }
            );
        }
        let x = graph.astype(&Array::from_slice(&[0.25f32; 192], [64, 3])?, DType::Bfloat16)?;
        let x = graph.transpose(&x, &[1, 0])?;
        let w = Array::from_slice(&[0x2222_2222u32; 40], [5, 8])?;
        let s = graph.astype(&Array::from_slice(&[127u32; 10], [5, 2])?, DType::Uint8)?;
        let [output] = compiled.call(&stream, [&x, &w, &s])?;
        assert_eq!(stream.read::<f32>(&graph.astype(&output, DType::Float32)?)?, vec![16.0; 15]);
        assert_eq!(count.load(Ordering::Relaxed), 2, "strides must remain runtime data");
        let other = Device::gpu(0).new_stream()?;
        assert!(matches!(compiled.call(&other, [&x, &w, &s]), Err(Error::StreamMismatch { .. })));
        let bad = graph.astype(&x, DType::Float32)?;
        assert!(compiled.call(&stream, [&bad, &w, &s]).is_err());
        let bad = graph.reshape(&s, &Shape::new([2, 5])?)?;
        assert!(compiled.call(&stream, [&x, &w, &bad]).is_err());
    }
    Ok(())
}
