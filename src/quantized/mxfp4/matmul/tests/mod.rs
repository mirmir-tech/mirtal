mod compiled;

use super::*;
use crate::Device;

#[test]
fn fp32_partitions_preserve_tail_shapes_and_known_dot_products() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let plan = MxFp4Matmul::new()?;
    for (rows, columns, inner, parts) in
        [(1, 3, 64, 2), (8, 33, 96, 3), (17, 35, 64, 2), (33, 35, 128, 4), (40, 32, 64, 2)]
    {
        let values = (0..rows * inner)
            .map(|i| f32::from(i16::try_from(i % 13).unwrap_or(0) - 6) * 0.125)
            .collect::<Vec<_>>();
        let words = (0..columns * inner / 8)
            .map(|i| {
                (if i % 2 == 0 {
                    0x7654_3210u32
                } else {
                    0xfedc_ba98u32
                })
                .rotate_left(u32::try_from(i % 8).unwrap_or(0) * 4)
            })
            .collect::<Vec<_>>();
        let scales = (0..columns * inner / 32)
            .map(|i| 125 + u32::try_from(i % 4).unwrap_or(0))
            .collect::<Vec<_>>();
        let input = graph.astype(&Array::from_slice(&values, [rows, inner])?, DType::Bfloat16)?;
        let weight = Array::from_slice(&words, [columns, inner / 8])?;
        let scale =
            graph.astype(&Array::from_slice(&scales, [columns, inner / 32])?, DType::Uint8)?;
        let quantized = MxFp4 { weight: &weight, scales: &scale };
        let mut expected = Vec::new();
        let magnitudes = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0.0;
                for k in 0..inner {
                    let code = (words[column * inner / 8 + k / 8] >> (4 * (k % 8))) & 15;
                    let value = magnitudes[(code & 7) as usize]
                        * 2.0f32.powi(i32::try_from(scales[column * inner / 32 + k / 32])? - 127);
                    sum = values[row * inner + k].mul_add(
                        if code < 8 {
                            value
                        } else {
                            -value
                        },
                        sum,
                    );
                }
                expected.push(sum);
            }
        }
        let expected =
            graph.astype(&Array::from_slice(&expected, [rows, columns])?, DType::Bfloat16)?;
        for (count, tile) in [1, parts].into_iter().flat_map(|count| {
            [MxFp4RowTile::Rows16, MxFp4RowTile::Rows32].map(|tile| (count, tile))
        }) {
            let plan = MxFp4Matmul::new()?.with_row_tile(tile);
            let count = NonZeroUsize::new(count).ok_or(crate::Error::ShapeOverflow)?;
            let actual = plan.matmul(&input, quantized, count, &stream)?;
            assert_eq!(actual.shape()?.dimensions(), [rows, columns]);
            assert_eq!(actual.dtype()?, DType::Bfloat16);
            assert_eq!(
                stream.read::<f32>(&graph.astype(&actual, DType::Float32)?)?,
                stream.read::<f32>(&graph.astype(&expected, DType::Float32)?)?
            );
        }
        if rows == 40 {
            let input = graph.reshape(&input, &Shape::new([5, 8, inner])?)?;
            let result = plan.matmul(&input, quantized, NonZeroUsize::MIN, &stream)?;
            assert_eq!(result.shape()?.dimensions(), [5, 8, columns]);
            assert_eq!(
                stream.read::<f32>(&graph.astype(&result, DType::Float32)?)?,
                stream.read::<f32>(&graph.astype(&expected, DType::Float32)?)?
            );
        }
        assert!(
            plan.matmul(
                &input,
                quantized,
                NonZeroUsize::new(5).ok_or(crate::Error::ShapeOverflow)?,
                &stream
            )
            .is_err()
        );
        assert!(
            plan.matmul(
                &graph.astype(&input, DType::Float32)?,
                quantized,
                NonZeroUsize::MIN,
                &stream
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn output_bias_follows_projection_rounding() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let plan = MxFp4Matmul::new()?;
    let mut values = [0.0f32; 64];
    values[0] = 1.0;
    values[32] = 1.0 / 256.0;
    let input = graph.astype(&Array::from_slice(&values, [1, 64])?, DType::Bfloat16)?;
    let weight = Array::from_slice(&[0x2222_2222u32; 8], [1, 8])?;
    let scales = graph.astype(&Array::from_slice(&[127u32; 2], [1, 2])?, DType::Uint8)?;
    let bias = graph.astype(&Array::from_slice(&[1.0f32 / 256.0], [1])?, DType::Bfloat16)?;
    let result = plan.matmul(
        &input,
        MxFp4 { weight: &weight, scales: &scales },
        NonZeroUsize::new(2).ok_or(crate::Error::ShapeOverflow)?,
        &stream,
    )?;
    let biased = graph.add(&result, &bias)?;
    assert_eq!(stream.read::<f32>(&graph.astype(&biased, DType::Float32)?)?, [1.0]);
    Ok(())
}

#[test]
fn copies_strided_input_and_rejects_malformed_scales() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let plan = MxFp4Matmul::new()?;
    let source =
        graph.astype(&Array::from_slice(&vec![0.25f32; 64 * 3], [64, 3])?, DType::Bfloat16)?;
    let input = graph.transpose(&source, &[1, 0])?;
    let weight = Array::from_slice(&[0x2222_2222u32; 40], [5, 8])?;
    let scales = graph.astype(&Array::from_slice(&[127u32; 10], [5, 2])?, DType::Uint8)?;
    let output = plan.matmul(
        &input,
        MxFp4 { weight: &weight, scales: &scales },
        NonZeroUsize::new(2).ok_or(crate::Error::ShapeOverflow)?,
        &stream,
    )?;
    assert_eq!(stream.read::<f32>(&graph.astype(&output, DType::Float32)?)?, vec![16.0; 15]);
    let bad = graph.reshape(&scales, &Shape::new([2, 5])?)?;
    assert!(
        plan.matmul(&input, MxFp4 { weight: &weight, scales: &bad }, NonZeroUsize::MIN, &stream)
            .is_err()
    );
    Ok(())
}
