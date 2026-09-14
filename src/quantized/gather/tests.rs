use super::*;
use crate::{DType, Device, Quantization, Stream};

#[test]
fn indexed_affine_and_mxfp4_match_materialized_broadcast_gathers() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let bank = Array::from_slice(&values(2 * 32 * 64)?, [2, 32, 64])?;
    let affine = graph.quantize(&bank, Quantization::new(32, 4)?)?;
    let mx = graph.quantize_mxfp4(&bank)?;
    for rows in [1, 3] {
        let input = Array::from_slice(&values(3 * rows * 64)?, [3, rows, 64])?;
        let input = graph.astype(&input, DType::Bfloat16)?;
        let lhs = Array::from_slice(&[2_u32, 0], [2, 1])?;
        for (rhs, sorted) in [([1_u32, 0, 1], false), ([0, 1, 1], true)] {
            let rhs = Array::from_slice(&rhs, [1, 3])?;
            let indices = GatherIndices { lhs: Some(&lhs), rhs: Some(&rhs) };
            let options = GatherQmmOptions { transpose: true, sorted_indices: sorted };
            let expanded = Array::from_slice(&[2_u32, 2, 2, 0, 0, 0], [2, 3])?;
            let materialized = graph.take(&input, &expanded, 0)?;
            let actual =
                graph.gather_qmm_with_indices(&input, affine.as_ref(), indices, options)?;
            let expected = graph.gather_qmm(&materialized, affine.as_ref(), &rhs, options)?;
            compare(&actual, &expected, &stream)?;
            let actual = graph.gather_mxfp4_with_indices(&input, mx.as_ref(), indices, options)?;
            let expected = graph.gather_mxfp4(&materialized, mx.as_ref(), &rhs, options)?;
            // Deferred outputs own their dependencies, not these borrowed wrappers.
            drop((materialized, expanded, rhs));
            compare(&actual, &expected, &stream)?;
        }
    }
    Ok(())
}

#[test]
fn optional_selectors_preserve_implicit_indexing_and_plain_matmul() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let bank = Array::from_slice(&values(2 * 64 * 64)?, [2, 64, 64])?;
    let affine = graph.quantize(&bank, Quantization::new(32, 4)?)?;
    let mx = graph.quantize_mxfp4(&bank)?;
    let input = Array::from_slice(&values(2 * 64)?, [1, 2, 64])?;
    let lhs = Array::from_slice(&[0_u32, 0], [2, 1])?;
    let rhs = Array::from_slice(&[0_u32, 1], [2])?;
    for transpose in [false, true] {
        let options = GatherQmmOptions { transpose, sorted_indices: true };
        let selectors = GatherIndices { lhs: Some(&lhs), rhs: None };
        let explicit = GatherIndices { lhs: Some(&lhs), rhs: Some(&rhs) };
        compare(
            &graph.gather_qmm_with_indices(&input, affine.as_ref(), selectors, options)?,
            &graph.gather_qmm_with_indices(&input, affine.as_ref(), explicit, options)?,
            &stream,
        )?;
        compare(
            &graph.gather_mxfp4_with_indices(&input, mx.as_ref(), selectors, options)?,
            &graph.gather_mxfp4_with_indices(&input, mx.as_ref(), explicit, options)?,
            &stream,
        )?;
        compare(
            &graph.gather_qmm_with_indices(
                &input,
                affine.as_ref(),
                GatherIndices::default(),
                options,
            )?,
            &graph.quantized_matmul(&input, affine.as_ref(), transpose)?,
            &stream,
        )?;
        compare(
            &graph.gather_mxfp4_with_indices(
                &input,
                mx.as_ref(),
                GatherIndices::default(),
                options,
            )?,
            &graph.mxfp4_matmul(&input, mx.as_ref(), transpose)?,
            &stream,
        )?;
    }
    Ok(())
}

#[test]
fn rejects_incompatible_index_shapes_without_executing() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let bank = Array::from_slice(&values(2 * 64 * 64)?, [2, 64, 64])?;
    let affine = graph.quantize(&bank, Quantization::new(32, 4)?)?;
    let mx = graph.quantize_mxfp4(&bank)?;
    let input = Array::from_slice(&values(2 * 64)?, [2, 1, 64])?;
    let lhs = Array::from_slice(&[0_u32, 1], [2])?;
    let rhs = Array::from_slice(&[0_u32, 1, 0], [3])?;
    let selectors = GatherIndices { lhs: Some(&lhs), rhs: Some(&rhs) };
    let options = GatherQmmOptions { transpose: true, sorted_indices: false };
    assert!(
        graph
            .gather_qmm_with_indices(&input, affine.as_ref(), selectors, options)
            .is_err()
    );
    assert!(
        graph
            .gather_mxfp4_with_indices(&input, mx.as_ref(), selectors, options)
            .is_err()
    );
    Ok(())
}

fn values(count: usize) -> Result<Vec<f32>> {
    (0..count)
        .map(|i| Ok((f32::from(i16::try_from(i % 37)?) - 18.0) / 16.0))
        .collect()
}

fn compare(actual: &Array, expected: &Array, stream: &Stream) -> Result<()> {
    assert_eq!(actual.shape()?, expected.shape()?);
    let a = stream.read::<f32>(&stream.graph().astype(actual, DType::Float32)?)?;
    let b = stream.read::<f32>(&stream.graph().astype(expected, DType::Float32)?)?;
    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
    assert!(a.iter().any(|v| v.abs() > 0.01));
    assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() <= 1e-4));
    Ok(())
}
