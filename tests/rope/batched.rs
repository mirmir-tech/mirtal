use super::*;

#[test]
fn device_offsets_match_scalar_rows_with_both_frequency_modes() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    for batch in [1, 2, 3, 5] {
        for sequence in [1, 3] {
            let data = (0..batch * sequence * 4 * 16)
                .map(|index| Ok(f32::from(u16::try_from(index % 31)?) - 15.0))
                .collect::<Result<Vec<_>>>()?;
            // Transposed views exercise non-contiguous batch/head input too.
            let input = Array::from_slice(&data, [batch, sequence, 4, 16])?;
            let input = graph.transpose(&input, &[0, 2, 1, 3])?;
            let offsets = [8193_u32, 0, 127, 32, 2049];
            let device_offsets = Array::from_slice(&offsets[..batch], [batch])?;
            for dtype in [DType::Float32, DType::Bfloat16] {
                let input = graph.astype(&input, dtype)?;
                for dimensions in [8, 16] {
                    for traditional in [false, true] {
                        let frequencies = (0..dimensions / 2)
                            .map(|i| {
                                Ok(10_000_f32.powf(
                                    f32::from(u16::try_from(2 * i)?)
                                        / f32::from(u16::try_from(dimensions)?),
                                ))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let frequencies = Array::from_slice(&frequencies, [dimensions / 2])?;
                        for explicit in [false, true] {
                            let output = if explicit {
                                graph.rope_with_frequencies_batched(
                                    &input,
                                    &frequencies,
                                    FrequencyRopeOptions {
                                        dimensions,
                                        traditional,
                                        offset: &device_offsets,
                                    },
                                )?
                            } else {
                                graph.rope_batched(
                                    &input,
                                    RopeOptions {
                                        dimensions,
                                        traditional,
                                        base: Some(10_000.0),
                                        scale: 0.5,
                                        offset: &device_offsets,
                                    },
                                )?
                            };
                            assert_eq!(output.dtype()?, dtype);
                            let actual = stream.read::<f32>(&output)?;
                            let row_size = 4 * sequence * 16;
                            for (row, &offset) in offsets[..batch].iter().enumerate() {
                                let single = graph.slice(
                                    &input,
                                    &[row, 0, 0, 0],
                                    &[row + 1, 4, sequence, 16],
                                )?;
                                let offset = usize::try_from(offset)?;
                                let expected = if explicit {
                                    graph.rope_with_frequencies(
                                        &single,
                                        &frequencies,
                                        FrequencyRopeOptions { dimensions, traditional, offset },
                                    )?
                                } else {
                                    graph.rope(
                                        &single,
                                        RopeOptions {
                                            dimensions,
                                            traditional,
                                            base: Some(10_000.0),
                                            scale: 0.5,
                                            offset,
                                        },
                                    )?
                                };
                                assert_eq!(
                                    &actual[row * row_size..(row + 1) * row_size],
                                    stream.read::<f32>(&expected)?,
                                    "batch={batch} sequence={sequence} row={row} dtype={dtype:?} explicit={explicit}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn rejects_invalid_offsets_without_readback() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1_f32; 3 * 4], [3, 1, 1, 4])?;
    for offsets in [
        Array::from_slice(&[0_u32, 1], [2])?,
        Array::from_slice(&[0_f32, 1.0, 2.0], [3])?,
        Array::from_slice(&[0_u32, 1, 2], [3, 1])?,
    ] {
        assert!(
            stream
                .graph()
                .rope_batched(
                    &input,
                    RopeOptions {
                        dimensions: 4,
                        traditional: false,
                        base: Some(10_000.0),
                        scale: 1.0,
                        offset: &offsets,
                    }
                )
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn one_device_offset_broadcasts_without_folding_batch_rows() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let input = Array::from_slice(&[1_f32; 12], [3, 1, 1, 4])?;
    let expected = graph.rope(
        &input,
        RopeOptions {
            dimensions: 4,
            traditional: false,
            base: Some(10_000.0),
            scale: 1.0,
            offset: 127,
        },
    )?;
    for offset in [Array::from_slice(&[127_u32], [])?, Array::from_slice(&[127_u32], [1])?] {
        let actual = graph.rope_batched(
            &input,
            RopeOptions {
                dimensions: 4,
                traditional: false,
                base: Some(10_000.0),
                scale: 1.0,
                offset: &offset,
            },
        )?;
        assert_eq!(stream.read::<f32>(&actual)?, stream.read::<f32>(&expected)?);
    }
    Ok(())
}
