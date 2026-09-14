use mirtal::{Array, DType, Device, FrequencyRopeOptions, Result, RopeOptions};

mod batched;

#[test]
fn rotates_with_base_and_explicit_frequencies_on_device() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let input = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], [1, 1, 1, 4])?;
    let base = graph.rope(
        &input,
        RopeOptions {
            dimensions: 4,
            traditional: false,
            base: Some(10_000.0),
            scale: 1.0,
            offset: 0,
        },
    )?;
    let frequencies = Array::from_slice(&[1.0_f32, 100.0], [2])?;
    let explicit = graph.rope_with_frequencies(
        &input,
        &frequencies,
        FrequencyRopeOptions {
            dimensions: 4,
            traditional: false,
            offset: 0,
        },
    )?;

    assert_eq!(stream.read::<f32>(&base)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(stream.read::<f32>(&explicit)?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn single_token_rotation_preserves_every_batched_row() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let data = (0_i16..3 * 4 * 16)
        .map(|index| f32::from(index % 31) - 15.0)
        .collect::<Vec<_>>();
    let input = Array::from_slice(&data, [3, 4, 1, 16])?;
    for dtype in [DType::Float32, DType::Bfloat16] {
        let input = graph.astype(&input, dtype)?;
        for width in [8_u16, 16] {
            for traditional in [false, true] {
                let options = RopeOptions {
                    dimensions: usize::from(width),
                    traditional,
                    base: Some(10_000.0),
                    scale: 1.0,
                    offset: 128,
                };
                let frequencies = (0..width / 2)
                    .map(|index| 10_000.0_f32.powf(f32::from(index * 2) / f32::from(width)))
                    .collect::<Vec<_>>();
                let frequencies = Array::from_slice(&frequencies, [usize::from(width / 2)])?;
                let explicit = FrequencyRopeOptions {
                    dimensions: usize::from(width),
                    traditional,
                    offset: 128,
                };
                for use_explicit in [false, true] {
                    let rotate = |array: &Array| {
                        if use_explicit {
                            graph.rope_with_frequencies(array, &frequencies, explicit)
                        } else {
                            graph.rope(array, options)
                        }
                    };
                    let actual = stream.read::<f32>(&rotate(&input)?)?;
                    for row in 0..3 {
                        let single = graph.slice(&input, &[row, 0, 0, 0], &[row + 1, 4, 1, 16])?;
                        let expected = stream.read::<f32>(&rotate(&single)?)?;
                        assert_eq!(&actual[row * 64..(row + 1) * 64], expected);
                    }
                }
            }
        }
    }
    Ok(())
}
