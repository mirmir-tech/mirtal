use mirtal::{Array, Device, FrequencyRopeOptions, Result, RopeOptions};

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
