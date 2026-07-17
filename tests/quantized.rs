use mirtal::{Array, Device, Quantization, Result};

#[test]
fn gathers_and_dequantizes_embedding_rows_on_device() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let mut values = vec![1.0_f32; 128];
    values[64..].fill(2.0);
    let dense = Array::from_slice(&values, [2, 64])?;
    let quantized = graph.quantize(&dense, Quantization::new(64, 4)?)?;
    let indices = Array::from_slice(&[1_u32, 0], [1, 2])?;
    let quantized = quantized.as_ref();
    let weight = graph.take(quantized.weight, &indices, 0)?;
    let scales = graph.take(quantized.scales, &indices, 0)?;
    let biases = graph.take(quantized.biases, &indices, 0)?;
    let output = graph.dequantize(mirtal::Quantized {
        weight: &weight,
        scales: &scales,
        biases: &biases,
        format: quantized.format,
    })?;
    let values = stream.read::<f32>(&output)?;

    assert_eq!(output.shape()?.dimensions(), &[1, 2, 64]);
    assert!(values[..64].iter().all(|value| (*value - 2.0).abs() < 1.0e-3));
    assert!(values[64..].iter().all(|value| (*value - 1.0).abs() < 1.0e-3));
    Ok(())
}
