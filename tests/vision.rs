use mirtal::{Array, Device, Result};

#[test]
fn executes_dense_vision_primitives() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let left = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], [2, 2])?;
    let right = Array::from_slice(&[2.0_f32, 0.0, 0.0, 3.0], [2, 2])?;
    let product = graph.matmul(&left, &right)?;
    assert_eq!(stream.read::<f32>(&product)?, vec![2.0, 6.0, 6.0, 12.0]);

    let input = Array::from_slice(&[1.0_f32, 3.0], [1, 2])?;
    let weight = Array::from_slice(&[2.0_f32, 3.0], [2])?;
    let bias = Array::from_slice(&[0.5_f32, -0.5], [2])?;
    let normalized = graph.layer_norm(&input, &weight, &bias, 1.0e-5)?;
    let normalized = stream.read::<f32>(&normalized)?;
    assert!((normalized[0] + 1.499_99).abs() < 1.0e-4);
    assert!((normalized[1] - 2.499_98).abs() < 1.0e-4);

    let gelu_input = Array::from_slice(&[-1.0_f32, 0.0, 1.0], [3])?;
    let approximate = stream.read::<f32>(&graph.gelu_tanh(&gelu_input)?)?;
    assert!((approximate[0] + 0.158_808).abs() < 1.0e-5);
    assert!(approximate[1].abs() < f32::EPSILON);
    assert!((approximate[2] - 0.841_192).abs() < 1.0e-5);
    let exact = stream.read::<f32>(&graph.gelu(&gelu_input)?)?;
    assert!((exact[0] + 0.158_655_3).abs() < 1.0e-5);
    assert!(exact[1].abs() < f32::EPSILON);
    assert!((exact[2] - 0.841_344_7).abs() < 1.0e-5);
    Ok(())
}
