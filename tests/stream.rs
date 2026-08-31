use mirtal::{Array, DType, Device, Result};

#[test]
fn builds_and_evaluates_a_graph_on_an_explicit_stream() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let left = Array::from_slice(&[1.0_f32, 2.0], [1, 2])?;
    let right = Array::from_slice(&[3.0_f32, 4.0], [1, 2])?;
    let output = stream.graph().add(&left, &right)?;

    stream.eval(&output)?;
    stream.synchronize()?;
    assert_eq!(output.dtype()?, DType::Float32);
    assert_eq!(output.shape()?.dimensions(), &[1, 2]);
    assert_eq!(stream.read::<f32>(&output)?, vec![4.0, 6.0]);
    Ok(())
}

#[test]
fn evaluates_multiple_lazy_roots_on_an_explicit_stream() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1.0_f32, 2.0], [2])?;
    let doubled = stream.graph().add(&input, &input)?;
    let squared = stream.graph().multiply(&input, &input)?;

    stream.eval_many(&[&doubled, &squared])?;
    stream.synchronize()?;

    assert_eq!(stream.read::<f32>(&doubled)?, vec![2.0, 4.0]);
    assert_eq!(stream.read::<f32>(&squared)?, vec![1.0, 4.0]);
    Ok(())
}

#[test]
fn builds_a_new_generation_from_scheduled_roots() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1.0_f32, 2.0], [2])?;
    let doubled = stream.graph().add(&input, &input)?;
    let squared = stream.graph().multiply(&input, &input)?;

    stream.eval_many(&[&doubled, &squared])?;
    let next = stream.graph().add(&doubled, &squared)?;
    stream.eval(&next)?;
    stream.synchronize()?;

    assert_eq!(stream.read::<f32>(&next)?, vec![3.0, 8.0]);
    Ok(())
}
