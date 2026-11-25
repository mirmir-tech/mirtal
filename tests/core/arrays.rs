//! Array handles, host reads, graph detachment and native state.

use mirtal::{Array, DType, Device, Result};

#[test]
fn reuses_storage_after_detaching_an_evaluated_graph() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1.0_f32, 2.0], [2])?;
    let output = stream.graph().add(&input, &input)?;
    stream.eval(&output)?;
    stream.synchronize()?;
    stream.detach_graph(&output)?;

    let next = stream.graph().add(&output, &input)?;
    assert_eq!(stream.read::<f32>(&next)?, vec![3.0, 6.0]);
    Ok(())
}

#[test]
fn clones_immutable_array_handles_without_copying_data() -> Result<()> {
    let input = Array::from_slice(&[1_u32, 2, 3], [3])?;
    let snapshot = input.clone();

    assert_eq!(input.shape()?, snapshot.shape()?);
    assert_eq!(input.dtype()?, snapshot.dtype()?);
    Ok(())
}

#[test]
fn reads_an_exact_u32_array_after_async_evaluation() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let logits = Array::from_slice(&[1.0_f32, 3.0, 2.0, 4.0, 0.0, 2.0], [2, 1, 3])?;
    let tokens = stream.graph().argmax_axis(&logits, -1, false)?;
    stream.eval(&tokens)?;

    assert_eq!(stream.read::<u32>(&tokens)?, vec![1, 0]);
    Ok(())
}

#[test]
fn converts_device_dtype_during_an_explicit_host_read() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1.5_f32, -2.25], [2])?;
    let bfloat = stream.graph().astype(&input, DType::Bfloat16)?;

    assert_eq!(stream.read::<f32>(&bfloat)?, vec![1.5, -2.25]);
    Ok(())
}

#[test]
fn reports_native_version_and_allocator_state() -> Result<()> {
    assert_ne!(mirtal::version()?, "");
    let stats = mirtal::memory::stats()?;
    assert!(stats.limit > 0);
    Ok(())
}
