use mirtal::{Array, DType, Device, Result};

#[test]
fn converts_e4m3_bytes_on_an_explicit_stream() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let word = Array::from_slice(&[0xb840_3800_u32], [1])?;
    let packed = stream.graph().view_dtype(&word, DType::Uint8)?;
    let packed = stream.graph().reshape(&packed, &mirtal::Shape::new([2, 2])?)?;
    let converted = stream.graph().from_fp8(&packed, DType::Bfloat16)?;

    assert_eq!(converted.dtype()?, DType::Bfloat16);
    assert_eq!(converted.shape()?.dimensions(), &[2, 2]);
    assert_eq!(stream.read::<f32>(&converted)?, [0.0, 1.0, 2.0, -1.0]);
    Ok(())
}

#[test]
fn round_trips_representable_e4m3_values() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[0.0_f32, 1.0, 2.0, -1.0], [2, 2])?;
    let packed = stream.graph().to_fp8(&input)?;
    let output = stream.graph().from_fp8(&packed, DType::Float32)?;

    assert_eq!(packed.dtype()?, DType::Uint8);
    assert_eq!(stream.read::<f32>(&output)?, [0.0, 1.0, 2.0, -1.0]);
    Ok(())
}
