use std::fs;

use mirtal::{DType, Device, TensorFile, TensorRange};

#[test]
fn loads_e5m2_safetensors_as_raw_bytes() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("mirtal-e5m2-{}.safetensors", std::process::id()));
    let mut header =
        r#"{"weight":{"dtype":"F8_E5M2","shape":[2,2],"data_offsets":[0,4]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&[0x3c, 0x40, 0xbc, 0x38]);
    fs::write(&path, data)?;

    let stream = Device::cpu(0).new_stream()?;
    let tensors = TensorFile::load(&path, &stream)?;
    let weight = tensors.get("weight")?;
    assert_eq!(weight.dtype()?, DType::Uint8);
    assert_eq!(weight.shape()?.dimensions(), &[2, 2]);
    assert_eq!(stream.read::<u32>(&weight)?, [0x3c, 0x40, 0xbc, 0x38]);

    drop(tensors);
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn loads_tensor_ranges_in_place() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("mirtal-ranges-{}.bin", std::process::id()));
    let mut data = (0_u8..32).collect::<Vec<_>>();
    data.extend_from_slice(&1.5_f32.to_le_bytes());
    data.extend_from_slice(&(-2.0_f32).to_le_bytes());
    fs::write(&path, data)?;
    let range = |name: &str, offset, dtype, shape: &[u64]| TensorRange {
        name: name.to_owned(),
        offset,
        dtype,
        shape: shape.to_vec(),
    };

    let stream = Device::cpu(0).new_stream()?;
    let tensors = TensorFile::load_ranges(
        &path,
        &[
            range("blocks", 5, DType::Uint8, &[3, 4]),
            range("tail", 30, DType::Uint8, &[1, 2]),
            range("norm", 32, DType::Float32, &[2]),
        ],
        &stream,
    )?;
    assert_eq!(tensors.len(), 3);
    let blocks = tensors.get("blocks")?;
    assert_eq!(blocks.dtype()?, DType::Uint8);
    assert_eq!(blocks.shape()?.dimensions(), &[3, 4]);
    assert_eq!(stream.read::<u32>(&blocks)?, (5..17).collect::<Vec<_>>());
    assert_eq!(stream.read::<u32>(&tensors.get("tail")?)?, [30, 31]);
    assert_eq!(stream.read::<f32>(&tensors.get("norm")?)?, [1.5, -2.0]);

    for invalid in [
        vec![range("late", 39, DType::Uint8, &[1, 2])],
        vec![range("wide", 36, DType::Float32, &[2])],
        vec![range("same", 0, DType::Uint8, &[1]), range("same", 1, DType::Uint8, &[1])],
        vec![range("none", 0, DType::Uint8, &[0, 4])],
        vec![range("scalar", 0, DType::Uint8, &[])],
    ] {
        assert!(TensorFile::load_ranges(&path, &invalid, &stream).is_err());
    }

    drop(tensors);
    fs::remove_file(path)?;
    Ok(())
}
