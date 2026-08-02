use std::fs;

use mirtal::{DType, Device, TensorFile};

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
