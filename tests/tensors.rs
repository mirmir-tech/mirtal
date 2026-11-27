use std::fs;

use mirtal::{DType, Device, TensorFile, TensorRange, TensorRangeError};

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

/// Each range reads its bytes from the file on evaluation; ranges that do
/// not fit the file, or repeat a name, are refused before anything is read.
#[test]
fn loads_tensor_ranges_from_their_offsets() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("mirtal-ranges-{}.bin", std::process::id()));
    let mut data = (0_u8..32).collect::<Vec<_>>();
    data.extend_from_slice(&1.5_f32.to_le_bytes());
    data.extend_from_slice(&(-2.0_f32).to_le_bytes());
    fs::write(&path, data)?;
    let range = |name: &str, offset, dtype, shape: &[u64]| {
        TensorRange::new(name, offset, dtype, shape.to_vec())
    };

    let stream = Device::cpu(0).new_stream()?;
    let tensors = TensorFile::load_ranges(
        &path,
        &[
            range("blocks", 5, DType::Uint8, &[3, 4])?,
            range("tail", 30, DType::Uint8, &[1, 2])?,
            range("norm", 32, DType::Float32, &[2])?,
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
    let mut tensors = tensors;
    let tail = tensors.take("tail")?;
    assert_eq!((tensors.len(), tensors.contains("tail")?), (2, false));
    assert_eq!(stream.read::<u32>(&tail)?, [30, 31]);
    assert!(tensors.take("tail").is_err());

    let past = |name: &str, end| TensorRangeError::PastEndOfFile { end, file: 40 }.named(name);
    for (invalid, expected) in [
        (vec![range("late", 39, DType::Uint8, &[1, 2])?], past("late", 41)),
        (vec![range("wide", 36, DType::Float32, &[2])?], past("wide", 44)),
        (
            vec![range("same", 0, DType::Uint8, &[1])?, range("same", 1, DType::Uint8, &[1])?],
            TensorRangeError::Duplicate.named("same"),
        ),
    ] {
        let error = TensorFile::load_ranges(&path, &invalid, &stream).err().map(|e| e.to_string());
        assert_eq!(error, Some(expected));
    }

    drop(tensors);
    fs::remove_file(path)?;
    Ok(())
}

/// A file that shrank after its ranges were checked fails the evaluation
/// that reads past its end, as an error rather than a terminated process.
#[test]
fn a_range_past_a_shrunk_file_fails_its_evaluation() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("mirtal-shrunk-{}.bin", std::process::id()));
    fs::write(&path, vec![7_u8; 64])?;
    let stream = Device::cpu(0).new_stream()?;
    let range = TensorRange::new("late", 32, DType::Uint8, vec![32])?;
    let tensors = TensorFile::load_ranges(&path, &[range], &stream)?;
    fs::write(&path, vec![7_u8; 40])?;
    let error = tensors.evaluate().err().map(|error| error.to_string()).unwrap_or_default();
    assert!(error.contains("unexpected end of file"), "{error}");
    let error = stream.eval(&tensors.get("late")?).err().map(|error| error.to_string());
    assert!(error.is_some_and(|error| error.contains("unexpected end of file")));
    drop(tensors);
    fs::remove_file(path)?;
    Ok(())
}

/// An evaluation returns with the range's bytes read: dropping the array
/// right after leaves no read behind to write into its freed buffer, and a
/// file cut after the evaluation does not change what was read.
#[test]
fn an_evaluated_range_holds_its_bytes_when_it_returns() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("mirtal-dropped-{}.bin", std::process::id()));
    let bytes = (0..4 << 20).map(|index: u32| index.to_le_bytes()[0]).collect::<Vec<_>>();
    fs::write(&path, &bytes)?;
    let stream = Device::cpu(0).new_stream()?;
    let range = || TensorRange::new("all", 0, DType::Uint8, vec![4 << 20]);
    for _ in 0..64 {
        let tensors = TensorFile::load_ranges(&path, &[range()?], &stream)?;
        let array = tensors.get("all")?;
        drop(tensors);
        stream.eval(&array)?;
        drop(array);
    }
    let tensors = TensorFile::load_ranges(&path, &[range()?], &stream)?;
    let array = tensors.get("all")?;
    stream.eval(&array)?;
    fs::write(&path, b"")?;
    let read = stream.read::<u32>(&array)?;
    assert!(read.iter().zip(&bytes).all(|(read, byte)| *read == u32::from(*byte)));
    drop((tensors, array));
    fs::remove_file(path)?;
    Ok(())
}

trait Named {
    fn named(self, name: &str) -> String;
}

impl Named for TensorRangeError {
    fn named(self, name: &str) -> String {
        mirtal::Error::TensorRange { name: name.to_owned(), reason: self }.to_string()
    }
}
