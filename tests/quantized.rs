use mirtal::{Array, DType, Device, GatherQmmOptions, Quantization, Result};

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

#[test]
fn executes_every_native_affine_bit_width() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    for bits in [2, 3, 4, 5, 6, 8] {
        let bins = usize::try_from((1_i32 << bits) - 1)?;
        let values = (0..128)
            .map(|index| u8::try_from(index % bins).map(f32::from))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let dense = Array::from_slice(&values, [2, 64])?;
        let quantized = graph.quantize(&dense, Quantization::new(32, bits)?)?;
        let restored = graph.dequantize(quantized.as_ref())?;
        let actual = stream.read::<f32>(&restored)?;
        let packed = stream.read::<u32>(&quantized.weight)?;
        let scales = stream.read::<f32>(&quantized.scales)?;
        let biases = stream.read::<f32>(&quantized.biases)?;
        let bits_usize = usize::try_from(bits)?;
        let words_per_row = 64 * bits_usize / 32;

        assert_eq!(quantized.weight.dtype()?, DType::Uint32);
        assert_eq!(quantized.weight.shape()?.dimensions(), &[2, words_per_row]);
        assert_eq!(actual.len(), values.len());
        for (index, (actual, expected)) in actual.iter().zip(&values).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.5,
                "{bits}-bit dequantized element {index}: expected {expected}, got {actual}"
            );
            let row = index / 64;
            let column = index % 64;
            let quantized = unpack(&packed[row * words_per_row..], column, bits_usize)?;
            let group = row * 2 + column / 32;
            let unpacked =
                scales[group].mul_add(f32::from(u16::try_from(quantized)?), biases[group]);
            assert!(
                (actual - unpacked).abs() < 1.0e-5,
                "{bits}-bit packed element {index}: expected {actual}, got {unpacked}"
            );
        }

        let input = Array::from_slice(&vec![1.0_f32; 64], [1, 64])?;
        let output = graph.quantized_matmul(&input, quantized.as_ref(), true)?;
        let sums = stream.read::<f32>(&output)?;
        for (row, sum) in sums.iter().enumerate() {
            let expected = actual[row * 64..(row + 1) * 64].iter().sum::<f32>();
            assert!(
                (sum - expected).abs() < 1.0e-2,
                "{bits}-bit row {row}: expected {expected}, got {sum}"
            );
        }

        let bank = Array::from_slice(&values, [2, 1, 64])?;
        let bank = graph.quantize(&bank, Quantization::new(32, bits)?)?;
        let inputs = Array::from_slice(&vec![1.0_f32; 128], [1, 1, 2, 1, 64])?;
        let indices = Array::from_slice(&[0_u32, 1], [1, 1, 2])?;
        let selected = graph.gather_qmm(
            &inputs,
            bank.as_ref(),
            &indices,
            GatherQmmOptions { transpose: true, sorted_indices: false },
        )?;
        let selected = stream.read::<f32>(&selected)?;
        for (row, sum) in selected.iter().enumerate() {
            let expected = actual[row * 64..(row + 1) * 64].iter().sum::<f32>();
            assert!(
                (sum - expected).abs() < 1.0e-2,
                "{bits}-bit selected row {row}: expected {expected}, got {sum}"
            );
        }
    }
    Ok(())
}

#[test]
fn executes_native_mxfp8_without_affine_biases() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let values = (0_u8..64).map(|index| (f32::from(index) - 31.5) / 16.0).collect::<Vec<_>>();
    let dense = Array::from_slice(&values, [2, 32])?;
    let quantized = graph.quantize_mxfp8(&dense)?;
    let restored = graph.dequantize_mxfp8(quantized.as_ref())?;

    assert_eq!(quantized.weight.dtype()?, DType::Uint32);
    assert_eq!(quantized.weight.shape()?.dimensions(), &[2, 8]);
    assert_eq!(quantized.scales.dtype()?, DType::Uint8);
    assert_eq!(quantized.scales.shape()?.dimensions(), &[2, 1]);
    let restored = stream.read::<f32>(&restored)?;
    for (index, (actual, expected)) in restored.iter().zip(&values).enumerate() {
        let tolerance = expected.abs().mul_add(0.1, 0.1);
        assert!(
            (actual - expected).abs() <= tolerance,
            "MXFP8 element {index}: expected {expected}, got {actual}"
        );
    }

    let input = Array::from_slice(&[1.0_f32; 32], [1, 32])?;
    let output = graph.mxfp8_matmul(&input, quantized.as_ref(), true)?;
    let actual = stream.read::<f32>(&output)?;
    for (row, actual) in actual.iter().enumerate() {
        let expected = restored[row * 32..(row + 1) * 32].iter().sum::<f32>();
        assert!(
            (actual - expected).abs() <= 0.2,
            "MXFP8 row {row}: expected {expected}, got {actual}"
        );
    }
    Ok(())
}

fn unpack(words: &[u32], index: usize, bits: usize) -> Result<u32> {
    let bit = index * bits;
    let word = bit / 32;
    let shift = bit % 32;
    let mut packed = u64::from(words[word]) >> shift;
    if shift + bits > 32 {
        packed |= u64::from(words[word + 1]) << (32 - shift);
    }
    Ok(u32::try_from(packed & ((1_u64 << bits) - 1))?)
}
