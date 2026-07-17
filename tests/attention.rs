use mirtal::{Array, AttentionMask, Device, Result, ScaledDotProductAttention};

#[test]
fn applies_none_and_causal_masks_on_the_explicit_stream() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let queries = Array::from_slice(&[0.0_f32, 0.0], [1, 1, 2, 1])?;
    let keys = Array::from_slice(&[1.0_f32, 1.0], [1, 1, 2, 1])?;
    let values = Array::from_slice(&[1.0_f32, 3.0], [1, 1, 2, 1])?;
    let graph = stream.graph();
    let full = graph.scaled_dot_product_attention(
        &queries,
        &keys,
        &values,
        ScaledDotProductAttention {
            scale: 1.0,
            mask: AttentionMask::None,
            sinks: None,
        },
    )?;
    let causal = graph.scaled_dot_product_attention(
        &queries,
        &keys,
        &values,
        ScaledDotProductAttention {
            scale: 1.0,
            mask: AttentionMask::Causal,
            sinks: None,
        },
    )?;

    assert_eq!(stream.read::<f32>(&full)?, vec![2.0, 2.0]);
    assert_eq!(stream.read::<f32>(&causal)?, vec![1.0, 2.0]);
    Ok(())
}

#[test]
fn rejects_invalid_attention_scale_before_native_execution() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1.0_f32], [1, 1, 1, 1])?;
    let result = stream.graph().scaled_dot_product_attention(
        &input,
        &input,
        &input,
        ScaledDotProductAttention {
            scale: f32::NAN,
            mask: AttentionMask::None,
            sinks: None,
        },
    );
    let Err(error) = result else {
        return Err(mirtal::Error::InvalidAttention(
            "non-finite attention scale was accepted".into(),
        ));
    };

    assert!(error.to_string().contains("scale must be finite and positive"));
    Ok(())
}

#[test]
fn applies_array_masks_and_attention_sinks_without_host_inputs() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let queries = Array::from_slice(&[0.0_f32, 0.0], [1, 1, 2, 1])?;
    let keys = Array::from_slice(&[1.0_f32, 1.0], [1, 1, 2, 1])?;
    let values = Array::from_slice(&[1.0_f32, 3.0], [1, 1, 2, 1])?;
    let mask = Array::from_slice(&[0.0_f32, -1.0e9, 0.0, 0.0], [1, 1, 2, 2])?;
    let sinks = Array::from_slice(&[0.0_f32], [1])?;
    let graph = stream.graph();
    let masked = graph.scaled_dot_product_attention(
        &queries,
        &keys,
        &values,
        ScaledDotProductAttention {
            scale: 1.0,
            mask: AttentionMask::Array(&mask),
            sinks: None,
        },
    )?;
    let sunk = graph.scaled_dot_product_attention(
        &queries,
        &keys,
        &values,
        ScaledDotProductAttention {
            scale: 1.0,
            mask: AttentionMask::None,
            sinks: Some(&sinks),
        },
    )?;

    assert_eq!(stream.read::<f32>(&masked)?, vec![1.0, 2.0]);
    for value in stream.read::<f32>(&sunk)? {
        assert!((value - 4.0 / 3.0).abs() < 1.0e-6);
    }
    Ok(())
}
