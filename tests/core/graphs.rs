//! Graph operations, ordinary and `#[compiled]` graph definitions, and
//! quantized matmuls.

use mirtal::{Array, CompileOptions, DType, Device, GatherQmmOptions, Quantization, Result, Shape};

#[mirtal::compiled(shapeless)]
fn zero_gate(graph: mirtal::Graph<'_>, [gate, input]: [Array; 2]) -> Result<[Array; 1]> {
    let activated = graph.silu(&gate)?;
    Ok([graph.multiply(&activated, &input)?])
}

#[test]
fn compiles_an_ordinary_rust_graph_definition() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let compiled =
        stream.compile::<2, 1, _>(CompileOptions::shapeless(), |graph, [gate, input]| {
            let activated = graph.silu(&gate)?;
            Ok([graph.multiply(&activated, &input)?])
        })?;
    let gate = Array::from_slice(&[0.0_f32, 0.0], [1, 2])?;
    let input = Array::from_slice(&[3.0_f32, 4.0], [1, 2])?;
    let [output] = compiled.call(&stream, [&gate, &input])?;

    assert_eq!(stream.read::<f32>(&output)?, vec![0.0, 0.0]);
    Ok(())
}

#[test]
fn compiled_attribute_generates_a_typed_factory() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let compiled = compile_zero_gate(&stream)?;
    let gate = Array::from_slice(&[0.0_f32, 0.0], [2])?;
    let input = Array::from_slice(&[5.0_f32, 7.0], [2])?;
    let [output] = compiled.call(&stream, [&gate, &input])?;

    assert_eq!(stream.read::<f32>(&output)?, vec![0.0, 0.0]);
    Ok(())
}

#[test]
fn keeps_compiled_graph_owned_after_factory_borrow_ends() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let compiled = { compile_zero_gate(&stream)? };
    let gate = Array::from_slice(&[1.0_f32, 2.0], [2])?;
    let input = Array::from_slice(&[2.0_f32, 4.0], [2])?;
    let [output] = compiled.call(&stream, [&gate, &input])?;

    let values = stream.read::<f32>(&output)?;
    assert!((values[0] - 1.462_117_2).abs() < 1.0e-5);
    assert!((values[1] - 7.046_376_7).abs() < 1.0e-5);
    Ok(())
}

#[test]
fn applies_power_and_division_without_leaving_the_graph() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[2.0_f32, 4.0], [2])?;
    let divisor = Array::from_slice(&[2.0_f32], [])?;
    let squared = stream.graph().power_scalar(&input, 2.0)?;
    let output = stream.graph().divide(&squared, &divisor)?;

    assert_eq!(stream.read::<f32>(&output)?, vec![2.0, 8.0]);
    Ok(())
}

#[test]
fn applies_elementwise_and_shape_graph_operations() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let gate = Array::from_slice(&[0.0_f32, 0.0], [1, 2])?;
    let input = Array::from_slice(&[2.0_f32, 4.0], [1, 2])?;
    let graph = stream.graph();
    let output = graph.sigmoid_multiply(&gate, &input)?;
    let expanded = graph.expand_dims(&output, &[-2])?;
    let squeezed = graph.squeeze_axis(&expanded, -2)?;

    assert_eq!(expanded.shape()?.dimensions(), &[1, 1, 2]);
    assert_eq!(stream.read::<f32>(&squeezed)?, vec![1.0, 2.0]);
    let values = Array::from_slice(&[0_u32, 1, 2], [3])?;
    let one = graph.full(&Shape::new([])?, 1.0, DType::Uint32)?;
    let upper = graph.less(&values, &graph.add_scalar(&one, 1.0)?)?;
    let lower = graph.greater_equal(&values, &one)?;
    let output = graph.astype(&graph.logical_and(&lower, &upper)?, DType::Uint32)?;
    assert_eq!(stream.read::<u32>(&output)?, vec![0, 1, 0]);
    Ok(())
}

#[test]
fn executes_affine_int4_matmul_and_gather_qmm() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let graph = stream.graph();
    let format = Quantization::new(64, 4)?;
    let mut dense_values = vec![1.0_f32; 128];
    dense_values[64..].fill(2.0);
    let dense = Array::from_slice(&dense_values, [2, 1, 64])?;
    let quantized = graph.quantize(&dense, format)?;
    let input = Array::from_slice(&vec![1.0_f32; 128], [1, 1, 2, 1, 64])?;
    let indices = Array::from_slice(&[0_u32, 1], [1, 1, 2])?;
    let output = graph.gather_qmm(
        &input,
        quantized.as_ref(),
        &indices,
        GatherQmmOptions { transpose: true, sorted_indices: false },
    )?;

    let values = stream.read::<f32>(&output)?;
    assert!((values[0] - 64.0).abs() < 1.0e-3);
    assert!((values[1] - 128.0).abs() < 1.0e-3);
    Ok(())
}
