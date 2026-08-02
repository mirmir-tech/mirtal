use mirtal::{
    Array, CompileOptions, DType, Device, Dispatch, GatherQmmOptions, OutputSpec, Quantization,
    Result, Shape,
};

mirtal::metal_kernel! {
    fn inline_double {
        name: "inline_double",
        templates: [],
        inputs: [input: f32],
        outputs: [output: f32],
        source: inline r"
            uint index = thread_position_in_grid.x;
            output[index] = input[index] * 2.0f;
        ",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[mirtal::compiled(shapeless)]
fn zero_gate(graph: mirtal::Graph<'_>, [gate, input]: [Array; 2]) -> Result<[Array; 1]> {
    let activated = graph.silu(&gate)?;
    Ok([graph.multiply(&activated, &input)?])
}

mirtal::metal_kernel! {
    fn included_double {
        name: "included_double",
        templates: [],
        inputs: [input: f32],
        outputs: [output: f32],
        source: file "tests/kernels/double.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

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
fn reuses_a_prepared_metal_launch_with_new_inputs() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let output = [OutputSpec::new(Shape::new([2])?, DType::Float32)];
    let dispatch = Dispatch::new([2, 1, 1], [2, 1, 1]);
    let mut prepared = inline_double()?.prepare(&output, &dispatch)?;
    let first = Array::from_slice(&[2.0_f32, 3.0], [2])?;
    let second = Array::from_slice(&[5.0_f32, 7.0], [2])?;

    let [first] = prepared.dispatch(&stream, [&first])?;
    assert_eq!(stream.read::<f32>(&first)?, vec![4.0, 6.0]);
    let [second] = prepared.dispatch(&stream, [&second])?;
    assert_eq!(stream.read::<f32>(&second)?, vec![10.0, 14.0]);
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
fn converts_device_dtype_during_an_explicit_host_read() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1.5_f32, -2.25], [2])?;
    let bfloat = stream.graph().astype(&input, DType::Bfloat16)?;

    assert_eq!(stream.read::<f32>(&bfloat)?, vec![1.5, -2.25]);
    Ok(())
}

#[test]
fn launches_inline_and_included_metal_sources() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[2.0_f32, 3.0], [2])?;
    for kernel in [inline_double()?, included_double()?] {
        let [output] = kernel.dispatch(
            &stream,
            [&input],
            &[OutputSpec::new(Shape::try_from([2])?, DType::Float32)],
            &Dispatch::new([2, 1, 1], [2, 1, 1]),
        )?;
        assert_eq!(stream.read::<f32>(&output)?, vec![4.0, 6.0]);
    }
    Ok(())
}

#[test]
fn rejects_metal_dispatch_with_an_undeclared_dtype() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[2_u32], [1])?;
    let result = inline_double()?.dispatch(
        &stream,
        [&input],
        &[OutputSpec::new(Shape::try_from([1])?, DType::Float32)],
        &Dispatch::new([1, 1, 1], [1, 1, 1]),
    );
    let Err(error) = result else {
        return Err(mirtal::Error::InvalidDispatch(
            "Metal accepted an undeclared input dtype".into(),
        ));
    };
    assert!(error.to_string().contains("declared Float32"));
    Ok(())
}

#[test]
fn reports_native_version_and_allocator_state() -> Result<()> {
    assert!(!mirtal::version()?.is_empty());
    let stats = mirtal::memory::stats()?;
    assert!(stats.limit > 0);
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
