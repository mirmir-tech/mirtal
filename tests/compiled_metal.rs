use mirtal::{Array, CompileOptions, DType, Device, Dispatch, OutputSpec, Result};

mirtal::metal_kernel! {
    fn inline_double {
        name: "compiled_inline_double",
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

#[test]
fn compiles_a_checked_metal_kernel_inside_a_rust_graph() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let kernel = inline_double()?;
    let compiled =
        stream.compile::<1, 1, _>(CompileOptions::default(), move |graph, [input]| {
            let shape = input.shape()?;
            kernel.dispatch_graph(
                graph,
                [&input],
                &[OutputSpec::new(shape, DType::Float32)],
                &Dispatch::new([input.len(), 1, 1], [input.len().min(256), 1, 1]),
            )
        })?;
    let input = Array::from_slice(&[2.0_f32, 3.0], [2])?;
    let [output] = compiled.call(&stream, [&input])?;

    assert_eq!(stream.read::<f32>(&output)?, vec![4.0, 6.0]);
    Ok(())
}
