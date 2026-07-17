#![allow(clippy::print_stdout)]

use mirtal::{Array, DType, Device, Dispatch, OutputSpec, Result, Shape};

mirtal::metal_kernel! {
    fn double {
        name: "mirtal_example_double",
        templates: [],
        inputs: [input: f32],
        outputs: [output: f32],
        source: file "examples/kernels/double.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

fn main() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[2.0_f32, 3.0, 5.0], [3])?;
    let outputs = [OutputSpec::new(Shape::new([input.len()])?, DType::Float32)];
    let dispatch = Dispatch::new([input.len(), 1, 1], [input.len(), 1, 1]);
    let [output] = double()?.dispatch(&stream, [&input], &outputs, &dispatch)?;

    println!("{:?}", stream.read::<f32>(&output)?);
    Ok(())
}
