#![allow(clippy::print_stdout)]

use mirtal::{Array, Device, Graph, Result};

#[mirtal::compiled(shapeless)]
fn swiglu(graph: Graph<'_>, [gate, value]: [Array; 2]) -> Result<[Array; 1]> {
    let activated = graph.silu(&gate)?;
    Ok([graph.multiply(&activated, &value)?])
}

fn main() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let compiled = compile_swiglu(&stream)?;
    let gate = Array::from_slice(&[0.0_f32, 1.0], [2])?;
    let value = Array::from_slice(&[2.0_f32, 4.0], [2])?;
    let [output] = compiled.call(&stream, [&gate, &value])?;

    println!("{:?}", stream.read::<f32>(&output)?);
    Ok(())
}
