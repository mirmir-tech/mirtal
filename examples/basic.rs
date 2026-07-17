#![allow(clippy::print_stdout)]

use mirtal::{Array, Device, Result};

fn main() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let left = Array::from_slice(&[1.0_f32, 2.0], [1, 2])?;
    let right = Array::from_slice(&[3.0_f32, 4.0], [1, 2])?;
    let graph = stream.graph();
    let sum = graph.add(&left, &right)?;
    let output = graph.multiply_scalar(&sum, 0.5)?;

    println!("{:?}", stream.read::<f32>(&output)?);
    Ok(())
}
