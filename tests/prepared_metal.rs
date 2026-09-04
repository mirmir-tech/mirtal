use std::{hint::black_box, io::Write, time::Instant};

use mirtal::{Array, DType, Device, Dispatch, OutputSpec, Result, Shape, TemplateArg};

const BUILD_ITERATIONS: usize = 1_000;
const ITERATIONS: usize = 100;
const SAMPLES: usize = 7;

#[test]
fn prepared_metal_does_not_retain_completed_input_arrays() -> Result<()> {
    const ELEMENTS: usize = 1024 * 1024;
    let stream = Device::gpu(0).new_stream()?;
    let kernel = sum_three()?;
    let outputs = [OutputSpec::new(Shape::new([ELEMENTS])?, DType::Float32)];
    let dispatch = Dispatch::new([ELEMENTS, 1, 1], [256, 1, 1])
        .templates([TemplateArg::dtype("T", DType::Float32), TemplateArg::int("WIDTH", 2)]);
    let mut prepared = kernel.prepare(&outputs, &dispatch)?;
    let baseline = mirtal::memory::stats()?.active;
    let output = {
        let input = Array::from_slice(&vec![1.0_f32; ELEMENTS], [ELEMENTS])?;
        let [output] = prepared.dispatch(&stream, [&input, &input, &input])?;
        output
    };
    assert!(
        stream
            .read::<f32>(&output)?
            .iter()
            .all(|value| (*value - 6.0).abs() < f32::EPSILON)
    );
    drop(output);
    stream.synchronize()?;
    mirtal::memory::clear_cache()?;
    assert!(mirtal::memory::stats()?.active <= baseline + 1024 * 1024);
    black_box(&prepared);
    Ok(())
}

mirtal::metal_kernel! {
    fn sum_three {
        name: "mirtal_bench_sum_three",
        templates: [T: dtype = f32, WIDTH: int = 2],
        inputs: [first: T, second: f32, third: f32],
        outputs: [output: T],
        source: inline r"
            uint index = thread_position_in_grid.x;
            output[index] = T(
                (float(first[index]) + second[index] + third[index]) * float(WIDTH)
            );
        ",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[test]
#[ignore = "synthetic launch benchmark"]
fn benchmarks_prepared_metal_launch() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let stream = Device::gpu(0).new_stream()?;
    let kernel = sum_three()?;
    let inputs = [
        Array::from_slice(&[1.0_f32, 2.0], [2])?,
        Array::from_slice(&[3.0_f32, 4.0], [2])?,
        Array::from_slice(&[5.0_f32, 6.0], [2])?,
    ];
    let outputs = [OutputSpec::new(Shape::new([2])?, DType::Float32)];
    let dispatch = Dispatch::new([2, 1, 1], [2, 1, 1])
        .templates([TemplateArg::dtype("T", DType::Float32), TemplateArg::int("WIDTH", 2)]);
    let mut prepared = kernel.prepare(&outputs, &dispatch)?;
    let references = [&inputs[0], &inputs[1], &inputs[2]];
    let mut dynamic_samples = Vec::with_capacity(SAMPLES);
    let mut prepared_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        if sample.is_multiple_of(2) {
            dynamic_samples.push(measure_sync(&stream, || {
                kernel.dispatch(&stream, references, &outputs, &dispatch)
            })?);
            prepared_samples
                .push(measure_sync(&stream, || prepared.dispatch(&stream, references))?);
        } else {
            prepared_samples
                .push(measure_sync(&stream, || prepared.dispatch(&stream, references))?);
            dynamic_samples.push(measure_sync(&stream, || {
                kernel.dispatch(&stream, references, &outputs, &dispatch)
            })?);
        }
    }
    let dynamic_build =
        measure_build(|| kernel.dispatch(&stream, references, &outputs, &dispatch))?;
    let prepared_build = measure_build(|| prepared.dispatch(&stream, references))?;
    writeln!(
        std::io::stderr().lock(),
        "prepared_metal.benchmark: samples={SAMPLES}, iterations={ITERATIONS}, dynamic={:.4}ms, prepared={:.4}ms, dynamic_build={dynamic_build:.3}us, prepared_build={prepared_build:.3}us",
        median(dynamic_samples),
        median(prepared_samples),
    )?;
    Ok(())
}

fn measure_sync(
    stream: &mirtal::Stream,
    mut launch: impl FnMut() -> Result<[Array; 1]>,
) -> Result<f64> {
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let [output] = launch()?;
        stream.eval(&output)?;
        stream.synchronize()?;
        black_box(output);
    }
    Ok(started.elapsed().as_secs_f64() * 1_000.0 / f64::from(u32::try_from(ITERATIONS)?))
}

fn measure_build(mut launch: impl FnMut() -> Result<[Array; 1]>) -> Result<f64> {
    let started = Instant::now();
    for _ in 0..BUILD_ITERATIONS {
        black_box(launch()?);
    }
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(u32::try_from(BUILD_ITERATIONS)?))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
}
