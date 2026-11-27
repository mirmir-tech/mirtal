//! Metal kernels declared with `metal_kernel!`: inline, included and
//! joined sources, prepared launches and declared dtypes.

use mirtal::{Array, DType, Device, Dispatch, OutputSpec, Result, Shape};

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
        header: inline "// Deliberately no terminal newline",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn included_double {
        name: "included_double",
        templates: [],
        inputs: [input: f32],
        outputs: [output: f32],
        source: file "tests/kernels/double.metal",
        header: inline "#include <metal_simdgroup_matrix>",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn joined_headers {
        name: "joined_headers",
        templates: [],
        inputs: [input: f32],
        outputs: [output: f32],
        source: inline "uint index = thread_position_in_grid.x;
output[index] = mirtal_offset(mirtal_scale(input[index]));",
        header: files ["tests/kernels/scale.h", "tests/kernels/offset.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[test]
fn joins_header_files_in_order() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[1.0_f32, -2.0], [2])?;
    let [output] = joined_headers()?.dispatch(
        &stream,
        [&input],
        &[OutputSpec::new(Shape::new([2])?, DType::Float32)],
        &Dispatch::new([2, 1, 1], [2, 1, 1]),
    )?;
    assert_eq!(stream.read::<f32>(&output)?, vec![4.0, -5.0]);
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
fn launches_inline_and_included_sources_with_unterminated_headers() -> Result<()> {
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
