use mirtal::{AliasingDispatch, Array, Device, Error, PreparedAliasing, Result};

mirtal::metal_library! {
    fn copy_library {
        name: "mirtal_test_copy",
        source: file "tests/kernels/copy.metal",
    }
}

#[test]
fn retains_a_checked_full_metal_library_for_native_primitives() -> Result<()> {
    let library = copy_library()?;
    assert_ne!(mirtal::interop::metal_library_native_handle(&library)?, 0);
    Ok(())
}

#[test]
fn rebinds_an_aliasing_dispatch_without_rebuilding_its_storage() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let input = Array::from_slice(&[3.0_f32, 7.0], [2])?;
    let output = Array::from_slice(&[0.0_f32, 0.0], [2])?;
    let mut dispatch = AliasingDispatch::new([1]).grid([2, 1, 1]).threadgroup([2, 1, 1]);
    let [first] = copy_library()?.export("copy_f32")?.dispatch_aliasing_array(
        &stream,
        &[&input, &output],
        &dispatch,
    )?;
    assert_eq!(stream.read::<f32>(&first)?, vec![3.0, 7.0]);

    dispatch.rebind(&[], [1, 1, 1], [1, 1, 1])?;
    let [second] = copy_library()?.export("copy_f32")?.dispatch_aliasing_array(
        &stream,
        &[&input, &output],
        &dispatch,
    )?;
    assert!((stream.read::<f32>(&second)?[0] - 3.0).abs() < f32::EPSILON);
    assert!(matches!(
        dispatch.rebind(&[1], [1, 1, 1], [1, 1, 1]),
        Err(Error::InvalidDispatch(_))
    ));
    Ok(())
}

#[test]
fn prepares_static_aliasing_state_and_reuses_native_inputs() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let first = Array::from_slice(&[2.0_f32, 5.0], [2])?;
    let second = Array::from_slice(&[11.0_f32, 13.0], [2])?;
    let output = Array::from_slice(&[0.0_f32, 0.0], [2])?;
    let dispatch = AliasingDispatch::new([1]).grid([2, 1, 1]).threadgroup([2, 1, 1]);
    let mut prepared: PreparedAliasing<2, 1> =
        copy_library()?.export("copy_f32")?.prepare_aliasing(dispatch)?;

    let [copied] = prepared.dispatch(&stream, [&first, &output])?;
    assert_eq!(stream.read::<f32>(&copied)?, vec![2.0, 5.0]);
    prepared.rebind(&[], [2, 1, 1], [2, 1, 1])?;
    let [rebound] = prepared.dispatch(&stream, [&second, &copied])?;
    assert_eq!(stream.read::<f32>(&rebound)?, vec![11.0, 13.0]);
    Ok(())
}
