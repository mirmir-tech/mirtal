use super::*;
use crate::{Device, PreparedAliasing};

crate::metal_library! {
    fn copy_library {
        name: "mirtal_rebind_validation_copy",
        source: file "tests/kernels/copy.metal",
    }
}

#[test]
fn invalid_rebind_preserves_constants_and_geometry() -> Result<()> {
    let mut dispatch =
        AliasingDispatch::new([1]).constants([7]).grid([2, 1, 1]).threadgroup([2, 1, 1]);
    for (grid, group) in invalid_geometry() {
        assert!(dispatch.rebind(&[9], grid, group).is_err(), "accepted {grid:?}/{group:?}");
        assert_eq!(dispatch.constants, [7]);
        assert_eq!(dispatch.grid, [2, 1, 1]);
        assert_eq!(dispatch.threadgroup, [2, 1, 1]);
    }
    assert!(dispatch.rebind(&[], [1, 1, 1], [1, 1, 1]).is_err());
    assert_eq!(dispatch.constants, [7]);
    dispatch.rebind(&[11], [1, 1, 1], [1, 1, 1])?;
    assert_eq!(dispatch.constants, [11]);
    assert_eq!(dispatch.grid, [1, 1, 1]);
    Ok(())
}

#[test]
fn prepared_plan_survives_rejected_geometry_and_keeps_lazy_outputs() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let library = copy_library()?;
    let function = library.export("copy_f32")?;
    let dispatch = AliasingDispatch::new([1]).grid([2, 1, 1]).threadgroup([2, 1, 1]);
    let mut plan: PreparedAliasing<2, 1> = function.prepare_aliasing(dispatch)?;
    let source = Array::from_slice(&[2.0f32, 5.0], [2])?;
    let target = Array::from_slice(&[0.0f32; 2], [2])?;
    let [pending] = plan.dispatch(&stream, [&source, &target])?;
    for (grid, group) in invalid_geometry() {
        assert!(plan.rebind(&[], grid, group).is_err());
        let invalid = AliasingDispatch::new([1]).grid(grid).threadgroup(group);
        assert!(function.prepare_aliasing::<2, 1>(invalid).is_err());
    }
    assert_eq!(stream.read::<f32>(&pending)?, [2.0, 5.0]);
    let next = Array::from_slice(&[11.0f32, 13.0], [2])?;
    let [result] = plan.dispatch(&stream, [&next, &pending])?;
    assert_eq!(stream.read::<f32>(&result)?, [11.0, 13.0]);
    plan.rebind(&[], [1, 1, 1], [1, 1, 1])?;
    let [result] = plan.dispatch(&stream, [&source, &result])?;
    assert_eq!(stream.read::<f32>(&result)?, [2.0, 13.0]);
    Ok(())
}

fn invalid_geometry() -> Vec<([usize; 3], [usize; 3])> {
    let mut cases = Vec::new();
    for value in [0, u32::MAX as usize + 1] {
        for axis in 0..3 {
            let mut dimension = [1, 1, 1];
            dimension[axis] = value;
            cases.push((dimension, [1, 1, 1]));
            cases.push(([1, 1, 1], dimension));
        }
    }
    cases
}
