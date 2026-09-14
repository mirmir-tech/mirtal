use std::collections::HashSet;

use mirtal::{Array, Device, Result};

#[test]
fn counts_full_storage_once_for_views_and_keeps_it_alive() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let parent = Array::from_slice(&[1.0_f32; 1024], [1024])?;
    let view = stream.graph().slice(&parent, &[100], &[103])?;
    assert!(view.allocation()?.is_none(), "metadata query must not evaluate");
    stream.eval(&view)?;
    stream.synchronize()?;
    let allocation = parent.allocation()?.ok_or(mirtal::Error::NullHandle("available parent"))?;
    let sliced = view.allocation()?.ok_or(mirtal::Error::NullHandle("available view"))?;
    assert_eq!(allocation, sliced);
    assert!(sliced.bytes() >= 1024 * size_of::<f32>());
    let mut unique = HashSet::from([allocation.clone(), sliced]);
    assert_eq!(unique.len(), 1);
    drop(parent);
    drop(view);
    let other = Array::from_slice(&[2.0_f32; 1024], [1024])?;
    unique.insert(other.allocation()?.ok_or(mirtal::Error::NullHandle("available array"))?);
    assert_eq!(unique.len(), 2, "retained allocation cannot be recycled");
    assert!(allocation.bytes() > 0);
    Ok(())
}
