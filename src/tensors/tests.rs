use super::*;
use crate::DType;

fn reason(result: Result<TensorRange>) -> Option<TensorRangeError> {
    match result {
        Err(Error::TensorRange { reason, .. }) => Some(reason),
        _ => None,
    }
}

#[test]
fn a_range_is_valid_by_construction() -> Result<()> {
    let range = TensorRange::new("blk.0.ffn_up.weight", 64, DType::Float16, vec![3, 4])?;
    assert_eq!(
        (range.name(), range.offset(), range.dtype()),
        ("blk.0.ffn_up.weight", 64, DType::Float16)
    );
    assert_eq!((range.shape(), range.bytes(), range.end()), (&[3, 4][..], 24, 88));
    let new =
        |name: &str, offset, shape: Vec<u64>| TensorRange::new(name, offset, DType::Float32, shape);
    assert_eq!(reason(new("scalar", 0, vec![])), Some(TensorRangeError::Scalar));
    assert_eq!(reason(new("empty", 0, vec![2, 0])), Some(TensorRangeError::EmptyDimension));
    let wide = u64::try_from(i32::MAX)? + 1;
    assert_eq!(
        reason(new("wide", 0, vec![wide])),
        Some(TensorRangeError::DimensionTooLarge(wide))
    );
    let huge = u64::try_from(i32::MAX)?;
    assert_eq!(reason(new("huge", 0, vec![huge; 3])), Some(TensorRangeError::SizeOverflow));
    assert_eq!(reason(new("late", u64::MAX - 3, vec![1])), Some(TensorRangeError::SizeOverflow));
    assert_eq!(reason(new("a\nb", 0, vec![1])), Some(TensorRangeError::ControlCharacter));
    Ok(())
}
