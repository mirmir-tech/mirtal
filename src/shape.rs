use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// A validated list of array dimensions.
pub struct Shape(Vec<usize>);

impl Shape {
    /// Creates a shape and rejects an overflowing element count.
    pub fn new(dimensions: impl IntoIterator<Item = usize>) -> Result<Self> {
        let shape = Self(dimensions.into_iter().collect());
        shape.elements()?;
        Ok(shape)
    }

    /// Returns the total number of elements described by this shape.
    pub fn elements(&self) -> Result<usize> {
        self.0.iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(*dimension).ok_or(Error::ShapeOverflow)
        })
    }

    #[must_use]
    /// Returns the dimensions in axis order.
    pub fn dimensions(&self) -> &[usize] {
        &self.0
    }

    pub(crate) fn native(&self) -> Result<Vec<i32>> {
        let native = self
            .0
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<std::result::Result<_, _>>()?;
        Ok(native)
    }
}

impl<const N: usize> TryFrom<[usize; N]> for Shape {
    type Error = Error;

    fn try_from(value: [usize; N]) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<&[usize]> for Shape {
    type Error = Error;

    fn try_from(value: &[usize]) -> Result<Self> {
        Self::new(value.iter().copied())
    }
}
