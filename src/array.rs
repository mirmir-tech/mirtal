use std::{fmt, path::Path};

use cxx::SharedPtr;
use mirtal_sys::ffi;

use crate::{DType, Element, Error, Result, Shape, element};

#[derive(Clone)]
/// An immutable, lazily evaluated MLX array.
pub struct Array {
    raw: SharedPtr<ffi::Array>,
}

impl Array {
    /// Creates an array by copying a typed slice with the given shape.
    pub fn from_slice<T: Element, const N: usize>(data: &[T], shape: [usize; N]) -> Result<Self> {
        Self::from_shape(data, &Shape::try_from(shape)?)
    }

    /// Creates an array by copying a typed slice described by `shape`.
    pub fn from_shape<T: Element>(data: &[T], shape: &Shape) -> Result<Self> {
        let elements = shape.elements()?;
        if elements != data.len() {
            return Err(Error::Shape {
                shape: shape.dimensions().to_vec(),
                elements,
                data: data.len(),
            });
        }
        Self::from_raw(element::create(data, &shape.native()?)?, "array")
    }

    /// Returns the array shape reported by MLX.
    pub fn shape(&self) -> Result<Shape> {
        let dimensions = ffi::array_shape(self.native()?)?
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Shape::new(dimensions)
    }

    /// Returns the array element type.
    pub fn dtype(&self) -> Result<DType> {
        DType::try_from(ffi::array_dtype(self.native()?)?)
    }

    #[must_use]
    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.raw.as_ref().map_or(0, ffi::array_len)
    }

    #[must_use]
    /// Returns whether the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn from_raw(raw: SharedPtr<ffi::Array>, name: &'static str) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle(name));
        }
        Ok(Self { raw })
    }

    pub(crate) fn native(&self) -> Result<&ffi::Array> {
        self.raw.as_ref().ok_or(Error::NullHandle("array"))
    }

    pub(crate) fn raw_clone(&self) -> SharedPtr<ffi::Array> {
        self.raw.clone()
    }

    /// Writes the lazy computation graph rooted at this array in DOT format.
    pub fn export_graph_dot(&self, path: &Path) -> Result<()> {
        let path = path.to_str().ok_or_else(|| Error::InvalidPath(path.to_path_buf()))?;
        Ok(mirtal_sys::graph::ffi::export_graph_dot(self.native()?, path)?)
    }

    /// Schedules evaluation without waiting for completion.
    pub fn async_eval(&self) -> Result<()> {
        Ok(ffi::array_eval(self.native()?)?)
    }

    /// Retains evaluated storage while releasing the lazy graph that produced it.
    ///
    /// The caller must ensure evaluation has completed before detaching the graph.
    pub fn detach_graph(&self) -> Result<()> {
        Ok(ffi::array_detach(self.native()?)?)
    }

    /// Evaluates the array and copies its values to the host as `f32`.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let mut output = vec![0.0; self.len()];
        mirtal_sys::read::ffi::array_read_f32(self.native()?, &mut output)?;
        Ok(output)
    }

    /// Evaluates a scalar array and copies its value to the host as `u32`.
    pub fn item_u32(&self) -> Result<u32> {
        Ok(mirtal_sys::read::ffi::array_read_u32_scalar(self.native()?)?)
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish_non_exhaustive()
    }
}
