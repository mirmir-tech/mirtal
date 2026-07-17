use cxx::SharedPtr;
use mirtal_sys::ffi;

use crate::{Result, Stream};

mod sealed {
    use cxx::SharedPtr;
    use mirtal_sys::ffi;

    type NativeResult<T> = std::result::Result<T, cxx::Exception>;

    pub trait Sealed: Copy + Sized {
        fn create(data: &[Self], shape: &[i32]) -> NativeResult<SharedPtr<ffi::Array>>;
        fn copy(array: &ffi::Array, stream: &ffi::Stream, output: &mut [Self]) -> NativeResult<()>;
    }

    impl Sealed for f32 {
        fn create(data: &[Self], shape: &[i32]) -> NativeResult<SharedPtr<ffi::Array>> {
            ffi::array_from_f32(data, shape)
        }

        fn copy(array: &ffi::Array, stream: &ffi::Stream, output: &mut [Self]) -> NativeResult<()> {
            ffi::array_copy_f32(array, stream, output)
        }
    }

    impl Sealed for u32 {
        fn create(data: &[Self], shape: &[i32]) -> NativeResult<SharedPtr<ffi::Array>> {
            ffi::array_from_u32(data, shape)
        }

        fn copy(array: &ffi::Array, stream: &ffi::Stream, output: &mut [Self]) -> NativeResult<()> {
            ffi::array_copy_u32(array, stream, output)
        }
    }
}

/// A host element type that can be copied to and from an MLX array.
pub trait Element: sealed::Sealed + Copy + Default + Send + Sync + 'static {}

impl Element for f32 {}

impl Element for u32 {}

pub fn create<T: Element>(data: &[T], shape: &[i32]) -> Result<SharedPtr<ffi::Array>> {
    Ok(<T as sealed::Sealed>::create(data, shape)?)
}

pub fn copy<T: Element>(array: &crate::Array, stream: &Stream) -> Result<Vec<T>> {
    let mut output = vec![T::default(); array.len()];
    <T as sealed::Sealed>::copy(array.native()?, stream.native()?, &mut output)?;
    Ok(output)
}
