use mirtal_sys::{ffi as core, ops::ffi};

use crate::{Array, Error, Graph, Result};

impl Graph<'_> {
    /// Extracts a unit-stride slice bounded by per-axis start and stop indices.
    pub fn slice(self, input: &Array, start: &[usize], stop: &[usize]) -> Result<Array> {
        if start.len() != stop.len() || start.is_empty() {
            return Err(Error::InvalidOperation(
                "slice bounds must have the same non-zero rank".into(),
            ));
        }
        let start = dimensions(start)?;
        let stop = dimensions(stop)?;
        Array::from_raw(ffi::slice(input.native()?, &start, &stop, self.native()?)?, "slice")
    }

    /// Returns `input` with a unit-stride slice replaced by `update`.
    pub fn slice_update(
        self,
        input: &Array,
        update: &Array,
        start: &[usize],
        stop: &[usize],
    ) -> Result<Array> {
        if start.len() != stop.len() || start.is_empty() {
            return Err(Error::InvalidOperation(
                "slice-update bounds must have the same non-zero rank".into(),
            ));
        }
        Array::from_raw(
            ffi::slice_update(
                input.native()?,
                update.native()?,
                &dimensions(start)?,
                &dimensions(stop)?,
                self.native()?,
            )?,
            "slice update",
        )
    }

    /// Adds evaluation dependencies while returning the value of `input`.
    pub fn depends(self, input: &Array, dependencies: &[&Array]) -> Result<Array> {
        let mut native = core::new_arrays();
        for dependency in dependencies {
            core::arrays_push(native.pin_mut(), dependency.native()?);
        }
        Array::from_raw(
            ffi::depends(
                input.native()?,
                native.as_ref().ok_or(Error::NullHandle("dependencies"))?,
                self.native()?,
            )?,
            "depends",
        )
    }

    /// Returns indices that partition values around `kth` along `axis`.
    pub fn argpartition(self, input: &Array, kth: i32, axis: i32) -> Result<Array> {
        Array::from_raw(
            ffi::argpartition(input.native()?, kth, axis, self.native()?)?,
            "argpartition",
        )
    }

    /// Returns indices that sort values along `axis`.
    pub fn argsort(self, input: &Array, axis: i32) -> Result<Array> {
        Array::from_raw(ffi::argsort(input.native()?, axis, self.native()?)?, "argsort")
    }

    /// Selects values at `indices` along `axis`.
    pub fn take(self, input: &Array, indices: &Array, axis: i32) -> Result<Array> {
        Array::from_raw(
            ffi::take(input.native()?, indices.native()?, axis, self.native()?)?,
            "take",
        )
    }

    /// Selects per-position values using indices aligned with `input`.
    pub fn take_along_axis(self, input: &Array, indices: &Array, axis: i32) -> Result<Array> {
        Array::from_raw(
            ffi::take_along_axis(input.native()?, indices.native()?, axis, self.native()?)?,
            "take along axis",
        )
    }
}

fn dimensions(values: &[usize]) -> Result<Vec<i32>> {
    Ok(values
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<_, _>>()?)
}
