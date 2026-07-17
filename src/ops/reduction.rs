use mirtal_sys::ops::ffi;

use crate::{Array, Graph, Result};

impl Graph<'_> {
    /// Computes softmax along `axis`, optionally using precise accumulation.
    pub fn softmax(self, input: &Array, axis: i32, precise: bool) -> Result<Array> {
        Array::from_raw(ffi::softmax(input.native()?, axis, precise, self.native()?)?, "softmax")
    }

    /// Computes log-sum-exp along `axis`.
    pub fn logsumexp(self, input: &Array, axis: i32, keepdims: bool) -> Result<Array> {
        Array::from_raw(
            ffi::logsumexp(input.native()?, axis, keepdims, self.native()?)?,
            "logsumexp",
        )
    }

    /// Computes a cumulative sum along `axis`.
    pub fn cumulative_sum(
        self,
        input: &Array,
        axis: i32,
        reverse: bool,
        inclusive: bool,
    ) -> Result<Array> {
        Array::from_raw(
            ffi::cumulative_sum(input.native()?, axis, reverse, inclusive, self.native()?)?,
            "cumulative sum",
        )
    }

    /// Returns maximum values along `axis`.
    pub fn reduce_max(self, input: &Array, axis: i32, keepdims: bool) -> Result<Array> {
        Array::from_raw(
            ffi::reduce_max(input.native()?, axis, keepdims, self.native()?)?,
            "reduce max",
        )
    }

    /// Returns sums along `axis`.
    pub fn reduce_sum(self, input: &Array, axis: i32, keepdims: bool) -> Result<Array> {
        Array::from_raw(
            ffi::reduce_sum(input.native()?, axis, keepdims, self.native()?)?,
            "reduce sum",
        )
    }

    /// Returns indices of maximum values along `axis`.
    pub fn argmax_axis(self, input: &Array, axis: i32, keepdims: bool) -> Result<Array> {
        Array::from_raw(
            ffi::argmax_axis(input.native()?, axis, keepdims, self.native()?)?,
            "argmax",
        )
    }
}
