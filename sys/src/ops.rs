#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mirtal/bridge.h");

        type Array = crate::ffi::Array;
        type Arrays = crate::ffi::Arrays;
        type Stream = crate::ffi::Stream;

        fn subtract(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn negative(input: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn exp(input: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn reciprocal(input: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn minimum(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn maximum(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn power(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn floor_divide(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn less(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn greater_equal(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn logical_and(left: &Array, right: &Array, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn arange(
            start: f32,
            stop: f32,
            stride: f32,
            dtype: u8,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn full(shape: &[i32], value: f32, dtype: u8, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn concatenate(inputs: &Arrays, axis: i32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn stack(inputs: &Arrays, axis: i32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn repeat(
            input: &Array,
            repeats: i32,
            axis: i32,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn conv1d(
            input: &Array,
            weight: &Array,
            stride: i32,
            padding: i32,
            dilation: i32,
            groups: i32,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn slice(
            input: &Array,
            start: &[i32],
            stop: &[i32],
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn slice_update(
            input: &Array,
            update: &Array,
            start: &[i32],
            stop: &[i32],
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn depends(
            input: &Array,
            dependencies: &Arrays,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn argpartition(
            input: &Array,
            kth: i32,
            axis: i32,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn argsort(input: &Array, axis: i32, stream: &Stream) -> Result<SharedPtr<Array>>;
        fn take(
            input: &Array,
            indices: &Array,
            axis: i32,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn take_along_axis(
            input: &Array,
            indices: &Array,
            axis: i32,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn softmax(
            input: &Array,
            axis: i32,
            precise: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn logsumexp(
            input: &Array,
            axis: i32,
            keepdims: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn cumulative_sum(
            input: &Array,
            axis: i32,
            reverse: bool,
            inclusive: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn reduce_max(
            input: &Array,
            axis: i32,
            keepdims: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn reduce_sum(
            input: &Array,
            axis: i32,
            keepdims: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
        fn argmax_axis(
            input: &Array,
            axis: i32,
            keepdims: bool,
            stream: &Stream,
        ) -> Result<SharedPtr<Array>>;
    }
}
