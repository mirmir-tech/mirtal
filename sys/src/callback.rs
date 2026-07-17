use crate::ffi;

type CallbackResult = std::result::Result<Vec<cxx::SharedPtr<ffi::Array>>, String>;
type Callback = dyn Fn(Vec<cxx::SharedPtr<ffi::Array>>, &ffi::Stream) -> CallbackResult + Send;

pub struct GraphCallback {
    function: Box<Callback>,
}

pub fn graph_callback(
    function: impl Fn(Vec<cxx::SharedPtr<ffi::Array>>, &ffi::Stream) -> CallbackResult + Send + 'static,
) -> Box<GraphCallback> {
    Box::new(GraphCallback { function: Box::new(function) })
}

pub fn invoke_graph(
    callback: &GraphCallback,
    inputs: cxx::UniquePtr<ffi::Arrays>,
) -> std::result::Result<cxx::UniquePtr<ffi::Arrays>, String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let stream = native_stream(&inputs)?;
        let values = native_arrays(&inputs)?;
        (callback.function)(values, stream)
    }));
    drop(inputs);
    let values = match result {
        Ok(values) => values?,
        Err(_) => return Err("compiled Rust graph callback panicked".into()),
    };
    let mut outputs = ffi::new_arrays();
    for value in values {
        let value = value.as_ref().ok_or_else(|| "graph returned a null array".to_owned())?;
        ffi::arrays_push(outputs.pin_mut(), value);
    }
    Ok(outputs)
}

fn native_arrays(
    arrays: &cxx::UniquePtr<ffi::Arrays>,
) -> std::result::Result<Vec<cxx::SharedPtr<ffi::Array>>, String> {
    let arrays = arrays.as_ref().ok_or_else(|| "compiled graph inputs are null".to_owned())?;
    (0..ffi::arrays_len(arrays))
        .map(|index| ffi::arrays_get(arrays, index).map_err(|error| error.to_string()))
        .collect()
}

fn native_stream(
    arrays: &cxx::UniquePtr<ffi::Arrays>,
) -> std::result::Result<&ffi::Stream, String> {
    let arrays = arrays.as_ref().ok_or_else(|| "compiled graph inputs are null".to_owned())?;
    ffi::arrays_stream(arrays).map_err(|error| error.to_string())
}
