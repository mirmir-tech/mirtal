use cxx::{SharedPtr, UniquePtr};
use mirtal_sys::{ffi, graph_callback};

use crate::{Array, Error, Graph, Result, Stream};

#[derive(Debug, Clone, Copy, Default)]
/// Controls how MLX compiles a graph function.
pub struct CompileOptions {
    /// Allows calls whose input shapes differ from those used during tracing.
    pub shapeless: bool,
}

impl CompileOptions {
    #[must_use]
    /// Creates options for shape-polymorphic compilation.
    pub const fn shapeless() -> Self {
        Self { shapeless: true }
    }
}

/// A reusable MLX graph compiled for fixed input and output arities.
pub struct Compiled<const INPUTS: usize, const OUTPUTS: usize> {
    raw: UniquePtr<ffi::Compiled>,
    stream_id: u64,
}

impl Stream {
    /// Traces and compiles `function`, binding the result to this stream.
    pub fn compile<const INPUTS: usize, const OUTPUTS: usize, Function>(
        &self,
        options: CompileOptions,
        function: Function,
    ) -> Result<Compiled<INPUTS, OUTPUTS>>
    where
        Function: for<'graph> Fn(Graph<'graph>, [Array; INPUTS]) -> Result<[Array; OUTPUTS]>
            + Send
            + 'static,
    {
        let callback = graph_callback(move |values, stream| {
            let inputs = callback_inputs::<INPUTS>(values)?;
            match function(Graph::from_native(stream), inputs) {
                Ok(outputs) => Ok(outputs.into_iter().map(|array| array.raw_clone()).collect()),
                Err(error) => Err(error.to_string()),
            }
        });
        let raw = ffi::new_compiled(callback, options.shapeless, self.native()?)?;
        if raw.is_null() {
            return Err(Error::NullHandle("compiled graph"));
        }
        Ok(Compiled { raw, stream_id: self.id() })
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> Compiled<INPUTS, OUTPUTS> {
    /// Calls the compiled graph with inputs on its creating stream.
    pub fn call(&self, stream: &Stream, inputs: [&Array; INPUTS]) -> Result<[Array; OUTPUTS]> {
        if self.stream_id != stream.id() {
            return Err(Error::StreamMismatch {
                expected: self.stream_id,
                actual: stream.id(),
            });
        }
        let mut native_inputs = ffi::new_arrays();
        for input in inputs {
            ffi::arrays_push(native_inputs.pin_mut(), input.native()?);
        }
        let raw = self.raw.as_ref().ok_or(Error::NullHandle("compiled graph"))?;
        let native_inputs = native_inputs.as_ref().ok_or(Error::NullHandle("compiled inputs"))?;
        let outputs = ffi::compiled_call(raw, native_inputs)?;
        output_array::<OUTPUTS>(native_arrays(&outputs)?, "compiled outputs")
    }

    pub(crate) fn native(&self) -> Result<&ffi::Compiled> {
        self.raw.as_ref().ok_or(Error::NullHandle("compiled graph"))
    }
}

impl<const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug for Compiled<INPUTS, OUTPUTS> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Compiled")
            .field("inputs", &INPUTS)
            .field("outputs", &OUTPUTS)
            .field("stream_id", &self.stream_id)
            .finish_non_exhaustive()
    }
}

fn callback_inputs<const LENGTH: usize>(
    values: Vec<SharedPtr<ffi::Array>>,
) -> std::result::Result<[Array; LENGTH], String> {
    let actual = values.len();
    let mut arrays = Vec::with_capacity(actual);
    for value in values {
        match Array::from_raw(value, "compiled callback input") {
            Ok(array) => arrays.push(array),
            Err(error) => return Err(error.to_string()),
        }
    }
    let Ok(arrays) = arrays.try_into() else {
        return Err(arity("compiled callback inputs", LENGTH, actual).to_string());
    };
    Ok(arrays)
}

fn native_arrays(arrays: &UniquePtr<ffi::Arrays>) -> Result<Vec<Array>> {
    let arrays = arrays.as_ref().ok_or(Error::NullHandle("array list"))?;
    (0..ffi::arrays_len(arrays))
        .map(|index| Array::from_raw(ffi::arrays_get(arrays, index)?, "array list item"))
        .collect()
}

fn output_array<const LENGTH: usize>(
    values: Vec<Array>,
    operation: &'static str,
) -> Result<[Array; LENGTH]> {
    let actual = values.len();
    let Ok(values) = values.try_into() else {
        return Err(arity(operation, LENGTH, actual));
    };
    Ok(values)
}

const fn arity(operation: &'static str, expected: usize, actual: usize) -> Error {
    Error::Arity { operation, expected, actual }
}
