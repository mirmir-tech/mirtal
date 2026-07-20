use mirtal_sys::ffi;

use crate::{Array, Error, Graph, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Symmetric packed format for accelerator-resident runtime state.
pub struct SymmetricQuantization {
    bits: u8,
}

impl SymmetricQuantization {
    /// Creates a signed symmetric format packed into 32-bit words.
    pub fn new(bits: u8) -> Result<Self> {
        if !matches!(bits, 4 | 8) {
            return Err(Error::InvalidQuantization(
                "symmetric packing supports 4-bit or 8-bit elements".into(),
            ));
        }
        Ok(Self { bits })
    }

    #[must_use]
    /// Returns the encoded element width.
    pub const fn bits(self) -> u8 {
        self.bits
    }

    #[must_use]
    /// Returns the number of encoded elements in one 32-bit word.
    pub const fn elements_per_word(self) -> usize {
        32 / self.bits as usize
    }

    #[must_use]
    /// Returns the number of 32-bit words needed for `elements` values.
    pub const fn packed_words(self, elements: usize) -> usize {
        elements.div_ceil(self.elements_per_word())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Affine quantization format shared by weights, scales, and biases.
pub struct Quantization {
    group_size: i32,
    bits: i32,
}

impl Quantization {
    /// Creates a validated affine quantization format.
    pub fn new(group_size: i32, bits: i32) -> Result<Self> {
        if group_size <= 0 {
            return Err(Error::InvalidQuantization("group size must be positive".into()));
        }
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Error::InvalidQuantization("unsupported affine bit width".into()));
        }
        Ok(Self { group_size, bits })
    }
}

#[derive(Debug, Clone, Copy)]
/// Borrowed arrays that form an affine-quantized tensor.
pub struct Quantized<'array> {
    /// Packed quantized weights.
    pub weight: &'array Array,
    /// Per-group scales.
    pub scales: &'array Array,
    /// Per-group biases.
    pub biases: &'array Array,
    /// Encoding shared by the three arrays.
    pub format: Quantization,
}

#[derive(Debug, Clone)]
/// Owned arrays that form an affine-quantized tensor.
pub struct QuantizedArrays {
    /// Packed quantized weights.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group biases.
    pub biases: Array,
    /// Encoding shared by the three arrays.
    pub format: Quantization,
}

impl QuantizedArrays {
    #[must_use]
    /// Borrows all component arrays as a [`Quantized`] value.
    pub const fn as_ref(&self) -> Quantized<'_> {
        Quantized {
            weight: &self.weight,
            scales: &self.scales,
            biases: &self.biases,
            format: self.format,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
/// Options for gathered quantized matrix multiplication.
pub struct GatherQmmOptions {
    /// Treats the quantized right-hand matrix as transposed.
    pub transpose: bool,
    /// Indicates that `rhs_indices` is already sorted.
    pub sorted_indices: bool,
}

impl Graph<'_> {
    /// Quantizes an array into affine-packed weights, scales, and biases.
    pub fn quantize(self, input: &Array, format: Quantization) -> Result<QuantizedArrays> {
        let outputs =
            ffi::quantize(input.native()?, format.group_size, format.bits, self.native()?)?;
        let outputs = outputs.as_ref().ok_or(Error::NullHandle("quantized arrays"))?;
        let actual = ffi::arrays_len(outputs);
        if actual != 3 {
            return Err(Error::Arity {
                operation: "quantize",
                expected: 3,
                actual,
            });
        }
        Ok(QuantizedArrays {
            weight: array(outputs, 0, "quantized weight")?,
            scales: array(outputs, 1, "quantized scales")?,
            biases: array(outputs, 2, "quantized biases")?,
            format,
        })
    }

    /// Multiplies an array by an affine-quantized matrix.
    pub fn quantized_matmul(
        self,
        input: &Array,
        quantized: Quantized<'_>,
        transpose: bool,
    ) -> Result<Array> {
        let format = quantized.format;
        let options = native_options(format, transpose, false);
        Array::from_raw(
            ffi::quantized_matmul(
                input.native()?,
                quantized.weight.native()?,
                quantized.scales.native()?,
                quantized.biases.native()?,
                &options,
                self.native()?,
            )?,
            "quantized matmul",
        )
    }

    /// Selects quantized matrices by index and multiplies them by `input`.
    pub fn gather_qmm(
        self,
        input: &Array,
        quantized: Quantized<'_>,
        rhs_indices: &Array,
        options: GatherQmmOptions,
    ) -> Result<Array> {
        let format = quantized.format;
        let options = native_options(format, options.transpose, options.sorted_indices);
        Array::from_raw(
            ffi::gather_qmm(
                input.native()?,
                quantized.weight.native()?,
                quantized.scales.native()?,
                quantized.biases.native()?,
                rhs_indices.native()?,
                &options,
                self.native()?,
            )?,
            "gather qmm",
        )
    }

    /// Reconstructs a floating-point array from affine-quantized components.
    pub fn dequantize(self, quantized: Quantized<'_>) -> Result<Array> {
        let options = native_options(quantized.format, false, false);
        Array::from_raw(
            ffi::dequantize(
                quantized.weight.native()?,
                quantized.scales.native()?,
                quantized.biases.native()?,
                &options,
                self.native()?,
            )?,
            "dequantize",
        )
    }
}

fn array(values: &ffi::Arrays, index: usize, name: &'static str) -> Result<Array> {
    Array::from_raw(ffi::arrays_get(values, index)?, name)
}

const fn native_options(
    format: Quantization,
    transpose: bool,
    sorted_indices: bool,
) -> ffi::QuantizationOptions {
    ffi::QuantizationOptions {
        group_size: format.group_size,
        bits: format.bits,
        transpose,
        sorted_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::SymmetricQuantization;
    use crate::Result;

    #[test]
    fn sizes_symmetric_packed_words() -> Result<()> {
        let int8 = SymmetricQuantization::new(8)?;
        let int4 = SymmetricQuantization::new(4)?;
        assert_eq!(int8.packed_words(129), 33);
        assert_eq!(int4.packed_words(129), 17);
        assert!(SymmetricQuantization::new(3).is_err());
        Ok(())
    }
}
