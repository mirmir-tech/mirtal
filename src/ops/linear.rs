use mirtal_sys::ops::ffi;

use crate::{Array, DType, Error, Graph, Result, Shape};

impl Graph<'_> {
    /// Multiplies matrices, including batches with MLX broadcasting semantics.
    pub fn matmul(self, left: &Array, right: &Array) -> Result<Array> {
        Array::from_raw(ffi::matmul(left.native()?, right.native()?, self.native()?)?, "matmul")
    }

    /// Applies `LayerNorm` over the final axis with learned scale and bias.
    pub fn layer_norm(
        self,
        input: &Array,
        weight: &Array,
        bias: &Array,
        eps: f32,
    ) -> Result<Array> {
        if !eps.is_finite() || eps <= 0.0 {
            return Err(Error::InvalidOperation(
                "layer_norm epsilon must be finite and positive".into(),
            ));
        }
        Array::from_raw(
            ffi::layer_norm(
                input.native()?,
                weight.native()?,
                bias.native()?,
                eps,
                self.native()?,
            )?,
            "layer_norm",
        )
    }

    /// Applies tanh-approximated `GELU` with PyTorch-compatible constants.
    pub fn gelu_tanh(self, input: &Array) -> Result<Array> {
        let cube = self.power_scalar(input, 3.0)?;
        let correction = self.multiply_scalar(&cube, 0.044_715)?;
        let inner = self.add(input, &correction)?;
        let scaled = self.multiply_scalar(&inner, 0.797_884_6)?;
        let activated = self.add_scalar(&self.tanh(&scaled)?, 1.0)?;
        self.multiply_scalar(&self.multiply(input, &activated)?, 0.5)
    }

    /// Applies exact `GELU` using the Gaussian error function.
    pub fn gelu(self, input: &Array) -> Result<Array> {
        let scaled = self.multiply_scalar(input, std::f32::consts::FRAC_1_SQRT_2)?;
        let activated = self.add_scalar(&self.erf(&scaled)?, 1.0)?;
        self.multiply_scalar(&self.multiply(input, &activated)?, 0.5)
    }

    /// Normalizes vectors along `axis` using an L2 norm with a minimum divisor.
    pub fn l2_normalize(self, input: &Array, axis: i32, epsilon: f32) -> Result<Array> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidOperation(
                "L2 normalization epsilon must be finite and positive".into(),
            ));
        }
        let squares = self.power_scalar(input, 2.0)?;
        let sum = self.reduce_sum(&squares, axis, true)?;
        let norm = self.power_scalar(&sum, 0.5)?;
        let minimum = self.full(&Shape::new([])?, epsilon, DType::Float32)?;
        let divisor = self.maximum(&norm, &minimum)?;
        self.divide(input, &divisor)
    }
}
