use mirtal_sys::ops::ffi;

use crate::{Array, Graph, Result};

macro_rules! binary {
    ($doc:literal, $name:ident, $native:ident) => {
        #[doc = $doc]
        pub fn $name(self, left: &Array, right: &Array) -> Result<Array> {
            Array::from_raw(
                ffi::$native(left.native()?, right.native()?, self.native()?)?,
                stringify!($name),
            )
        }
    };
}

macro_rules! unary {
    ($doc:literal, $name:ident, $native:ident) => {
        #[doc = $doc]
        pub fn $name(self, input: &Array) -> Result<Array> {
            Array::from_raw(ffi::$native(input.native()?, self.native()?)?, stringify!($name))
        }
    };
}

impl Graph<'_> {
    binary!("Subtracts two arrays with MLX broadcasting rules.", subtract, subtract);

    binary!("Returns the elementwise minimum of two arrays.", minimum, minimum);

    binary!("Returns the elementwise maximum of two arrays.", maximum, maximum);

    binary!(
        "Raises each element of `left` to the corresponding power in `right`.",
        power,
        power
    );

    binary!(
        "Divides and rounds each result toward negative infinity.",
        floor_divide,
        floor_divide
    );

    binary!("Compares two arrays elementwise with `<`.", less, less);

    binary!("Compares two arrays elementwise with `>=`.", greater_equal, greater_equal);

    binary!("Computes elementwise logical conjunction.", logical_and, logical_and);

    unary!("Negates every array element.", negative, negative);

    unary!("Applies the natural exponential elementwise.", exp, exp);

    unary!("Applies the Gauss error function elementwise.", erf, erf);

    unary!("Applies the cosine elementwise.", cos, cos);

    unary!("Applies the sine elementwise.", sin, sin);

    unary!("Computes the reciprocal of every array element.", reciprocal, reciprocal);

    /// Clips every value between broadcastable `minimum` and `maximum` arrays.
    pub fn clip(self, input: &Array, minimum: &Array, maximum: &Array) -> Result<Array> {
        Array::from_raw(
            ffi::clip(input.native()?, minimum.native()?, maximum.native()?, self.native()?)?,
            "clip",
        )
    }
}
