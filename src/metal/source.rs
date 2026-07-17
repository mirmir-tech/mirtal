#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Statically owned Metal source and its diagnostic origin.
pub struct MetalSource {
    code: &'static str,
    origin: &'static str,
}

impl MetalSource {
    #[must_use]
    /// Creates a source value from code validated at build time.
    pub const fn new(code: &'static str, origin: &'static str) -> Self {
        Self { code, origin }
    }

    #[must_use]
    /// Creates an empty optional source fragment.
    pub const fn empty() -> Self {
        Self::new("", "<empty>")
    }

    #[must_use]
    /// Returns the Metal source code.
    pub const fn code(self) -> &'static str {
        self.code
    }

    #[must_use]
    /// Returns the source label used in diagnostics.
    pub const fn origin(self) -> &'static str {
        self.origin
    }
}
