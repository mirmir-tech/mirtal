use crate::{Result, Stream};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// A class of MLX execution device.
pub enum DeviceKind {
    /// A CPU device.
    Cpu,
    /// A Metal GPU device.
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Identifies an MLX execution device by kind and index.
pub struct Device {
    kind: DeviceKind,
    index: u32,
}

impl Device {
    #[must_use]
    /// Selects the CPU with `index`.
    pub const fn cpu(index: u32) -> Self {
        Self { kind: DeviceKind::Cpu, index }
    }

    #[must_use]
    /// Selects the GPU with `index`.
    pub const fn gpu(index: u32) -> Self {
        Self { kind: DeviceKind::Gpu, index }
    }

    /// Creates an explicit execution stream on this device.
    pub fn new_stream(self) -> Result<Stream> {
        Stream::new(self)
    }

    pub(crate) const fn native_kind(self) -> u8 {
        match self.kind {
            DeviceKind::Cpu => 0,
            DeviceKind::Gpu => 1,
        }
    }

    pub(crate) fn native_index(self) -> Result<i32> {
        Ok(i32::try_from(self.index)?)
    }
}
