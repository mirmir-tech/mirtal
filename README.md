# mirtal

Mirtal is a model-agnostic Rust execution layer for Apple MLX and Metal. It
provides the accelerator primitives used by `libmir` while keeping model,
generation, and application policy out of the hardware boundary.

Use mirtal when native Rust code needs explicit control over Apple accelerator
execution without exposing C++ ownership or depending on a Python runtime.

## What it provides

- safe arrays, devices, and explicit streams;
- lazy graph operations and compiled Rust graphs;
- accelerator-resident execution and explicit synchronization;
- safetensor loading and model-neutral tensor operations;
- build-time validation of custom Metal source;
- prepared kernels and reusable execution state;
- a private CXX bridge to native MLX.

## Add it to a project

```toml
[dependencies]
mirtal = "0.3.1"
```

Mirtal currently targets Apple Silicon on macOS. Building requires full Xcode,
the Metal toolchain component, and native MLX. Consumers should depend on
`mirtal`, not its private `mirtal-sys` or `mirtal-macros` implementation crates.

## Typical uses

- build model-neutral tensor and graph operations for Apple GPUs;
- compile and reuse explicit execution graphs;
- add checked custom Metal kernels to a Rust project;
- implement a higher-level inference or numerical library over MLX.

## Links

- [crates.io](https://crates.io/crates/mirtal)
- [Rust API documentation](https://docs.rs/mirtal)
- [Source and issues](https://github.com/mirmir-tech/mirtal)
- [MiRMiR architecture](https://mirmir.tech/modules)

Licensed under Apache-2.0.
