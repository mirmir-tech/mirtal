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
mirtal = "0.6.0"
```

Mirtal currently targets Apple Silicon on macOS. Building requires full Xcode,
the Metal toolchain component, and native MLX 0.32.3. Set `MLX_PREFIX` to
the native installation prefix; Python is not required. Consumers should depend on
`mirtal`, not its private `mirtal-sys` or `mirtal-macros` implementation crates.

## Typical uses

- build model-neutral tensor and graph operations for Apple GPUs;
- compile and reuse explicit execution graphs;
- add checked custom Metal kernels to a Rust project;
- implement a higher-level inference or numerical library over MLX.

## Per-batch rotary positions

`Graph::rope_batched` and `rope_with_frequencies_batched` accept the existing
options with `offset: &device_offsets`. Use an integer array with one offset
per leading batch row; scalar and one-element arrays broadcast. Explicit
frequencies remain on device. These lazy operations use the graph's stream
without reading or synchronizing tensors. Scalar `rope` calls retain
`offset: usize` and their existing behavior.

## Indexed quantized multiplication

`Graph::gather_qmm_with_indices` (affine) and
`Graph::gather_mxfp4_with_indices` accept `GatherIndices { lhs, rhs }`.
Each optional selector addresses matrices in flattened batch dimensions;
the selector shapes broadcast together. Omitted selectors use native implicit
indexing. The existing `gather_qmm` and `gather_mxfp4` methods retain their
right-hand-only selection behavior.

These operations stay lazy, borrow device arrays, and use the graph's explicit
stream. Supplying LHS indices can avoid a caller-side gather, but changes native
kernel selection: MLX 0.32.2 disables its sorted-RHS batching shortcut when LHS
indices are explicit. Measure the complete operation before choosing a path.

## MXFP4 with FP32 accumulation

`MxFp4Matmul` is an explicit alternative for ordinary `input @ weight.T`.
Create a plan once and reuse it with BF16 input `[..., K]` (rank at least two),
packed U32 weights `[N, K/8]`, U8 scales `[N, K/32]`, and an explicit stream.
K must be divisible by 32. The caller supplies a nonzero partition count
that divides K/32; native MLX selection remains unchanged.

SIMD matrix accumulation, split-K storage, and reduction use FP32. The output
is BF16; add any bias afterward to preserve projection-rounding semantics.
With multiple partitions, temporary storage costs `4 * partitions * M * N`
bytes per projection, where M is the flattened input row count. One partition
writes the output directly without a partial buffer or reduction dispatch.
Execution stays lazy and adds no host reads or synchronization. Measure the
complete operation and its memory cost before selecting this path.

Use `matmul_graph` inside `Stream::compile` with `CompileOptions::default()`
to reuse the traced projection. Shapes specialize the dispatch geometry;
shapeless compilation is not supported for this operation. Pass input, weights
and scales as the three graph arguments. The compiled object remains bound to
its creating stream and retraces for new shapes. This avoids rebuilding custom
kernel source for each matching call, while preserving the same GPU kernels.
Prepared launch metadata alone does not avoid that native MLX work.

```rust,no_run
use std::num::NonZeroUsize;
use mirtal::{CompileOptions, Device, MxFp4, MxFp4Matmul};

let stream = Device::gpu(0).new_stream()?;
let plan = MxFp4Matmul::new()?;
let projection = stream.compile(CompileOptions::default(), move |graph, [x, w, s]| {
    Ok([plan.matmul_graph(
        graph, &x, MxFp4 { weight: &w, scales: &s }, NonZeroUsize::MIN,
    )?])
})?;
// projection.call(&stream, [&input, &weight, &scales])?;
# Ok::<(), mirtal::Error>(())
```

`MxFp4Matmul::with_row_tile(MxFp4RowTile::Rows32)` selects a 32×32 output tile
with four SIMD groups. The default `Rows16` uses a 16×32 tile and two groups.
Both keep BK32, the same per-output FP32 accumulation order and the same split-K
buffer. The larger row tile reuses weights across more rows, but increases
threadgroup memory from 3,840 to 5,120 bytes and may do more padded row work.
Selection is explicit and does not imply a speedup; measure the relevant shape.

Compilation is optional. Lower host construction cost does not guarantee lower
complete latency for every shape; keep selection and measurement caller-owned.

## Links

- [crates.io](https://crates.io/crates/mirtal)
- [Rust API documentation](https://docs.rs/mirtal)
- [Source and issues](https://github.com/mirmir-tech/mirtal)
- [MiRMiR architecture](https://mirmir.tech/modules)

Licensed under Apache-2.0.
