# mirtal

Mirtal is a native Rust execution gateway for Apple MLX and Metal. It provides
explicit devices and streams, lazy tensor graphs, MLX graph compilation,
safetensor loading, and compile-time checked custom Metal kernels without
exposing C++ ownership to its consumers.

Mirtal is intentionally model-agnostic. It does not parse model configuration,
render prompts, own K/V policy, schedule inference requests, or implement an
HTTP server. Those responsibilities belong to libraries such as `libmir`.

## Status

Mirtal is pre-1.0 and currently targets Apple Silicon on macOS. Its safe Rust
surface is already used for complete local LLM execution, but API changes may
still occur while native interop is reduced and execution contracts stabilize.

The runtime path contains no Python, `PyTorch`, `NumPy`, or Transformers dependency.
`mirtal-sys` is the private CXX/C++20 bridge to MLX; applications depend only on
`mirtal` and never receive native smart pointers through the safe API.

## Requirements

- Apple Silicon and macOS;
- full Xcode selected through `xcode-select`, including Clang with C++20 and the
  macOS SDK;
- the Xcode Metal toolchain component;
- native MLX installed through Homebrew or selected with `MLX_PREFIX`;
- rustup and the pinned `nightly-2026-07-13` toolchain with Clippy and rustfmt.

Prepare and verify the Apple toolchain before building:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
xcrun --sdk macosx --find clang++
xcrun --sdk macosx --find metal
brew install mlx
```

Mirtal requires `include/mlx/mlx.h`, `lib/libmlx.dylib`, and
`lib/mlx.metallib`. It searches `/opt/homebrew/opt/mlx` and
`/usr/local/opt/mlx`; set `MLX_PREFIX` for another installation. Its procedural
macros invoke the Metal compiler during `cargo build`, so standalone Xcode
command-line tools without the downloaded component are insufficient.

`Python`, `PyTorch`, `NumPy`, `Transformers`, `mlx_lm`, model checkpoints, and `Trunk` are
not required to compile or use mirtal.

## Dependency

```toml
[dependencies]
mirtal = "0.1"
```

Consumers should not depend directly on `mirtal-sys` or `mirtal-macros`.

## Basic Graph

Every operation is associated with an explicit stream. Graph construction is
lazy; `eval`, `read`, `read_scalar_u32`, `async_eval`, and `synchronize` are the
observable execution boundaries.

```rust,no_run
use mirtal::{Array, Device, Result};

fn main() -> Result<()> {
    let stream = Device::gpu(0).new_stream()?;
    let left = Array::from_slice(&[1.0_f32, 2.0], [1, 2])?;
    let right = Array::from_slice(&[3.0_f32, 4.0], [1, 2])?;
    let output = stream.graph().add(&left, &right)?;

    assert_eq!(stream.read::<f32>(&output)?, vec![4.0, 6.0]);
    Ok(())
}
```

`Array::clone` shares the native MLX array handle and tensor storage. It does
not copy the tensor. Host data enters through `Array::from_slice` or
`Array::from_shape`; host data leaves through an explicit stream read.

## Compiled Rust Graphs

`Stream::compile` traces a typed Rust closure through MLX. The `compiled`
attribute generates a named factory while keeping graph composition as ordinary
Rust code:

```rust,no_run
use mirtal::{Array, Graph, Result};

#[mirtal::compiled(shapeless)]
fn swiglu(graph: Graph<'_>, [gate, value]: [Array; 2]) -> Result<[Array; 1]> {
    let activated = graph.silu(&gate)?;
    Ok([graph.multiply(&activated, &value)?])
}

# fn prepare(stream: &mirtal::Stream) -> Result<()> {
let compiled = compile_swiglu(stream)?;
# let gate = Array::from_slice(&[0.0_f32], [1])?;
# let value = Array::from_slice(&[1.0_f32], [1])?;
let [output] = compiled.call(stream, [&gate, &value])?;
# drop(output);
# Ok(())
# }
```

`Compiled<INPUTS, OUTPUTS>` records the creating stream and rejects calls made
with another stream. Compilation does not imply a host read or synchronization.

## Checked Metal Kernels

`metal_kernel!` declares input/output arity and dtype constraints in Rust. Its
source can be inline or loaded from a manifest-relative `.metal` file. During
the Rust build, the macro invokes `xcrun metal -fsyntax-only`, turning invalid
MSL into a compile error at the macro invocation.

```rust,ignore
mirtal::metal_kernel! {
    fn double {
        name: "double",
        templates: [],
        inputs: [input: f32],
        outputs: [output: f32],
        source: file "examples/kernels/double.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}
```

Create `OutputSpec` and `Dispatch` explicitly, then call
`MetalKernel::dispatch`. When output shapes, templates, and launch geometry stay
constant, `MetalKernel::prepare` creates a `PreparedMetalKernel` that reuses
native launch storage while accepting new arrays.

Complete MSL translation units use `metal_library!`. `MetalLibrary::export`
returns a named `MetalFunction`; functions that mutate or reuse existing output
allocations use `AliasingDispatch` or typed `PreparedAliasing`. Aliases,
constants, stride bindings, grid size, and threadgroup size remain explicit in
safe Rust.

## Execution Contract

- `Device` selects CPU or GPU and creates a `Stream`.
- `Stream` is the execution and synchronization context and is deliberately not
  `Sync`; each independent execution session should own its stream state.
- `Graph<'stream>` borrows a stream, preventing accidental implicit-device
  operations.
- MLX operations remain lazy until an explicit evaluation boundary.
- `Array` is immutable and cheap to clone; aliases share storage.
- Kernel dispatch returns lazy device arrays and does not read them into Rust.
- `interop` is a transitional unsafe/native escape hatch, not an application
  API for normal execution.

Keeping decode loops on the accelerator requires retaining arrays, compiled
graphs, prepared launches, and one stable stream rather than reading values or
rebuilding static launch descriptions per token.

## Public API

### Runtime and storage

| API | Purpose |
|---|---|
| `Device`, `DeviceKind` | Select an MLX CPU/GPU device and create a stream. |
| `Stream` | Build graphs, evaluate, synchronize, and explicitly read host values. |
| `Array` | Shared RAII tensor handle, metadata, async evaluation, and graph export. |
| `Shape`, `shape!` | Validated dimensions and checked element counts. |
| `DType`, `Element` | Device dtypes and supported host transfer types (`f32`, `u32`). |
| `TensorFile` | Load, inspect, evaluate, and retrieve arrays from safetensors. |
| `memory` | Allocator statistics, cache clearing, and recommended wired limit. |
| `version` | Report the linked MLX version. |
| `Error`, `Result` | Typed `thiserror` boundary for validation and native failures. |

### Graph operations

`Graph` exposes these lazy operation groups:

| Group | Methods |
|---|---|
| Arithmetic | `add`, `subtract`, `multiply`, `divide`, `power`, `floor_divide`, scalar add/multiply/power, `negative`, `exp`, `reciprocal`, `minimum`, `maximum`. |
| Activations and norm | `sigmoid`, `sigmoid_multiply`, `silu`, `tanh`, `rms_norm`, `rms_norm_unit`. |
| Shape and dtype | `astype`, `reshape`, `transpose`, `expand_dims`, `squeeze_axis`. |
| Creation | `arange`, `full`, `concatenate`, `stack`, `repeat`, `conv1d`. |
| Reduction | `softmax`, `logsumexp`, `cumulative_sum`, `reduce_max`, `reduce_sum`, `argmax_axis`. |
| Selection | `slice`, `slice_update`, `take`, `take_along_axis`, `argpartition`, `argsort`, `depends`. |
| Logic | `less`, `greater_equal`, `logical_and`. |
| Quantized | `quantize`, `dequantize`, `quantized_matmul`, `gather_qmm`. |
| Transformer primitives | `rope`, `rope_with_frequencies`, `scaled_dot_product_attention`. |

Model-specific tensor layouts, attention policy, expert routing, K/V paging, and
sampling are composed by the consumer from these operations.

### Compilation and Metal

| API | Purpose |
|---|---|
| `CompileOptions`, `Compiled`, `compiled` | Typed MLX compilation of Rust graph functions. |
| `metal_kernel!`, `MetalKernel`, `KernelDescriptor` | Compile-time checked fixed-arity custom kernels. |
| `Dispatch`, `OutputSpec` | Grid, threadgroup, templates, initialization, output shape, and dtype. |
| `TemplateArg`, `TemplateValue`, `TemplateParameter`, `TemplateKind` | Typed Metal specialization arguments and declarations. |
| `DTypeConstraint` | Exact, floating-point, or template-bound tensor dtype validation. |
| `PreparedMetalKernel` | Reusable native launch plan for ordinary kernels. |
| `metal_library!`, `MetalLibrary`, `MetalLibraryDescriptor` | Checked complete MSL libraries and retained MLX library cache. |
| `MetalFunction`, `AliasingDispatch`, `StrideBinding` | Named functions with explicit persistent-output aliasing. |
| `PreparedAliasing` | Reusable typed aliasing pipeline, metadata, and input storage. |
| `MetalSource` | Dynamic low-level source escape hatch without macro-time validation. |

### Typed options

`Quantization`, `Quantized`, `QuantizedArrays`, and `GatherQmmOptions` describe
affine quantized execution without loosely ordered scalar arguments.
`RopeOptions`, `FrequencyRopeOptions`, `AttentionMask`, and
`ScaledDotProductAttention` make transformer primitive configuration explicit.

### Native interop

The public `interop` module exposes native handles only for staged migration of
existing native integrations. Handle lifetimes remain tied to their Rust
owners; adoption is unsafe and single-use. New consumers should request a safe,
generic mirtal operation instead of building execution around native addresses.

## Examples

- `cargo run --example basic` builds and reads a lazy graph.
- `cargo run --example compiled` compiles and reuses Rust graph code.
- `cargo run --example metal_kernel` launches an included, checked MSL kernel.

## Documentation

```sh
make docs
make docs-open
```

These targets build rustdoc for `mirtal`, `mirtal-macros`, and `mirtal-sys`
with warnings denied. This README is embedded as the public crate landing page;
rustdoc provides signatures and implementors for every exported item.

## Validation

```sh
make examples
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

## Releases

The Forgejo workflow in `.forgejo/workflows/publish.yml` is currently disabled
with a job-level `if: false` guard, so it does not allocate a runner. Once the
guard is removed, it validates every push to `main` and publishes `mirtal-sys`,
`mirtal-macros`, and `mirtal`, in that order, when the version under
`[workspace.package]` is not present on crates.io. Pushes that retain an already
published version are validated and then skipped.

Publishing requires a self-hosted Apple Silicon Forgejo runner labeled
`macos-arm64` with Xcode, the Metal toolchain, Homebrew MLX, and rustup. The
Codeberg repository must provide a `CARGO_REGISTRY_TOKEN` Actions secret.
