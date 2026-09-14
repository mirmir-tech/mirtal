# Mirtal Agent Notes

This repository contains the standalone `mirtal` crate and its private support
crates. The notes below define its complete project-specific constraints.

## Purpose

Mirtal is a standalone native Rust API for Apple MLX. It must not depend on
Mirmir model, runtime, protocol, cache, tokenizer, or server crates.

## Hard Rules

- Rust source files must never exceed 250 lines; split large modules by focused
  responsibility.
- Split modules only as `module/mod.rs` plus focused files such as
  `module/feature.rs`. Never flatten a child into a sibling such as
  `module_feature.rs`. Tests follow the same rule: use `module/tests.rs` or
  `module/tests/feature.rs`, not `module_tests.rs` or `feature_tests.rs` for an
  existing module.
- Use `thiserror` conversions and `?`; reserve `map_err` for foreign boundaries
  that require additional context.
- Keep nightly rustfmt and clippy clean with all targets and features.
- Do not add Python or PyTorch to any runtime or build path.
- Dependencies on separately released projects must use versioned crates.io
  entries. Relative paths are limited to crates owned by this workspace.
- Do not commit or publish automatically without explicit authorization.
- Keep CXX bridge declarations and every `unsafe` operation inside `mirtal-sys`.
- `Array::allocation` observes only already available storage without evaluating
  or synchronizing. Its safe `memory::Allocation` handle retains buffer ownership,
  exposes allocator bytes, and deduplicates views through private identity.
  Never expose the native address or replace allocation bytes with logical
  tensor size. Retained allocation handles may prevent donation; keep them out
  of ordinary execution loops and use them for ownership/accounting snapshots.
- Native smart pointers may exist only in private implementation fields; public
  APIs use RAII and must not expose `cxx` types.
- Every operation receives an explicit stream, directly or through `Graph`.
- Host reads and stream synchronization must always be explicit.
- Do not add implicit default streams or thread-local execution state.
- Macros generate typed Rust bindings; they must not conceal host transfers.
- `metal_kernel!` must validate inline and file-backed MSL with Apple's Metal
  compiler during Rust compilation. Do not add unchecked convenience macros.
- Normalize the custom-kernel header boundary with a terminating newline before
  forwarding to MLX, matching macro validation. MLX concatenates generated code
  directly after headers; a trailing directive or line comment must not absorb
  the kernel declaration. Keep the runtime regression for both header forms.
- `metal_library!` applies the same rule to complete MSL translation units.
  `MetalLibrary::export` exposes checked named functions. Buffer mutation uses
  generic `AliasingDispatch`; never add a model- or cache-specific CXX API.
  Prefer static export names, fixed-arity `dispatch_aliasing_array`, and cached
  dispatch/output specifications in repeated execution paths. Use `rebind` for
  equal-arity dynamic constants and geometry instead of rebuilding a dispatch.
  Use typed `PreparedAliasing<INPUTS, OUTPUTS>` when a hot loop repeats one
  export. Its mutable dispatch contract is session-local and must not be shared
  concurrently; snapshots receive independent prepared plans.
- `PreparedMetalKernel<INPUTS, OUTPUTS>` may cache an invariant ordinary launch,
  but MLX still creates its lazy custom primitive per call. Do not replace a
  production `MetalKernel::dispatch` without alternating host-build and
  synchronized benchmarks; retain negative benchmark results in client docs.
- `interop` exists only for zero-copy migration of established MLX ABIs. Every
  ownership transfer must document allocation, lifetime, and single-adoption
  requirements; ordinary operations must never depend on native addresses.
- `Compiled` owns its native stream copy and callback but remains bound to the
  creating stream ID. Native compiled, kernel, and library handles are
  non-owning and must never outlive their Rust owners.
- Mirtal exposes generic MLX operations, not inference workflows. Sampling,
  expert routing/dispatch, embedding lookup, RoPE scaling policy, KV storage,
  and model assembly belong to clients such as Mirmir.
- `mirtal-sys` C++ may bind a native MLX primitive or launch mechanism. It must
  not compose multi-operation model algorithms; those are Rust graph functions
  in the owning application or checked Metal kernels.
- Quantize, dequantize, QMM, GatherQMM, fast RoPE, and fast SDPA are legitimate
  thin MLX primitive bindings. Vocabulary gathering and frequency construction
  are not.
- `GatherIndices` must preserve optional LHS/RHS semantics and native index
  broadcasting. Legacy RHS-only methods pass no LHS selector; never synthesize
  one for them. MLX 0.32.2 disables its sorted-RHS batching shortcut when an
  explicit LHS selector is present. Forward the native hint faithfully and do
  not assume that avoiding a caller-side gather preserves arithmetic or speed.
- Mirtal owns ordinary scaled dot-product attention execution. Clients choose
  tensor layout, scale, mask policy, KV storage, and paged-attention strategy.
  Tensor masks and attention sinks are borrowed device arrays; never copy them
  through the host or allocate substitutes in a decode step.
- `mirtal-macros` is re-exported by `mirtal` and is not a direct user dependency.
- Run rustfmt, Clippy with all targets/features, tests, and check before completion.
- Device-offset RoPE must preserve the leading batch axis: scalar-offset
  `single_token_layout` folding would assign offsets to the wrong rows.
  `RopeOptions` and `FrequencyRopeOptions` default to scalar `usize`; batched
  calls take borrowed integer arrays. Keep explicit frequencies device-resident
  and preserve MLX's scalar/one-element offset broadcasting without host reads.
