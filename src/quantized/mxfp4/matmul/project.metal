// SIMD fragment layout and accumulation order adapted from MLX 0.32.2
// steel/gemm/mma.h, loader.h and fp4.h. Copyright © 2024-2026 Apple Inc.
// See LICENSE-MLX.txt.
// BM=16/32, BN=32, BK=32. Ordinary matrix, FP32 split-K storage.
const uint first = threadgroup_position_in_grid.y * BM;
const uint end = min(uint(M), first + BM);
const uint partition = threadgroup_position_in_grid.z;
const uint col = threadgroup_position_in_grid.x * 32;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = tid % 32;
const uint warp = tid / 32;
const uint warp_m = warp / 2;
const uint warp_n = warp % 2;
const uint fm = ((lane / 4) & 4) + (lane / 2) % 4;
const uint fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
threadgroup T xs[BM * 40];
threadgroup T ws[32 * 40];
simdgroup_matrix<float, 8, 8> accum[4];
for (uint i = 0; i < 4; ++i) {
    accum[i].thread_elements()[0] = 0;
    accum[i].thread_elements()[1] = 0;
}
struct InputVector { ushort values[8]; };
for (uint k = partition * PART; k < (partition + 1) * PART; k += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint xr = tid / 4;
    const uint xc = (tid % 4) * 8;
    if (first + xr < end) {
        *reinterpret_cast<threadgroup InputVector*>(xs + xr * 40 + xc) =
            *reinterpret_cast<const device InputVector*>(input + (size_t(first) + xr) * K + k + xc);
    } else {
        #pragma unroll
        for (uint j = 0; j < 8; ++j) xs[xr * 40 + xc + j] = T(0);
    }
    // Four SIMD groups distribute the same weight tile over twice as many threads.
    constexpr uint values_per_thread = 256 / BM;
    constexpr uint threads_per_row = 32 / values_per_thread;
    const uint wr = tid / threads_per_row;
    const uint wc = (tid % threads_per_row) * values_per_thread;
    const size_t row = col + wr;
    if (row < N) {
    const uint exponent = scales[row * (K / 32) + k / 32];
    const float scale = float(T(as_type<float>(exponent == 255 ? 0x7fc00000u : (exponent == 0 ? 0x00400000u
        : exponent << 23))));
    #pragma unroll
    for (uint p = 0; p < values_per_thread / 8; ++p) {
        const uint packed = weight[row * (K / 8) + (k + wc) / 8 + p];
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            const uint code = (packed >> (4 * j)) & 15;
            half decoded = as_type<half>(ushort((code & 7) << 9));
            decoded *= half(16384.0f);
            const float value = float((code & 8) ? -decoded : decoded);
            ws[wr * 40 + wc + p * 8 + j] = T(scale * value);
        }
    }
    } else {
        for (uint j = 0; j < values_per_thread; ++j) ws[wr * 40 + wc + j] = T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint kk = 0; kk < 32; kk += 8) {
        simdgroup_matrix<float, 8, 8> a[2], b[2];
        #pragma unroll
        for (uint i = 0; i < 2; ++i) {
            a[i].thread_elements()[0] = float(xs[(warp_m * 16 + fm + i * 8) * 40 + kk + fn]);
            a[i].thread_elements()[1] = float(xs[(warp_m * 16 + fm + i * 8) * 40 + kk + fn + 1]);
            b[i].thread_elements()[0] = float(ws[(warp_n * 8 + fn + i * 16) * 40 + kk + fm]);
            b[i].thread_elements()[1] = float(ws[(warp_n * 8 + fn + i * 16 + 1) * 40 + kk + fm]);
        }
        simdgroup_multiply_accumulate(accum[0], a[0], b[0], accum[0]);
        simdgroup_multiply_accumulate(accum[1], a[0], b[1], accum[1]);
        simdgroup_multiply_accumulate(accum[3], a[1], b[1], accum[3]);
        simdgroup_multiply_accumulate(accum[2], a[1], b[0], accum[2]);
    }
}
#pragma unroll
for (uint m = 0; m < 2; ++m) {
    const uint row = first + warp_m * 16 + fm + m * 8;
    if (row < end) {
        #pragma unroll
        for (uint n = 0; n < 2; ++n) {
            const uint column = col + warp_n * 8 + fn + n * 16;
            const size_t base = (size_t(partition) * M + row) * N;
            if (column < N) output[base + column] = O(accum[m * 2 + n].thread_elements()[0]);
            if (column + 1 < N) output[base + column + 1] = O(accum[m * 2 + n].thread_elements()[1]);
        }
    }
}
