/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <string_view>
// Private HIP preparation: sort V visible primitives rather than I instances
// by all depth bits, then a stable tile-only instance sort completes the same
// lexicographic order. Parameter storage and compact work indices stay intact.
struct LfsHipDepthOrder {
    const unsigned* indices = nullptr;
    const std::uint64_t* tile_counts = nullptr;
};
__global__ void lfs_hip_depth_sort_initialize(const unsigned* source, unsigned* keys,
                                             unsigned* indices, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i] = source[i];
        indices[i] = i;
    }
}
__global__ void lfs_hip_depth_sort_counts(const unsigned* order, const std::uint64_t* source,
                                        std::uint64_t* output, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = source[order[i]];
}
inline LfsHipDepthOrder lfs_hip_prepare_depth_order(const std::function<char*(size_t)>& allocate,
    const unsigned* depths, const std::uint64_t* counts, unsigned n, hipStream_t stream,
    bool allowed, bool* attempted) {
    static const bool enabled = [] {
        const char* value = std::getenv("LFS_HIP_FAST_DEPTH_PRESORT");
        const bool requested = !value || std::string_view(value) != "0";
        if (requested)
            std::fprintf(stderr, "[info] HIP stable visible-depth presort enabled\n");
        return requested;
    }();
    if (!enabled || !allowed || n < 32768)
        return {nullptr, counts};
    if (attempted)
        *attempted = true;
    hipcub::DoubleBuffer<unsigned> keys(nullptr, nullptr), indices(nullptr, nullptr);
    size_t scratch_bytes = 0;
    lfs_fast_hip_check(lfs_fast_full_sort_pairs(nullptr, scratch_bytes, keys, indices, n, 0, 32, stream),
                       "HIP visible-depth workspace query");
    constexpr size_t bytes_per_primitive = 4 * sizeof(unsigned) + sizeof(std::uint64_t);
    if (n > (std::numeric_limits<size_t>::max() - 255) / bytes_per_primitive)
        throw std::overflow_error("HIP depth sort size overflow");
    const size_t scratch_offset = (static_cast<size_t>(n) * bytes_per_primitive + 255) & ~size_t(255);
    if (scratch_bytes > std::numeric_limits<size_t>::max() - scratch_offset)
        throw std::overflow_error("HIP depth sort workspace overflow");
    char* base = allocate(scratch_offset + scratch_bytes);
    if (!base)
        throw std::runtime_error("OUT_OF_MEMORY: HIP visible-depth sort storage");
    auto* k0 = reinterpret_cast<unsigned*>(base);
    auto* k1 = k0 + n;
    auto* v0 = k1 + n;
    auto* v1 = v0 + n;
    auto* ordered_counts = reinterpret_cast<std::uint64_t*>(v1 + n);
    keys = hipcub::DoubleBuffer<unsigned>(k0, k1);
    indices = hipcub::DoubleBuffer<unsigned>(v0, v1);
    lfs_hip_depth_sort_initialize<<<(n + 255u) / 256u, 256, 0, stream>>>(depths, k0, v0, n);
    lfs_fast_hip_check(hipGetLastError(), "HIP visible-depth initialize");
    lfs_fast_hip_check(lfs_fast_full_sort_pairs(base + scratch_offset, scratch_bytes, keys, indices, n, 0, 32, stream),
                       "HIP visible-depth sort");
    lfs_hip_depth_sort_counts<<<(n + 255u) / 256u, 256, 0, stream>>>(indices.Current(), counts, ordered_counts, n);
    lfs_fast_hip_check(hipGetLastError(), "HIP visible-depth counts");
    return {indices.Current(), ordered_counts};
}
