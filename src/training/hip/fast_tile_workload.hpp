/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

// Diagnostic only: a completed forward supplies a conservative upper bound
// for each tile's reverse walk. This is not a GPU duration measurement.
// An explicit path enables the readbacks; normal training does no extra work.
inline void lfs_hip_report_tile_workload(const uint2* ranges, const unsigned* contributions,
                                        unsigned width, unsigned height, unsigned grid_width,
                                        unsigned tiles, hipStream_t stream) {
    static const std::string destination = [] {
        const char* value = std::getenv("LFS_HIP_FAST_TILE_WORKLOAD_DUMP");
        return value ? std::string(value) : std::string{};
    }();
    if (destination.empty())
        return;
    static std::atomic<unsigned> sequence{0};
    const unsigned frame = sequence.fetch_add(1, std::memory_order_relaxed);
    if (frame >= 80)
        return;
    std::vector<uint2> host_ranges(tiles);
    std::vector<unsigned> host_contributions(static_cast<size_t>(width) * height);
    std::vector<unsigned> costs(tiles);
    lfs_fast_hip_check(hipStreamSynchronize(stream), "HIP workload diagnostic completion");
    lfs_fast_hip_check(hipMemcpy(host_ranges.data(), ranges, host_ranges.size() * sizeof(uint2),
                               hipMemcpyDeviceToHost), "HIP workload range readback");
    lfs_fast_hip_check(hipMemcpy(host_contributions.data(), contributions,
                               host_contributions.size() * sizeof(unsigned), hipMemcpyDeviceToHost),
                       "HIP workload contribution readback");
    for (unsigned tile = 0; tile < tiles; ++tile) {
        const unsigned x0 = tile % grid_width * 16u;
        const unsigned y0 = tile / grid_width * 16u;
        unsigned maximum = 0;
        for (unsigned y = y0; y < std::min(y0 + 16u, height); ++y)
            for (unsigned x = x0; x < std::min(x0 + 16u, width); ++x)
                maximum = std::max(maximum, host_contributions[static_cast<size_t>(y) * width + x]);
        costs[tile] = std::min(maximum, host_ranges[tile].y - host_ranges[tile].x);
    }
    static std::mutex output_mutex;
    const std::lock_guard lock(output_mutex);
    FILE* file = std::fopen(destination.c_str(), "ab");
    if (!file) {
        std::fprintf(stderr, "[warning] HIP workload diagnostic could not open its output file\n");
        return;
    }
    std::fprintf(file, "{\"frame\":%u,\"width\":%u,\"height\":%u,\"costs\":[", frame, width, height);
    for (unsigned tile = 0; tile < tiles; ++tile)
        std::fprintf(file, "%s%u", tile ? "," : "", costs[tile]);
    std::fputs("]}\n", file);
    std::fclose(file);
}
