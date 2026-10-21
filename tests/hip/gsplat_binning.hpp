/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "Intersect.h"
#include "Projection.h"
#include "hip_check.hpp"
#include <hipcub/hipcub.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

// Test-owned buffers, one fixed small-scene capacity, no application allocator.
// Every frame recomputes projection, tile counts, scan, emitted intersections,
// depth sorting and offsets on the same HIP stream. One extra sentinel guarantees
// that even a completely full fixture has a defined terminal tile offset.
class HipGsplatBinning {
    struct Storage {
        void* pointer = nullptr;
        explicit Storage(size_t bytes) { hip_check(hipMalloc(&pointer, bytes), "binning allocation"); }
        ~Storage() { (void)hipFree(pointer); }
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        template <class T>
        T* as() const { return static_cast<T*>(pointer); }
    };

    struct Widen {
        __host__ __device__ int64_t operator()(int32_t value) const { return value; }
    };

    unsigned count_, width_, height_, tile_width_, tile_height_, capacity_;
    Storage radii_, means2d_, depths_, conics_, counts_, prefix_, keys_, sorted_keys_, ids_, sorted_ids_, offsets_;
    size_t scan_bytes_ = 0, sort_bytes_ = 0;
    void* scratch_ = nullptr;
    using Iterator = hipcub::TransformInputIterator<int64_t, Widen, const int32_t*>;

public:
    HipGsplatBinning(unsigned count, unsigned width, unsigned height, hipStream_t stream)
        : count_(count), width_(width), height_(height), tile_width_((width + 15) / 16), tile_height_((height + 15) / 16), capacity_(count * tile_width_ * tile_height_ + 1), radii_(count * 2 * sizeof(int32_t)), means2d_(count * 2 * sizeof(float)), depths_(count * sizeof(float)), conics_(count * 3 * sizeof(float)), counts_(count * sizeof(int32_t)), prefix_(count * sizeof(int64_t)), keys_(capacity_ * sizeof(int64_t)), sorted_keys_(capacity_ * sizeof(int64_t)), ids_(capacity_ * sizeof(int32_t)), sorted_ids_(capacity_ * sizeof(int32_t)), offsets_((tile_width_ * tile_height_ + 1) * sizeof(int32_t)) {
        hip_check(hipcub::DeviceScan::InclusiveSum(nullptr, scan_bytes_,
                                                   Iterator(counts_.as<int32_t>(), Widen{}), prefix_.as<int64_t>(), count_, stream),
                  "scan query");
        hip_check(hipcub::DeviceRadixSort::SortPairs(nullptr, sort_bytes_, keys_.as<int64_t>(), sorted_keys_.as<int64_t>(),
                                                     ids_.as<int32_t>(), sorted_ids_.as<int32_t>(), capacity_, 0, 64, stream),
                  "sort query");
        hip_check(hipMalloc(&scratch_, std::max(scan_bytes_, sort_bytes_)), "binning scratch allocation");
    }
    ~HipGsplatBinning() { (void)hipFree(scratch_); }
    HipGsplatBinning(const HipGsplatBinning&) = delete;
    HipGsplatBinning& operator=(const HipGsplatBinning&) = delete;

    const int32_t* offsets() const { return offsets_.as<int32_t>(); }
    const int32_t* ids() const { return sorted_ids_.as<int32_t>(); }
    unsigned capacity() const { return capacity_; }

    void run(const float* means, const float* quats, const float* scales,
             const float* view, const float* intrinsics, hipStream_t stream) {
        // Omit opacity-aware shrinking in this first correctness experiment;
        // retain the conservative geometric extent from the existing kernel.
        gsplat_lfs::launch_projection_ut_3dgs_fused_kernel(means, quats, scales, nullptr,
                                                           view, nullptr, intrinsics, count_, 1, width_, height_, 0.3f, 0.01f, 1000, 0,
                                                           PINHOLE, {}, ShutterType::GLOBAL, nullptr, nullptr, nullptr,
                                                           radii_.as<int32_t>(), means2d_.as<float>(), depths_.as<float>(), conics_.as<float>(), nullptr, stream);
        gsplat_lfs::launch_intersect_tile_kernel(means2d_.as<float>(), radii_.as<int32_t>(), depths_.as<float>(),
                                                 nullptr, nullptr, 1, count_, 0, false, 16, tile_width_, tile_height_, nullptr,
                                                 counts_.as<int32_t>(), nullptr, nullptr, stream);
        auto bytes = scan_bytes_;
        hip_check(hipcub::DeviceScan::InclusiveSum(scratch_, bytes,
                                                   Iterator(counts_.as<int32_t>(), Widen{}), prefix_.as<int64_t>(), count_, stream),
                  "tile count scan");
        gsplat_lfs::launch_fill_isect_sentinels_kernel(keys_.as<int64_t>(), ids_.as<int32_t>(), capacity_,
                                                       std::numeric_limits<int64_t>::max(), stream);
        gsplat_lfs::launch_intersect_tile_kernel(means2d_.as<float>(), radii_.as<int32_t>(), depths_.as<float>(),
                                                 nullptr, nullptr, 1, count_, 0, false, 16, tile_width_, tile_height_, prefix_.as<int64_t>(),
                                                 nullptr, keys_.as<int64_t>(), ids_.as<int32_t>(), stream, capacity_ - 1);
        bytes = sort_bytes_;
        hip_check(hipcub::DeviceRadixSort::SortPairs(scratch_, bytes, keys_.as<int64_t>(), sorted_keys_.as<int64_t>(),
                                                     ids_.as<int32_t>(), sorted_ids_.as<int32_t>(), capacity_, 0, 64, stream),
                  "tile depth sort");
        gsplat_lfs::launch_intersect_offset_kernel(sorted_keys_.as<int64_t>(), capacity_, 1, tile_width_, tile_height_, offsets_.as<int32_t>(), stream);
    }

    void verify(hipStream_t stream) const {
        hip_check(hipStreamSynchronize(stream), "binning validation completion");
        std::vector<int64_t> keys(capacity_);
        std::vector<int64_t> prefix(count_);
        std::vector<int32_t> offsets(tile_width_ * tile_height_ + 1), ids(capacity_);
        std::vector<int32_t> radii(count_ * 2);
        std::vector<float> means2d(count_ * 2), depths(count_);
        int64_t total = 0;
        hip_check(hipMemcpy(keys.data(), sorted_keys_.pointer, keys.size() * sizeof(int64_t), hipMemcpyDeviceToHost), "read sorted keys");
        hip_check(hipMemcpy(ids.data(), sorted_ids_.pointer, ids.size() * sizeof(int32_t), hipMemcpyDeviceToHost), "read sorted IDs");
        hip_check(hipMemcpy(offsets.data(), offsets_.pointer, offsets.size() * sizeof(int32_t), hipMemcpyDeviceToHost), "read offsets");
        hip_check(hipMemcpy(&total, prefix_.as<int64_t>() + count_ - 1, sizeof(total), hipMemcpyDeviceToHost), "read intersection count");
        hip_check(hipMemcpy(prefix.data(), prefix_.pointer, prefix.size() * sizeof(int64_t), hipMemcpyDeviceToHost), "read prefix");
        hip_check(hipMemcpy(radii.data(), radii_.pointer, radii.size() * sizeof(int32_t), hipMemcpyDeviceToHost), "read radii");
        hip_check(hipMemcpy(means2d.data(), means2d_.pointer, means2d.size() * sizeof(float), hipMemcpyDeviceToHost), "read projected means");
        hip_check(hipMemcpy(depths.data(), depths_.pointer, depths.size() * sizeof(float), hipMemcpyDeviceToHost), "read depths");
        if (total < 0 || total >= capacity_ || !std::is_sorted(keys.begin(), keys.end()) || offsets.back() != total)
            throw std::runtime_error("Binning total, sort or terminal offset mismatch");
        for (size_t tile = 0; tile + 1 < offsets.size(); ++tile) {
            if (offsets[tile] < 0 || offsets[tile] > offsets[tile + 1])
                throw std::runtime_error("Nonmonotonic tile offsets");
            for (int32_t i = offsets[tile]; i < offsets[tile + 1]; ++i)
                if ((keys[i] >> 32) != static_cast<int64_t>(tile) || ids[i] < 0 || ids[i] >= static_cast<int32_t>(count_))
                    throw std::runtime_error("Sorted tile contains an invalid Gaussian");
        }
        for (size_t i = static_cast<size_t>(total); i < keys.size(); ++i)
            if (keys[i] != std::numeric_limits<int64_t>::max() || ids[i] != -1)
                throw std::runtime_error("Intersection sentinel was overwritten");
        std::vector<std::pair<int64_t, int32_t>> expected, actual;
        for (unsigned g = 0; g < count_; ++g) {
            if (radii[g * 2] > 0 && radii[g * 2 + 1] > 0) {
                const float x = means2d[g * 2] / 16, y = means2d[g * 2 + 1] / 16;
                const float rx = float(radii[g * 2]) / 16, ry = float(radii[g * 2 + 1]) / 16;
                const int left = int(std::clamp(std::floor(x - rx), 0.0f, float(tile_width_)));
                const int top = int(std::clamp(std::floor(y - ry), 0.0f, float(tile_height_)));
                const int right = int(std::clamp(std::ceil(x + rx), 0.0f, float(tile_width_)));
                const int bottom = int(std::clamp(std::ceil(y + ry), 0.0f, float(tile_height_)));
                uint32_t depth_bits = 0;
                std::memcpy(&depth_bits, &depths[g], sizeof(depth_bits));
                for (int row = top; row < bottom; ++row)
                    for (int col = left; col < right; ++col)
                        expected.emplace_back((int64_t(row * tile_width_ + col) << 32) | depth_bits, g);
            }
            if (prefix[g] != static_cast<int64_t>(expected.size()))
                throw std::runtime_error("HIP tile count/scan differs from CPU rectangle enumeration");
        }
        for (size_t i = 0; i < static_cast<size_t>(total); ++i)
            actual.emplace_back(keys[i], ids[i]);
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        if (expected != actual)
            throw std::runtime_error("HIP intersections differ from CPU rectangle enumeration");
        std::printf("PASS dynamic binning: %lld real intersections, %u capacity slots\n", static_cast<long long>(total), capacity_);
    }
};
