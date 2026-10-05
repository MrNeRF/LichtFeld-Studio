/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "splat_tile_binner.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
    using namespace lfs::core;
    using lfs::rendering::SplatTileBinner;

    struct Splat {
        std::array<float, 4> mean_depth, conic_opacity, color;
        std::array<uint32_t, 4> bounds;
    };
    struct Raster {
        uint32_t count, width, height, columns, tiles, capacity, mode, unused;
        std::array<float, 4> background, render_origin, intrinsics, clip;
        std::array<uint32_t, 4> camera;
        std::array<float, 4> panorama;
        std::array<uint32_t, 4> mask_limits;
    };
    struct Status {
        uint64_t required;
        uint32_t error, blend_threads, maximum_tile_instances, padding;
    };
    static_assert(sizeof(Splat) == SplatTileBinner::kProjectedSplatBytes);
    static_assert(sizeof(Raster) == SplatTileBinner::kRasterParametersBytes);
    static_assert(sizeof(Status) == SplatTileBinner::kRasterStatusBytes);

    template <class T>
    Tensor upload(const std::vector<T>& values) {
        const size_t bytes = values.size() * sizeof(T);
        return Tensor::from_blob(const_cast<T*>(values.data()), {bytes}, Device::CPU, DataType::UInt8).to(Device::GPU);
    }
    template <class T>
    std::vector<T> download(const Tensor& tensor, size_t count) {
        auto host = tensor.to(Device::CPU);
        std::vector<T> result(count);
        std::memcpy(result.data(), host.data_ptr(), count * sizeof(T));
        return result;
    }

    // Random splats in a width x height viewport, a few invalid or off-screen.
    std::vector<Splat> make_splats(size_t count, uint32_t width, uint32_t height, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<uint32_t> x(0, width + 40), y(0, height + 40), extent(1, 70);
        std::uniform_real_distribution<float> depth(0.1f, 50.0f);
        std::vector<Splat> splats(count);
        for (size_t i = 0; i < count; ++i) {
            auto& s = splats[i];
            const float d = depth(rng);
            s.mean_depth = {1, 1, d, 0};
            s.conic_opacity = {1, 0, 1, 0.5f};
            // Ties exercise stability: every 7th splat shares a depth.
            s.color = {0.5f, 0.5f, 0.5f, i % 7 == 0 ? 3.25f : d};
            const uint32_t x0 = x(rng), y0 = y(rng);
            s.bounds = {x0, y0, x0 + extent(rng), y0 + extent(rng)};
            if (i % 97 == 5)
                s.mean_depth[2] = -1; // behind the camera: culled
        }
        return splats;
    }

    struct Reference {
        std::vector<uint64_t> keys;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> ranges;
    };
    Reference bin_on_cpu(const std::vector<Splat>& splats, const Raster& r) {
        std::vector<std::pair<uint64_t, uint32_t>> instances;
        for (uint32_t i = 0; i < splats.size(); ++i) {
            const auto& s = splats[i];
            if (!(s.mean_depth[2] > 0))
                continue;
            const uint32_t x0 = std::min(s.bounds[0], r.width), y0 = std::min(s.bounds[1], r.height);
            const uint32_t x1 = std::min(s.bounds[2], r.width), y1 = std::min(s.bounds[3], r.height);
            if (x1 <= x0 || y1 <= y0)
                continue;
            for (uint32_t ty = y0 / 16; ty < (y1 + 15) / 16; ++ty)
                for (uint32_t tx = x0 / 16; tx < (x1 + 15) / 16; ++tx)
                    instances.push_back({(uint64_t(ty * r.columns + tx) << 32) | std::bit_cast<uint32_t>(s.color[3]), i});
        }
        std::stable_sort(instances.begin(), instances.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        Reference result;
        result.ranges.assign(size_t(r.tiles) * 2, 0);
        for (size_t i = 0; i < instances.size(); ++i) {
            result.keys.push_back(instances[i].first);
            result.indices.push_back(instances[i].second);
            const uint32_t tile = uint32_t(instances[i].first >> 32);
            if (i == 0 || tile != uint32_t(instances[i - 1].first >> 32))
                result.ranges[tile * 2] = uint32_t(i);
            result.ranges[tile * 2 + 1] = uint32_t(i + 1);
        }
        return result;
    }

    Raster make_raster(uint32_t count, uint32_t width, uint32_t height, uint32_t capacity) {
        Raster r{};
        r.count = count;
        r.width = width;
        r.height = height;
        r.columns = (width + 15) / 16;
        r.tiles = r.columns * ((height + 15) / 16);
        r.capacity = capacity;
        r.mask_limits = {0, 0, 0, 0};
        return r;
    }

    class TileBinning : public testing::TestWithParam<GpuBackend> {};

    TEST_P(TileBinning, MatchesStableCpuSortAndRanges) {
        if (!gpu_backend_available(GetParam()) || GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        SplatTileBinner binner(GetParam());
        // Several sizes: one partial block, many sort blocks, and an empty frame.
        for (const auto& [count, width, height] : {std::tuple{300u, 200u, 120u}, std::tuple{20000u, 1280u, 720u}, std::tuple{0u, 64u, 64u}}) {
            const auto splats = make_splats(count, width, height, count + width);
            auto raster = make_raster(count, width, height, 4'000'000);
            const auto expected = bin_on_cpu(splats, raster);
            ASSERT_TRUE(binner.reserve(std::max(count, 1u), raster.tiles, raster.capacity));
            const auto splat_tensor = upload(splats.empty() ? std::vector<Splat>(1) : splats);
            const auto raster_tensor = upload(std::vector<Raster>{raster});
            auto binned = binner.bin(splat_tensor, raster_tensor, count, raster.tiles);
            ASSERT_TRUE(binned) << binned.error().detail();
            const auto status = download<Status>(binner.status(), 1)[0];
            ASSERT_EQ(status.error, 0u);
            ASSERT_EQ(status.required, expected.keys.size()) << count;
            if (!expected.keys.empty()) {
                EXPECT_EQ(download<uint64_t>(binner.keys(), expected.keys.size()), expected.keys) << count;
                EXPECT_EQ(download<uint32_t>(binner.indices(), expected.indices.size()), expected.indices) << count;
            }
            EXPECT_EQ(download<uint32_t>(binner.ranges(), expected.ranges.size()), expected.ranges) << count;
        }
    }

    TEST_P(TileBinning, OverflowReportsRequiredAndBinsNothing) {
        if (!gpu_backend_available(GetParam()) || GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        SplatTileBinner binner(GetParam());
        const auto splats = make_splats(5000, 640, 480, 11);
        auto raster = make_raster(5000, 640, 480, 100);
        const auto expected = bin_on_cpu(splats, raster);
        ASSERT_GT(expected.keys.size(), raster.capacity);
        ASSERT_TRUE(binner.reserve(5000, raster.tiles, raster.capacity));
        auto binned = binner.bin(upload(splats), upload(std::vector<Raster>{raster}), 5000, raster.tiles);
        ASSERT_TRUE(binned) << binned.error().detail();
        const auto status = download<Status>(binner.status(), 1)[0];
        EXPECT_EQ(status.error, 1u);
        EXPECT_EQ(status.required, expected.keys.size());
        EXPECT_EQ(download<uint32_t>(binner.ranges(), size_t(raster.tiles) * 2), std::vector<uint32_t>(size_t(raster.tiles) * 2, 0u));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, TileBinning, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
