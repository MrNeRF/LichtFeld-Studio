/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance_sort.cuh"
#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace {
    namespace raster = fast_lfs::rasterization;

    template <typename Key>
    void check_tiles(const unsigned tile_count) {
        SCOPED_TRACE(::testing::Message() << "tiles=" << tile_count << " key bytes=" << sizeof(Key));
        const std::array<unsigned, 6> tiles{tile_count - 1, 0, tile_count - 1, tile_count > 1 ? 1u : 0u, 0, tile_count - 1};
        std::array<Key, 6> keys{};
        std::array<unsigned, 6> ids{0x01000003u, 0x01000001u, 0x01000002u, 0xfffffffeu, 0x00ffffffu, 5u};
        for (size_t i = 0; i < keys.size(); ++i)
            keys[i] = static_cast<Key>(tiles[i]);
        Key *keys_in = nullptr, *keys_out = nullptr;
        unsigned *ids_in = nullptr, *ids_out = nullptr;
        uint2* ranges = nullptr;
        raster::FastGSForwardStatus* status = nullptr;
        void* workspace = nullptr;
        ASSERT_EQ(cudaMalloc(&keys_in, sizeof(keys)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&keys_out, sizeof(keys)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&ids_in, sizeof(ids)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&ids_out, sizeof(ids)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&ranges, tile_count * sizeof(uint2)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&status, sizeof(*status)), cudaSuccess);
        ASSERT_EQ(cudaMemset(ranges, 0, tile_count * sizeof(uint2)), cudaSuccess);
        ASSERT_EQ(cudaMemset(status, 0, sizeof(*status)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(keys_in, keys.data(), sizeof(keys), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(ids_in, ids.data(), sizeof(ids), cudaMemcpyHostToDevice), cudaSuccess);
        size_t bytes = 0;
        ASSERT_EQ(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_in, keys_out, ids_in, ids_out, 6), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&workspace, bytes), cudaSuccess);
        ASSERT_EQ(cub::DeviceRadixSort::SortPairs(workspace, bytes, keys_in, keys_out, ids_in, ids_out, 6), cudaSuccess);
        raster::kernels::forward::extract_instance_ranges_cu<Key><<<1, 32>>>(keys_out, ranges, status, tile_count, 6, nullptr);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        std::array<unsigned, 6> actual{};
        std::vector<uint2> actual_ranges(tile_count);
        raster::FastGSForwardStatus actual_status{};
        ASSERT_EQ(cudaMemcpy(actual.data(), ids_out, sizeof(actual), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(actual_ranges.data(), ranges, actual_ranges.size() * sizeof(uint2), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&actual_status, status, sizeof(actual_status), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(actual_status.flags, 0u);
        std::array<size_t, 6> order{};
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return tiles[a] < tiles[b]; });
        std::vector<uint2> expected_ranges(tile_count, uint2{0, 0});
        for (unsigned i = 0; i < order.size(); ++i) {
            EXPECT_EQ(actual[i], ids[order[i]]);
            const auto tile = tiles[order[i]];
            if (i == 0 || tile != tiles[order[i - 1]])
                expected_ranges[tile].x = i;
            expected_ranges[tile].y = i + 1;
        }
        for (size_t i = 0; i < actual_ranges.size(); ++i) {
            EXPECT_EQ(actual_ranges[i].x, expected_ranges[i].x);
            EXPECT_EQ(actual_ranges[i].y, expected_ranges[i].y);
        }
        EXPECT_EQ(cudaFree(keys_in), cudaSuccess);
        EXPECT_EQ(cudaFree(keys_out), cudaSuccess);
        EXPECT_EQ(cudaFree(ids_in), cudaSuccess);
        EXPECT_EQ(cudaFree(ids_out), cudaSuccess);
        EXPECT_EQ(cudaFree(ranges), cudaSuccess);
        EXPECT_EQ(cudaFree(status), cudaSuccess);
        EXPECT_EQ(cudaFree(workspace), cudaSuccess);
    }

    // Raw CUDA launches: skip without a CUDA device, like the other CUDA-only tests.
    class RasterTileKeys : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "CUDA device unavailable";
        }
    };

    TEST_F(RasterTileKeys, NarrowTileKeysRetainFullIdsAndStableTies) {
        check_tiles<std::uint16_t>(1);
        check_tiles<std::uint16_t>(65535);
        check_tiles<std::uint16_t>(65536);
        check_tiles<unsigned>(65537);
    }
} // namespace
