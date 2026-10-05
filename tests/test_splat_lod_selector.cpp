/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "splat_lod_selector.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
    using namespace lfs::core;
    using lfs::rendering::SplatLodParameters;
    using lfs::rendering::SplatLodSelector;
    using lfs::rendering::SplatLodTree;

    template <class T, size_t N>
    Tensor upload(const std::array<T, N>& values) {
        return Tensor::from_blob(const_cast<T*>(values.data()), {sizeof(values)}, Device::CPU, DataType::UInt8).to(Device::GPU);
    }
    template <class T>
    std::vector<T> download(const Tensor& tensor, const size_t count) {
        const auto host = tensor.to(Device::CPU);
        std::vector<T> result(count);
        std::memcpy(result.data(), host.data_ptr(), count * sizeof(T));
        return result;
    }

    class LodSelection : public testing::TestWithParam<GpuBackend> {};

    TEST_P(LodSelection, MatchesNativeThresholdFadeBudgetAndResidencyContracts) {
        if (!gpu_backend_available(GetParam()) || GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        const std::array<uint32_t, 21> links = {1, 2, 0xffffffffu, 3, 2, 0, 5, 2, 0,
                                                0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0, 2};
        for (int scenario = 0; scenario < 10; ++scenario) {
            SCOPED_TRACE(scenario);
            const std::array<float, 7> sizes = scenario == 3   ? std::array<float, 7>{.05f, 1, 1, .02f, .02f, .02f, .02f}
                                               : scenario == 4 ? std::array<float, 7>{.525f, .1f, .1f, .02f, .02f, .02f, .02f}
                                                               : std::array<float, 7>{1, .1f, .1f, .02f, .02f, .02f, .02f};
            std::array<uint32_t, 14> bounds{};
            std::array<std::array<float, 4>, 8> frames{};
            for (size_t page = 0; page < 2; ++page) {
                const size_t begin = page * 4, end = std::min(begin + 4, size_t(7));
                float lo = std::numeric_limits<float>::infinity();
                float hi = -std::numeric_limits<float>::infinity();
                for (size_t n = begin; n < end; ++n) {
                    lo = std::min(lo, std::log(sizes[n]));
                    hi = std::max(hi, std::log(sizes[n]));
                }
                frames[page * 4 + 1] = {0, 0, -5, lo};
                frames[page * 4 + 2] = {0, 0, 0, hi - lo};
                for (size_t n = begin; n < end; ++n) {
                    const uint32_t quant = hi > lo ? uint32_t(std::lround(std::clamp((std::log(sizes[n]) - lo) / (hi - lo), 0.f, 1.f) * 65535.f)) : 0;
                    bounds[n * 2 + 1] = quant << 16;
                }
            }
            std::array<uint32_t, 2> chunks = {0, 1}, pages = {0, 1}, age{};
            if (scenario == 6)
                chunks[1] = pages[1] = 0xffffffffu;
            if (scenario == 7)
                age[1] = 10;
            if (scenario == 9)
                frames[1][2] = frames[5][2] = 5;

            const auto bounds_tensor = upload(bounds);
            const auto links_tensor = upload(links);
            const auto chunks_tensor = upload(chunks);
            const auto age_tensor = upload(age);
            const auto frames_tensor = upload(frames);
            const auto pages_tensor = upload(pages);
            const SplatLodTree tree{&bounds_tensor, &links_tensor, &chunks_tensor,
                                    &age_tensor, &frames_tensor, &pages_tensor};
            SplatLodParameters p;
            p.node_count = 7;
            p.physical_node_count = scenario == 8 ? 5 : 7;
            p.logical_chunk_count = 2;
            p.chunk_splats = 4;
            p.output_capacity = scenario == 5 ? 1 : 7;
            p.pixel_scale_limit = scenario == 0 ? .21f : scenario == 2 || scenario == 5 || scenario == 6 || scenario == 7 || scenario == 8 ? .01f
                                                     : scenario == 3                                                                       ? .02f
                                                                                                                                           : .1f;
            p.viewport_foveation = 0;
            p.behind_camera_penalty = 1;
            p.cone_foveation = 1;
            if (scenario == 7) {
                p.fade_frames = 10;
                p.current_frame = 15;
            }
            if (scenario == 9)
                p.behind_camera_penalty = .2f;
            p.view_row0 = {1, 0, 0, 0};
            p.view_row1 = {0, 1, 0, 0};
            p.view_row2 = {0, 0, 1, 0};

            SplatLodSelector selector(GetParam());
            ASSERT_TRUE(selector.reserve(p.output_capacity, 7, 2));
            const auto selected = selector.select(tree, p);
            ASSERT_TRUE(selected) << selected.error().detail();
            const auto counts = download<uint32_t>(selector.counts(), 8);
            const uint32_t count = counts[0];
            ASSERT_LE(count, p.output_capacity);
            EXPECT_EQ(counts[1], 0u);
            const auto ids = download<uint32_t>(selector.indices(), count);
            const auto logical = download<uint32_t>(selector.logical_indices(), count);
            const auto weights = download<float>(selector.weights(), count);
            const auto levels = download<uint32_t>(selector.levels(), count);
            auto actual = ids;
            std::sort(actual.begin(), actual.end());
            const std::vector<uint32_t> expected = scenario == 7                                                      ? std::vector<uint32_t>{1, 2, 3, 4, 5, 6}
                                                   : scenario == 8                                                    ? std::vector<uint32_t>{2, 3, 4}
                                                   : scenario == 0 || scenario == 3 || scenario == 5 || scenario == 9 ? std::vector<uint32_t>{0}
                                                   : scenario == 2                                                    ? std::vector<uint32_t>{3, 4, 5, 6}
                                                   : scenario == 4                                                    ? std::vector<uint32_t>{0, 1, 2}
                                                                                                                      : std::vector<uint32_t>{1, 2};
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(logical, ids);
            EXPECT_EQ(levels, std::vector<uint32_t>(count, 0u));
            for (uint32_t n = 0; n < count; ++n) {
                EXPECT_TRUE(std::isfinite(weights[n]));
                EXPECT_GE(weights[n], 0);
                EXPECT_LE(weights[n], 1);
                if (scenario == 4) {
                    const double child = (1.05 - 1) / .18;
                    EXPECT_NEAR(weights[n], ids[n] == 0 ? 1 - child : child, .001);
                }
            }
            if (scenario == 7)
                for (uint32_t n = 0; n < count; ++n)
                    EXPECT_NEAR(weights[n], ids[n] == 3 ? 1.f : .5f, .001);
            if (scenario == 5)
                EXPECT_GT(std::bit_cast<float>(counts[2]), 1);
            if (scenario == 6)
                EXPECT_GT(download<uint32_t>(selector.touches(), 2)[1], 0u);

            if (scenario == 2) {
                auto invalid = p;
                invalid.pixel_scale_limit = std::numeric_limits<float>::quiet_NaN();
                EXPECT_FALSE(selector.select(tree, invalid));
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LodSelection, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
