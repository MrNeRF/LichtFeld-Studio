/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_contract_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>

namespace {
    using namespace lfs::core;
    using namespace lfs::rendering;
    using namespace lfs::test::splat;

    class SplatRasterContracts : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (backend_unavailable_or_cuda(GetParam()))
                GTEST_SKIP();
            scope_ = std::make_unique<GpuBackendScope>(GetParam());
        }
        std::unique_ptr<GpuBackendScope> scope_;
    };

    void compare_cpu(const RasterReadback& actual, const std::vector<ProjectedSplat>& splats,
                     uint32_t width, uint32_t height, std::array<float, 4> background,
                     SplatRasterMode mode, bool expected_depth = false, float far = 100) {
        std::vector<uint32_t> sorted(splats.size());
        for (uint32_t i = 0; i < sorted.size(); ++i)
            sorted[i] = i;
        std::stable_sort(sorted.begin(), sorted.end(), [&](auto a, auto b) { return splats[a].color[3] < splats[b].color[3]; });
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                double rgb[3]{}, trans = 1, weighted = 0, valid_weight = 0, near = 0, median = 1e10;
                uint32_t picked = 0xffffffffu;
                for (auto id : sorted) {
                    const auto& s = splats[id];
                    if (s.bounds[2] <= s.bounds[0] || s.bounds[3] <= s.bounds[1] ||
                        x / 16 < s.bounds[0] / 16 || x / 16 >= (s.bounds[2] + 15) / 16 ||
                        y / 16 < s.bounds[1] / 16 || y / 16 >= (s.bounds[3] + 15) / 16)
                        continue;
                    const double dx = x - s.mean_depth[0], dy = y - s.mean_depth[1];
                    const double q = s.conic_opacity[0] * dx * dx + 2 * s.conic_opacity[1] * dx * dy + s.conic_opacity[2] * dy * dy;
                    double alpha = mode == SplatRasterMode::Points  ? (dx * dx + dy * dy <= s.mean_depth[3] * s.mean_depth[3] ? s.conic_opacity[3] : 0)
                                   : mode == SplatRasterMode::Discs ? (q <= 9 ? s.conic_opacity[3] : 0)
                                                                    : s.conic_opacity[3] * std::exp(-.5 * q);
                    alpha = std::min(alpha, double(.999f));
                    if (alpha < .5 / 255)
                        continue;
                    if (picked == 0xffffffffu) {
                        picked = id;
                        near = s.mean_depth[2];
                    }
                    for (int c = 0; c < 3; ++c)
                        rgb[c] += s.color[c] * alpha * trans;
                    if (!expected_depth || s.mean_depth[2] <= far) {
                        weighted += s.mean_depth[2] * alpha * trans;
                        valid_weight += alpha * trans;
                    }
                    const double next = trans * (1 - alpha);
                    if (trans > .5 && next <= .5)
                        median = s.mean_depth[2];
                    trans = next;
                    if (trans < 1e-4)
                        break;
                }
                const size_t at = size_t(y) * width + x;
                for (int c = 0; c < 3; ++c)
                    EXPECT_NEAR(half_to_float(actual.color[at * 4 + c]), rgb[c] + background[c] * background[3] * trans, .002) << x << ',' << y;
                EXPECT_NEAR(half_to_float(actual.color[at * 4 + 3]), 1 - trans + background[3] * trans, .001) << x << ',' << y;
                EXPECT_NEAR(actual.depth[at * 4], weighted, .001) << x << ',' << y;
                EXPECT_NEAR(actual.depth[at * 4 + 1], 1 - trans, 2e-5) << x << ',' << y;
                EXPECT_NEAR(actual.depth[at * 4 + 2], expected_depth ? valid_weight : near, 1e-5) << x << ',' << y;
                EXPECT_NEAR(actual.depth[at * 4 + 3], median, 1e-5) << x << ',' << y;
                EXPECT_EQ(actual.pick[at], picked) << x << ',' << y;
            }
    }

    SplatRasterParameters params(uint32_t count, uint32_t w, uint32_t h, SplatRasterMode mode, uint32_t capacity,
                                 std::array<float, 4> bg, uint32_t extra = 0) {
        uint32_t flags = extra;
        if (mode == SplatRasterMode::Gaussian && !(extra & 16))
            flags |= 128;
        if (bg[3] == 1)
            flags |= 4096;
        auto r = raster_parameters(count, w, h, mode, capacity, flags, bg);
        return r;
    }

    TEST_P(SplatRasterContracts, UnusedEmphasisLaneDoesNotChangeRenderedColor) {
        const ProjectedSplat splat{{16, 16, 3, 3}, {1, 0, 1, .8f}, {.2f, .3f, .4f, 1}, {0, 0, 32, 32}};
        auto input = upload_one(splat);
        std::array<std::array<float, 4>, 207> parameters{};
        parameters[24][3] = -1;
        std::array<std::array<float, 4>, 128> colors{};
        // Bit 2 and emphasis.w are reserved and must not affect selection color.
        const std::array<uint32_t, 1> flags{4};
        auto p = upload(parameters), f = upload(flags);
        SplatRasterizer rasterizer(GetParam());
        ASSERT_TRUE(rasterizer.reserve(1, 32, 32, 4));
        const SplatRasterOverlay overlay{&p, &f, nullptr, nullptr, std::as_bytes(std::span(colors))};
        const auto request = params(1, 32, 32, SplatRasterMode::Gaussian, 4, {}, 1);
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian, request, &overlay));
        const auto before = readback(rasterizer, 32, 32);
        parameters[20][3] = 1;
        p = upload(parameters);
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian, request, &overlay));
        const auto after = readback(rasterizer, 32, 32);
        EXPECT_EQ(before.color, after.color);
        EXPECT_EQ(before.depth, after.depth);
        EXPECT_EQ(before.pick, after.pick);
    }

    TEST_P(SplatRasterContracts, StableCompositingModesScanBoundariesAndReuseMatchCpuOracle) {
        constexpr uint32_t width = 37, height = 29, capacity = 131073;
        const std::array<float, 4> bg{.15f, .1f, .2f, .4f};
        SplatRasterizer rasterizer(GetParam());
        ASSERT_TRUE(rasterizer.reserve(capacity, width, height, capacity * 6));
        std::mt19937 random(0x1939);
        uint32_t sequence = 0;
        for (uint32_t count : {capacity, capacity, capacity, capacity, 2049u, 1u, 2u, 0u, 255u, 256u, 257u, 2047u, 2048u, 4097u, capacity, 257u, 2u, 0u}) {
            SCOPED_TRACE(count);
            const bool fully_culled = sequence++ == 2;
            std::vector<ProjectedSplat> splats(count);
            uint64_t expected_instances = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const float x = float(random() % width) + .5f, y = float(random() % height) + .5f;
                auto& s = splats[i];
                s.mean_depth = {x, y, 1 + float(random() % 11), 3};
                s.conic_opacity = {.5f, 0, .5f, .05f + float(random() % 800) / 1000};
                s.color = {float(random() % 100) / 100, float(random() % 100) / 100, float(random() % 100) / 100, 1 + float((i * 7) % 11)};
                s.bounds = {uint32_t(std::max(0.f, x - 6)), uint32_t(std::max(0.f, y - 6)), uint32_t(std::min(float(width), x + 7)), uint32_t(std::min(float(height), y + 7))};
                if (fully_culled || count == 2 || i % 19 == 3 || (i / 256) % 4 == 1)
                    s.bounds = {};
                else
                    expected_instances += ((s.bounds[2] + 15) / 16 - s.bounds[0] / 16) * ((s.bounds[3] + 15) / 16 - s.bounds[1] / 16);
            }
            auto input = splats.empty() ? Tensor::zeros({64}, Device::GPU, DataType::UInt8) : upload(splats);
            for (auto mode : {SplatRasterMode::Gaussian, SplatRasterMode::Points, SplatRasterMode::Discs}) {
                if (count > 4097 && mode != SplatRasterMode::Gaussian)
                    continue;
                auto r = params(count, width, height, mode, capacity * 6, bg);
                ASSERT_TRUE(rasterizer.rasterize(input, nullptr, count, mode, r));
                const auto actual = readback(rasterizer, width, height);
                EXPECT_EQ(actual.status.error, 0u);
                EXPECT_EQ(actual.status.required_instances, expected_instances);
                compare_cpu(actual, splats, width, height, bg, mode);
            }
        }
    }

    TEST_P(SplatRasterContracts, MedianThresholdSaturationOverflowGrowthAndSparseRecovery) {
        constexpr uint32_t w = 37, h = 29;
        const std::array<float, 4> bg{.15f, .1f, .2f, .4f};
        SplatRasterizer rasterizer(GetParam());
        ASSERT_TRUE(rasterizer.reserve(4097, w, h, 64));
        const std::vector<ProjectedSplat> threshold{
            {{{18, 14, 3, 3}}, {{1, 0, 1, .5f}}, {{1, 0, 0, 9}}, {{16, 12, 21, 17}}},
            {{{18, 14, 6, 3}}, {{1, 0, 1, .9f}}, {{0, 1, 0, 36}}, {{16, 12, 21, 17}}}};
        auto input = upload(threshold);
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 2, SplatRasterMode::Gaussian, params(2, w, h, SplatRasterMode::Gaussian, 64, bg)));
        compare_cpu(readback(rasterizer, w, h), threshold, w, h, bg, SplatRasterMode::Gaussian);

        const std::vector<ProjectedSplat> opaque{
            {{{18, 14, 3, 3}}, {{1, 0, 1, .95f}}, {{.7f, .1f, .1f, 9}}, {{16, 12, 21, 17}}},
            {{{18, 14, 6, 3}}, {{1, 0, 1, .999f}}, {{.8f, .8f, .8f, 36}}, {{16, 12, 21, 17}}}};
        input = upload(opaque);
        for (bool omit : {false, true}) {
            auto r = params(2, w, h, SplatRasterMode::Gaussian, 64, bg, 2 | (omit ? 32 : 0));
            r.clip[1] = 100;
            ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 2, SplatRasterMode::Gaussian, r));
            const auto actual = readback(rasterizer, w, h);
            const size_t at = (14 * w + 18) * 4;
            const float trans = omit ? 1 - .95f : (1 - .95f) * (1 - .999f);
            for (int c = 0; c < 3; ++c) {
                const float expected = opaque[0].color[c] * .95f + (omit ? 0 : .8f * (1 - .95f) * .999f) + bg[c] * bg[3] * trans;
                EXPECT_NEAR(half_to_float(actual.color[at + c]), expected, .001);
            }
            EXPECT_NEAR(actual.depth[at], 3 * .95f + 6 * (1 - .95f) * .999f, 1e-5);
            EXPECT_NEAR(actual.depth[at + 2], .95f + (1 - .95f) * .999f, 1e-6);
            EXPECT_NEAR(actual.depth[at + 1], 1 - trans, 1e-6);
            EXPECT_EQ(actual.depth[at + 3], 3);
        }

        const ProjectedSplat large{{18, 14, 1, 30}, {.01f, 0, .01f, .9f}, {1, 0, 0, 1}, {0, 0, w, h}};
        input = upload_one(large);
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian, params(1, w, h, SplatRasterMode::Gaussian, 1, bg)));
        auto actual = readback(rasterizer, w, h);
        EXPECT_EQ(actual.status.error, 1u);
        EXPECT_EQ(actual.status.required_instances, 6u);
        compare_cpu(actual, {}, w, h, bg, SplatRasterMode::Gaussian);
        auto dummy = Tensor::zeros({64}, Device::GPU, DataType::UInt8);
        ASSERT_TRUE(rasterizer.rasterize(dummy, nullptr, 0, SplatRasterMode::Gaussian, params(0, w, h, SplatRasterMode::Gaussian, 1, bg)));
        EXPECT_EQ(readback(rasterizer, w, h).status.error, 0u);

        std::vector<ProjectedSplat> sparse(4097, large);
        for (auto& s : sparse)
            s.bounds = {};
        sparse[0] = {{20, 4, 1, 3}, {1, 0, 1, .9f}, {1, 0, 0, 1}, {16, 0, 25, 9}};
        for (uint32_t phase = 0; phase < 4; ++phase) {
            if (phase == 2)
                sparse[0].bounds = large.bounds;
            else if (phase == 3)
                sparse[0].bounds = {};
            input = upload(sparse);
            ASSERT_TRUE(rasterizer.rasterize(input, nullptr, sparse.size(), SplatRasterMode::Gaussian,
                                             params(sparse.size(), w, h, SplatRasterMode::Gaussian, 1, bg)));
            actual = readback(rasterizer, w, h);
            EXPECT_EQ(actual.status.required_instances, phase == 2 ? 6u : phase == 3 ? 0u
                                                                                     : 1u);
            EXPECT_EQ(actual.status.error, phase == 2 ? 1u : 0u);
            compare_cpu(actual, phase == 2 ? std::vector<ProjectedSplat>{} : sparse, w, h, bg, SplatRasterMode::Gaussian);
        }
    }

    TEST_P(SplatRasterContracts, DepthChunksGrowthAndDenseGutAdaptationPreserveOutputs) {
        constexpr uint32_t w = 19, h = 17;
        const std::array<float, 4> bg{.1f, .2f, .3f, 1};
        for (uint32_t n : {8193u, 32769u}) {
            std::vector<ProjectedSplat> splats(n);
            SplatRasterizer rasterizer(GetParam());
            ASSERT_TRUE(rasterizer.reserve(n, w, h, n * 4));
            for (uint32_t scenario = 0; scenario < 4; ++scenario) {
                const uint32_t first = scenario == 0 ? 0 : 1088;
                const uint32_t stride = scenario == 0 ? 1 : scenario == 1 ? 97
                                                                          : 31;
                const float opacity = scenario == 1 ? .02f : .25f;
                for (uint32_t i = 0; i < n; ++i) {
                    const bool visible = i >= first && (i - first) % stride == 0;
                    splats[i] = {{visible ? 9.f : 1000.f, 8, .5f + i * .0001f, 20}, {.02f, .004f, .03f, opacity}, {float(i % 7) / 7, float(i % 11) / 11, float(i % 13) / 13, float(i / 4 + 1)}, {0, 0, w, h}};
                }
                auto input = upload(splats);
                for (uint32_t count : {n, 0u, 1u, n, n}) {
                    const bool expected_depth = scenario == 3;
                    auto r = params(count, w, h, SplatRasterMode::Gaussian, n * 4, bg, expected_depth ? 2 : 0);
                    r.clip[1] = 1.5f;
                    ASSERT_TRUE(rasterizer.rasterize(input, nullptr, count, SplatRasterMode::Gaussian, r));
                    const auto actual = readback(rasterizer, w, h);
                    EXPECT_EQ(actual.status.error, 0u);
                    EXPECT_EQ(actual.status.required_instances, uint64_t(count) * 4);
                    compare_cpu(actual, std::vector<ProjectedSplat>(splats.begin(), splats.begin() + count), w, h, bg,
                                SplatRasterMode::Gaussian, expected_depth, 1.5f);
                }
            }
            for (bool expanded : {false, false, true, true, true}) {
                for (auto& splat : splats)
                    splat.bounds = {0, 0, expanded ? w : 16, expanded ? h : 16};
                auto input = upload(splats);
                ASSERT_TRUE(rasterizer.rasterize(input, nullptr, n, SplatRasterMode::Gaussian, params(n, w, h, SplatRasterMode::Gaussian, n * 4, bg)));
                const auto actual = readback(rasterizer, w, h);
                EXPECT_EQ(actual.status.required_instances, uint64_t(n) * (expanded ? 4 : 1));
                compare_cpu(actual, splats, w, h, bg, SplatRasterMode::Gaussian);
            }
        }

        constexpr uint32_t n = 8193;
        std::vector<ProjectedSplat> splats(n);
        std::vector<GutSplat> guts(n);
        for (uint32_t i = 0; i < n; ++i) {
            splats[i] = {{9, 8, 3, 20}, {1, 0, 1, .05f}, {float(i % 7) / 7, float(i % 11) / 11, float(i % 13) / 13, float(i / 3 + 1)}, {0, 0, w, h}};
            guts[i] = {{2.5f, 0, 0, 2}, {0, 2.5f, 0, 0}, {0, 0, 2.5f, 0}, {float(int(i % 5) - 2) * .15f, float(int(i % 7) - 3) * .15f, 3 + .02f * (i % 8), .05f}};
        }
        auto input = upload(splats), geometry = upload(guts);
        SplatRasterizer adaptive(GetParam());
        ASSERT_TRUE(adaptive.reserve(n, w, h, n * 4));
        for (uint32_t model : {0u, 1u, 2u})
            for (bool spark : {false, true}) {
                for (uint32_t i = 0; i < n; ++i) {
                    splats[i].bounds[2] = model == 2 ? 32 : w;
                    guts[i].mean_opacity[3] = spark ? 1.2f : .05f;
                }
                input = upload(splats);
                geometry = upload(guts);
                std::vector<RasterReadback> dense_frames;
                for (uint32_t count : {n, n, 1u, 0u, n, n}) {
                    const std::array<float, 4> background = spark ? std::array<float, 4>{.1f, .2f, .3f, 1} : std::array<float, 4>{};
                    auto r = params(count, w, h, SplatRasterMode::Gut, n * 4, background, spark ? 16 : 0);
                    r.camera[2] = model;
                    r.panorama = {float(w), float(h), 0, 0};
                    r.intrinsics = {17, 17, w * .5f, h * .5f};
                    ASSERT_TRUE(adaptive.rasterize(input, &geometry, count, SplatRasterMode::Gut, r));
                    const auto actual = readback(adaptive, w, h);
                    EXPECT_EQ(actual.status.error, 0u);
                    EXPECT_EQ(actual.status.required_instances, uint64_t(count) * 4);
                    if (count == n)
                        dense_frames.push_back(actual);
                }
                ASSERT_EQ(dense_frames.size(), 4u);
                EXPECT_EQ(dense_frames[0].color, dense_frames[1].color);
                EXPECT_EQ(dense_frames[0].depth, dense_frames[1].depth);
                EXPECT_EQ(dense_frames[0].pick, dense_frames[1].pick);
                EXPECT_EQ(dense_frames[0].color, dense_frames[3].color);
                EXPECT_EQ(dense_frames[0].depth, dense_frames[3].depth);
                EXPECT_EQ(dense_frames[0].pick, dense_frames[3].pick);
            }
    }

    TEST_P(SplatRasterContracts, WeakTransparentLayersUnalignedMacroCropAndHalfRingThreshold) {
        {
            constexpr uint32_t n = 2145;
            std::vector<ProjectedSplat> splats(n);
            for (uint32_t i = 0; i < n; ++i)
                splats[i] = {{0, 0, 5, 1}, {1, 0, 1, i ? 1.f / 256 : 15.f / 16}, {i ? .8f : .4f, i ? .5f : .3f, i ? .25f : .2f, float(i + 1)}, {0, 0, 1, 1}};
            for (uint32_t i = 1; i <= 96; ++i)
                splats[i].mean_depth[0] = splats[i].mean_depth[1] = 3;
            auto input = upload(splats);
            SplatRasterizer rasterizer(GetParam());
            ASSERT_TRUE(rasterizer.reserve(n, 1, 1, n));
            ASSERT_TRUE(rasterizer.rasterize(input, nullptr, n, SplatRasterMode::Gaussian,
                                             params(n, 1, 1, SplatRasterMode::Gaussian, n, {}, 64)));
            const auto actual = readback(rasterizer, 1, 1);
            double rgb[3]{}, trans = 1;
            for (const auto& s : splats) {
                const double a = s.conic_opacity[3] * std::exp(-.5 * (s.mean_depth[0] * s.mean_depth[0] + s.mean_depth[1] * s.mean_depth[1]));
                if (a < .5 / 255)
                    continue;
                for (int c = 0; c < 3; ++c)
                    rgb[c] += s.color[c] * a * trans;
                trans *= 1 - a;
                if (trans < 1e-4)
                    break;
            }
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(half_to_float(actual.color[c]), rgb[c], .003);
            EXPECT_NEAR(actual.depth[1], 1 - trans, .0001);
            EXPECT_EQ(actual.pick[0], 0u);
            EXPECT_EQ(actual.depth[3], 5);
        }
        {
            constexpr uint32_t w = 129, h = 97, cw = 33, ch = 27, ox = 59, oy = 29;
            std::array<ProjectedSplat, 2> source{{{{63.4f, 31.4f, 5, 100}, {.16f, .015f, .12f, .8f}, {.55f, .2f, .8f, 25}, {0, 0, w, h}},
                                                  {{72.2f, 41.1f, 6, 100}, {.12f, -.02f, .2f, .65f}, {.2f, .8f, .4f, 36}, {0, 0, w, h}}}};
            auto input = upload(source);
            SplatRasterizer full(GetParam()), crop(GetParam());
            ASSERT_TRUE(full.reserve(2, w, h, 126));
            ASSERT_TRUE(crop.reserve(2, cw, ch, 12));
            ASSERT_TRUE(full.rasterize(input, nullptr, 2, SplatRasterMode::Gaussian, params(2, w, h, SplatRasterMode::Gaussian, 126, {}, 64)));
            const auto a = readback(full, w, h);
            for (auto& s : source) {
                s.mean_depth[0] -= ox;
                s.mean_depth[1] -= oy;
                s.bounds = {0, 0, cw, ch};
            }
            input = upload(source);
            auto r = params(2, cw, ch, SplatRasterMode::Gaussian, 12, {}, 64);
            r.render_origin = {ox, oy, 0, 0};
            ASSERT_TRUE(crop.rasterize(input, nullptr, 2, SplatRasterMode::Gaussian, r));
            const auto b = readback(crop, cw, ch);
            for (uint32_t y = 0; y < ch; ++y)
                for (uint32_t x = 0; x < cw; ++x) {
                    const size_t left = size_t(y + oy) * w + x + ox, right = size_t(y) * cw + x;
                    for (int c = 0; c < 4; ++c) {
                        EXPECT_EQ(a.color[left * 4 + c], b.color[right * 4 + c]);
                        EXPECT_EQ(a.depth[left * 4 + c], b.depth[right * 4 + c]);
                    }
                    EXPECT_EQ(a.pick[left], b.pick[right]);
                }
        }
        {
            const std::array<ProjectedSplat, 2> source{{{{34.547904968f, 66.692443848f, 4, 3}, {2.882635355f, .016116982f, 2.940344095f, .790994585f}, {.4f, .2f, .1f, 16}, {0, 0, 128, 96}},
                                                        {{82.307426453f, 71.455528259f, 4, 3}, {2.921408415f, -.013935118f, 2.862745047f, .772662878f}, {.4f, .2f, .1f, 16}, {0, 0, 128, 96}}}};
            const std::array<std::array<uint32_t, 2>, 2> pixels{{{{33, 68}}, {{81, 73}}}};
            std::array<std::array<float, 4>, 207> parameters{};
            parameters[21][3] = .02f;
            parameters[22][0] = 1;
            std::array<std::array<float, 4>, 128> colors{};
            const std::array<uint32_t, 1> flags{};
            auto p = upload(parameters), f = upload(flags);
            SplatRasterizer rasterizer(GetParam());
            ASSERT_TRUE(rasterizer.reserve(1, 128, 96, 48));
            for (size_t n = 0; n < source.size(); ++n) {
                auto input = upload_one(source[n]);
                const SplatRasterOverlay overlay{&p, &f, nullptr, nullptr, std::as_bytes(std::span(colors))};
                ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian,
                                                 params(1, 128, 96, SplatRasterMode::Gaussian, 48, {}, 1 | 64), &overlay));
                const auto actual = readback(rasterizer, 128, 96);
                const size_t at = size_t(pixels[n][1]) * 128 + pixels[n][0];
                if (n == 0) {
                    EXPECT_EQ(half_to_float(actual.color[at * 4 + 3]), 0);
                    EXPECT_EQ(actual.pick[at], 0xffffffffu);
                } else {
                    EXPECT_NEAR(half_to_float(actual.color[at * 4 + 3]), .8f, .001);
                    EXPECT_EQ(actual.pick[at], 0u);
                }
            }
        }
    }

    TEST_P(SplatRasterContracts, GutPerspectivePanoramaExpectedDepthPortalAndLogicalIdsMatchOracles) {
        constexpr uint32_t w = 37, h = 29;
        const std::array<float, 4> bg{.15f, .1f, .2f, .4f};
        ProjectedSplat projected{{w * .5f - .5f, h * .5f - .5f, 99, 100}, {1000, 0, 1000, .1f}, {1, .25f, .125f, 9}, {0, 0, w, h}};
        GutSplat gut{{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 3, .8f}};
        auto input = upload_one(projected), geometry = upload_one(gut);
        SplatRasterizer rasterizer(GetParam());
        ASSERT_TRUE(rasterizer.reserve(2, w, h, 12));
        auto r = params(1, w, h, SplatRasterMode::Gut, 12, bg);
        r.intrinsics = {32, 32, w * .5f, h * .5f};
        ASSERT_TRUE(rasterizer.rasterize(input, &geometry, 1, SplatRasterMode::Gut, r));
        auto actual = readback(rasterizer, w, h);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                const double u = (x + .5 - .5 * w) / 32, v = (y + .5 - .5 * h) / 32;
                const double alpha = .8 * std::exp(-18 * (1 - 1 / (1 + u * u + v * v)));
                const bool contributes = alpha >= .5 / 255;
                const double a = contributes ? alpha : 0;
                const size_t at = size_t(y) * w + x;
                EXPECT_NEAR(half_to_float(actual.color[at * 4]), a + bg[0] * bg[3] * (1 - a), .001);
                EXPECT_NEAR(actual.depth[at * 4 + 1], a, 1e-5);
                EXPECT_EQ(actual.pick[at], contributes ? 0u : 0xffffffffu);
                if (contributes) {
                    const double z = 3 / (1 + u * u + v * v);
                    EXPECT_NEAR(actual.depth[at * 4], z * a, 1e-4);
                    EXPECT_NEAR(actual.depth[at * 4 + 2], z, 1e-4);
                    if (a > .5)
                        EXPECT_NEAR(actual.depth[at * 4 + 3], z, 1e-4);
                    else
                        EXPECT_GE(actual.depth[at * 4 + 3], 1e9);
                }
            }

        projected.bounds = {32, 0, 80, h};
        input = upload_one(projected);
        for (bool subregion : {false, true}) {
            r.camera[2] = 2;
            r.panorama = subregion ? std::array<float, 4>{float(w * 2), float(h * 2), float(w), float(h) / 2}
                                   : std::array<float, 4>{float(w), float(h), 0, 0};
            ASSERT_TRUE(rasterizer.rasterize(input, &geometry, 1, SplatRasterMode::Gut, r));
            actual = readback(rasterizer, w, h);
            EXPECT_EQ(actual.status.required_instances, 6u);
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x) {
                    const double az = 2 * M_PI * ((x + .5 + r.panorama[2]) / r.panorama[0] - .5);
                    const double el = M_PI * ((y + .5 + r.panorama[3]) / r.panorama[1] - .5);
                    const double ray_z = std::cos(az) * std::cos(el), alpha = .8 * std::exp(-18 * (1 - ray_z * ray_z));
                    const bool contributes = alpha >= .5 / 255;
                    const double a = contributes ? alpha : 0;
                    const size_t at = size_t(y) * w + x;
                    EXPECT_NEAR(half_to_float(actual.color[at * 4]), a + bg[0] * bg[3] * (1 - a), .001);
                    EXPECT_NEAR(actual.depth[at * 4 + 1], a, 1e-5);
                    EXPECT_EQ(actual.pick[at], contributes ? 0u : 0xffffffffu);
                    if (contributes) {
                        const double z = 3 * ray_z * ray_z;
                        if (ray_z > 0 && z > r.clip[0])
                            EXPECT_NEAR(actual.depth[at * 4 + 2], z, 1e-4);
                        else
                            EXPECT_GE(actual.depth[at * 4 + 2], 1e9);
                    }
                }
        }

        const std::array<ProjectedSplat, 2> mixed{{{{18, 14, 3, 100}, {1, 0, 1, .8f}, {1, 0, 0, 9}, {0, 0, w, h}},
                                                   {{18, 14, 3, 100}, {1, 0, 1, .8f}, {0, 1, 0, 9}, {0, 0, w, h}}}};
        const std::array<GutSplat, 2> mixed_gut{{{{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, -3, .8f}},
                                                 {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 3, .8f}}}};
        input = upload(mixed);
        geometry = upload(mixed_gut);
        for (float far : {100.f, 2.5f}) {
            r = params(2, w, h, SplatRasterMode::Gut, 12, {}, 2);
            r.intrinsics = {32, 32, w * .5f, h * .5f};
            r.clip[1] = far;
            ASSERT_TRUE(rasterizer.rasterize(input, &geometry, 2, SplatRasterMode::Gut, r));
            actual = readback(rasterizer, w, h);
            const size_t at = (14 * w + 18) * 4;
            EXPECT_NEAR(actual.depth[at + 1], .96f, 1e-5);
            if (far > 3) {
                EXPECT_NEAR(actual.depth[at], .48f, 1e-5);
                EXPECT_NEAR(actual.depth[at + 2], .16f, 1e-5);
                EXPECT_NEAR(actual.depth[at] / actual.depth[at + 2], 3, 1e-5);
            } else {
                EXPECT_EQ(actual.depth[at], 0);
                EXPECT_EQ(actual.depth[at + 2], 0);
            }
        }

        const ProjectedSplat portal{{18, 14, 3, 100}, {1, 0, 1, .8f}, {1, 0, 0, 9}, {0, 0, w, h}};
        input = upload_one(portal);
        r = params(1, w, h, SplatRasterMode::Gaussian, 12, {}, 4);
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian, r));
        actual = readback(rasterizer, w, h);
        for (uint32_t x = 18; x <= 22; ++x) {
            const double q = double(x - 18) * (x - 18), edge = std::exp(-4.);
            double alpha = .8 * std::max(0., (std::exp(-.5 * q) - edge) / (1 - edge));
            if (alpha < 1. / 255)
                alpha = 0;
            EXPECT_NEAR(half_to_float(actual.color[(14 * w + x) * 4 + 3]), alpha, .001);
        }

        const std::array<uint32_t, 1> logical_id{7};
        auto logical = upload(logical_id);
        const SplatRasterLogical logical_map{&logical, 8};
        r = params(1, w, h, SplatRasterMode::Gaussian, 12, {}, 8);
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian, r, nullptr, &logical_map));
        actual = readback(rasterizer, w, h);
        EXPECT_EQ(actual.pick[14 * w + 18], 7u);

        std::array<std::array<float, 4>, 207> overlay_params{};
        overlay_params[24][0] = 1;
        std::array<std::array<float, 4>, 258> colors{};
        colors[2] = {0, 1, 0, 1};
        const std::array<uint8_t, 2> selection{2, 2};
        const std::array<uint32_t, 1> overlay_flags{};
        auto op = upload(overlay_params), oc = upload(colors), os = upload(selection), of = upload(overlay_flags);
        const SplatRasterOverlay overlay{&op, &of, &os, nullptr, std::as_bytes(std::span(colors))};
        r = params(1, w, h, SplatRasterMode::Gaussian, 12, {}, 1 | 8);
        r.mask_limits[0] = 2;
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian, r, &overlay, &logical_map));
        actual = readback(rasterizer, w, h);
        EXPECT_EQ(actual.pick[14 * w + 18], 7u);
        EXPECT_NEAR(half_to_float(actual.color[(14 * w + 18) * 4]), .8f, .001);
        EXPECT_EQ(half_to_float(actual.color[(14 * w + 18) * 4 + 1]), 0);

        ProjectedSplat density = portal;
        density.conic_opacity[3] = 2;
        input = upload_one(density);
        r = params(1, w, h, SplatRasterMode::Gaussian, 12, {}, 16);
        ASSERT_TRUE(rasterizer.rasterize(input, nullptr, 1, SplatRasterMode::Gaussian, r));
        actual = readback(rasterizer, w, h);
        const double density_value = std::exp(3. / std::exp(1.));
        for (uint32_t x = 18; x <= 22; ++x) {
            const double power = .5 * double(x - 18) * (x - 18);
            double alpha = power > .5 * std::pow(std::sqrt(8.) + .7, 2) ? 0 : std::min(.999, 1 - std::pow(1 - std::exp(-power), density_value));
            if (alpha < .5 / 255)
                alpha = 0;
            EXPECT_NEAR(half_to_float(actual.color[(14 * w + x) * 4 + 3]), alpha, .001);
        }
    }

    TEST_P(SplatRasterContracts, PortalGutMedianUsesRayDepth) {
        const ProjectedSplat splat{{0, 0, 17, 2}, {1, 0, 1, .75f}, {.2f, .3f, .4f, 9}, {0, 0, 1, 1}};
        const GutSplat gut{{1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 1, 2}, {0, 0, 3, .75f}};
        auto input = upload_one(splat), geometry = upload_one(gut);
        SplatRasterizer rasterizer(GetParam());
        ASSERT_TRUE(rasterizer.reserve(1, 1, 1, 1));
        for (uint32_t model : {0u, 1u})
            for (bool exact : {false, true}) {
                auto r = params(1, 1, 1, SplatRasterMode::Gut, 1, {0, 0, 0, 1}, 4 | (exact ? 2048 : 0));
                r.intrinsics = {1, 1, .5f, .5f};
                r.camera[2] = model;
                ASSERT_TRUE(rasterizer.rasterize(input, &geometry, 1, SplatRasterMode::Gut, r));
                const auto actual = readback(rasterizer, 1, 1);
                EXPECT_NEAR(actual.depth[3], 3, 1e-6);
                EXPECT_NEAR(actual.depth[2], 3, 1e-6);
                EXPECT_NEAR(actual.depth[0], 2.25f, 1e-6);
            }
    }

    TEST_P(SplatRasterContracts, TightProjectionAndTileKeyWidthsPreserveEveryOutputBit) {
        constexpr uint32_t n = 257, w = 129, h = 97;
        std::vector<float> xyz(n * 3), logs(n * 3), rotations(n * 4), opacity(n), dc(n * 3);
        for (uint32_t i = 0; i < n; ++i) {
            xyz[i * 3] = float(int(i % 17) - 8) * .09f;
            xyz[i * 3 + 1] = float(int(i % 13) - 6) * .08f;
            xyz[i * 3 + 2] = 2 + float(i % 7) * .1f;
            logs[i * 3] = i % 11 == 0 ? 0 : -1.5f;
            logs[i * 3 + 1] = i % 11 == 0 ? -12 : -5;
            logs[i * 3 + 2] = -6;
            const float angle = float(i % 9) * .19f;
            rotations[i * 4] = std::cos(angle);
            rotations[i * 4 + 3] = std::sin(angle);
            opacity[i] = -5 + float(i % 11);
            for (uint32_t c = 0; c < 3; ++c)
                dc[i * 3 + c] = float(int((i + c * 3) % 13) - 6) * .1f;
        }
        auto m = upload(xyz), s = upload(logs), q = upload(rotations), a = upload(opacity), d = upload(dc);
        SplatSources source{&m, &s, &q, &a, &d, nullptr, nullptr, nullptr, n};
        SplatProjector projector(GetParam());
        SplatRasterizer loose_raster(GetParam()), tight_raster(GetParam());
        std::array<SplatRasterizer*, 2> rasters{&loose_raster, &tight_raster};
        ASSERT_TRUE(rasters[0]->reserve(n, w, h, n * 63));
        ASSERT_TRUE(rasters[1]->reserve(n, w, h, n * 63));
        auto loose = Tensor::empty({size_t(n) * 64}, Device::GPU, DataType::UInt8), tight = Tensor::empty({size_t(n) * 64}, Device::GPU, DataType::UInt8);
        size_t reduced = 0;
        for (uint32_t scenario = 0; scenario < 8; ++scenario) {
            auto camera = projection(w, h);
            camera.intrinsics = {100, 100, 64.25f, 48.75f};
            camera.clip_scale = {.01f, 1000, 1, scenario % 2 ? 0.f : .3f};
            camera.extent = {w, h, scenario / 4, scenario % 2};
            if (scenario % 4 >= 2)
                camera.model_to_world[12] = .7f;
            RasterReadback images[2];
            uint64_t instances[2]{};
            for (uint32_t variant = 0; variant < 2; ++variant) {
                Tensor& out = variant ? tight : loose;
                ASSERT_TRUE(projector.project(source, camera, 0, SplatPrimitive::Gaussian, variant, out));
                auto r = params(n, w, h, SplatRasterMode::Gaussian, n * 63, {.1f, .2f, .3f, 1});
                r.intrinsics = camera.intrinsics;
                r.clip = camera.clip_scale;
                r.camera = camera.extent;
                ASSERT_TRUE(rasters[variant]->rasterize(out, nullptr, n, SplatRasterMode::Gaussian, r));
                images[variant] = readback(*rasters[variant], w, h);
                instances[variant] = images[variant].status.required_instances;
            }
            EXPECT_LE(instances[1], instances[0]);
            reduced += instances[0] - instances[1];
            EXPECT_EQ(images[0].color, images[1].color);
            EXPECT_EQ(images[0].depth, images[1].depth);
            EXPECT_EQ(images[0].pick, images[1].pick);
        }
        EXPECT_GT(reduced, n);

        constexpr uint32_t wide_n = 4097;
        for (uint32_t width : {4096u, 4097u}) {
            std::vector<ProjectedSplat> splats(wide_n);
            for (uint32_t i = 0; i < wide_n; ++i) {
                const uint32_t x = i % width;
                splats[i] = {{float(x), 0, float(i % 7 + 1), 35}, {.02f, 0, 1, .2f}, {float(i % 11) / 11, float(i % 13) / 13, float(i % 17) / 17, float(i % 5 + 1)}, {x > 35 ? x - 35 : 0, 0, std::min(width, x + 36), 1}};
            }
            auto input = upload(splats);
            SplatRasterizer rasterizer(GetParam());
            ASSERT_TRUE(rasterizer.reserve(wide_n, width, 1, wide_n * 6));
            RasterReadback reference;
            bool have_reference = false;
            uint64_t instance_count = 0;
            for (uint32_t count : {wide_n, wide_n, 0u, wide_n, wide_n}) {
                ASSERT_TRUE(rasterizer.rasterize(input, nullptr, count, SplatRasterMode::Gaussian,
                                                 params(count, width, 1, SplatRasterMode::Gaussian, wide_n * 6, {.1f, .2f, .3f, 1})));
                const auto actual = readback(rasterizer, width, 1);
                EXPECT_EQ(actual.status.error, 0u);
                if (!count) {
                    compare_cpu(actual, {}, width, 1, {.1f, .2f, .3f, 1}, SplatRasterMode::Gaussian);
                    continue;
                }
                if (!have_reference) {
                    reference = actual;
                    have_reference = true;
                    instance_count = actual.status.required_instances;
                    EXPECT_GT(instance_count, wide_n * 5 / 4);
                } else {
                    EXPECT_EQ(actual.status.required_instances, instance_count);
                    EXPECT_EQ(actual.color, reference.color);
                    EXPECT_EQ(actual.depth, reference.depth);
                    EXPECT_EQ(actual.pick, reference.pick);
                }
            }
        }
    }

    TEST_P(SplatRasterContracts, GutSupportSphereCullingIsBitExactForSparseAndDenseLists) {
        constexpr uint32_t w = 129, h = 97;
        for (uint32_t n : {259u, 4097u}) {
            std::mt19937 random(1939);
            std::uniform_real_distribution<float> unit(0, 1);
            std::vector<float> means(n * 3), scales(n * 3), rotations(n * 4), sh0(n * 3), opacity(n);
            for (uint32_t i = 0; i < n; ++i) {
                const float angle = unit(random) * 6.28f;
                rotations[4 * i] = std::cos(angle * .5f);
                const float sine = std::sin(angle * .5f) / std::sqrt(3.f);
                for (uint32_t c = 0; c < 3; ++c) {
                    means[3 * i + c] = c == 2 ? .06f + 5 * unit(random) : (unit(random) - .5f) * 2;
                    scales[3 * i + c] = std::log(.008f + .18f * unit(random));
                    rotations[4 * i + c + 1] = sine;
                    sh0[3 * i + c] = (unit(random) - .5f) * 2;
                }
                opacity[i] = unit(random) * 5 - 2;
            }
            auto m = upload(means), s = upload(scales), q = upload(rotations), d = upload(sh0), a = upload(opacity);
            SplatSources source{&m, &s, &q, &a, &d, nullptr, nullptr, nullptr, n};
            SplatProjector projector(GetParam());
            SplatRasterizer rasterizer(GetParam());
            ASSERT_TRUE(rasterizer.reserve(n, w, h, n * 63));
            for (uint32_t camera_model : {0u, 1u})
                for (bool spark : {false, true})
                    for (bool ill : {false, true}) {
                        auto camera = projection(w, h);
                        camera.intrinsics = {64, 57, w * .5f, h * .5f};
                        camera.clip_scale = {.01f, 100, 1, .3f};
                        camera.extent[2] = camera_model;
                        camera.model_to_world[0] = 1.4f;
                        camera.model_to_world[4] = .35f;
                        camera.model_to_world[5] = ill ? 1e-7f : .7f;
                        camera.model_to_world[9] = .12f;
                        camera.model_to_world[10] = 1.1f;
                        camera.display[2] = spark;
                        std::vector<float> encoded;
                        Tensor encoded_tensor;
                        if (spark) {
                            encoded.resize(n);
                            for (uint32_t i = 0; i < n; ++i)
                                encoded[i] = .3f + float(i % 8) * .2f;
                            encoded_tensor = upload(encoded);
                            source.opacity = &encoded_tensor;
                        } else
                            source.opacity = &a;
                        auto projected = Tensor::empty({size_t(n) * 64}, Device::GPU, DataType::UInt8), gut = Tensor::empty({size_t(n) * 64}, Device::GPU, DataType::UInt8);
                        ASSERT_TRUE(projector.project(source, camera, 0, SplatPrimitive::Gut, false, projected, &gut));
                        auto splats = download<ProjectedSplat>(projected, n);
                        auto geometry = download<GutSplat>(gut, n);
                        size_t bounded = 0;
                        for (uint32_t i = 0; i < n; ++i)
                            if (splats[i].bounds[2] > splats[i].bounds[0] && splats[i].bounds[3] > splats[i].bounds[1]) {
                                EXPECT_TRUE(std::isfinite(geometry[i].inverse0[3]));
                                bounded += geometry[i].inverse0[3] > 0;
                                if (n >= 4096)
                                    splats[i].bounds = {0, 0, w, h};
                            }
                        EXPECT_EQ(ill ? bounded == 0 : bounded > 0, true);
                        auto projected_culled = upload(splats), gut_culled = upload(geometry);
                        auto r = params(n, w, h, SplatRasterMode::Gut, n * 63, {.1f, .2f, .3f, 1}, spark ? 16 : 0);
                        r.intrinsics = camera.intrinsics;
                        r.clip = camera.clip_scale;
                        r.camera = camera.extent;
                        ASSERT_TRUE(rasterizer.rasterize(projected_culled, &gut_culled, n, SplatRasterMode::Gut, r));
                        const auto culled = readback(rasterizer, w, h);
                        for (auto& g : geometry)
                            g.inverse0[3] = 0;
                        auto gut_full = upload(geometry);
                        ASSERT_TRUE(rasterizer.rasterize(projected_culled, &gut_full, n, SplatRasterMode::Gut, r));
                        const auto full = readback(rasterizer, w, h);
                        ASSERT_TRUE(rasterizer.rasterize(projected_culled, &gut_culled, n, SplatRasterMode::Gut, r));
                        const auto reused = readback(rasterizer, w, h);
                        EXPECT_EQ(culled.color, full.color);
                        EXPECT_EQ(culled.depth, full.depth);
                        EXPECT_EQ(culled.pick, full.pick);
                        EXPECT_EQ(reused.color, full.color);
                        EXPECT_EQ(reused.depth, full.depth);
                        EXPECT_EQ(reused.pick, full.pick);
                    }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SplatRasterContracts, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
