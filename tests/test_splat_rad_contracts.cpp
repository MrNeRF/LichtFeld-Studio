/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_contract_test_utils.hpp"

#include "core/splat_data.hpp"
#include "core/tensor_rad.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    using namespace lfs::rendering;
    using namespace lfs::test::splat;

    class SplatRadContracts : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (backend_unavailable_or_cuda(GetParam()))
                GTEST_SKIP();
            scope_ = std::make_unique<GpuBackendScope>(GetParam());
        }
        std::unique_ptr<GpuBackendScope> scope_;
    };

    TEST_P(SplatRadContracts, SignedShHalfGeometryPageBoundariesAndSparseCuts) {
        constexpr uint32_t count = 131, page = 64;
        std::vector<float> xyz(count * 3);
        std::vector<uint16_t> rgb(count * 4), scales(count * 4), rotation(count * 4), alpha(count);
        for (uint32_t n = 0; n < count; ++n) {
            xyz[n * 3 + 2] = 3;
            alpha[n] = float_to_half(1);
            rotation[n * 4] = float_to_half(1);
            for (uint32_t c = 0; c < 3; ++c) {
                scales[n * 4 + c] = float_to_half(-3);
                rgb[n * 4 + c] = float_to_half(float(int((n + c) % 7) - 3) / 8);
            }
            rgb[n * 4 + 3] = scales[n * 4 + 3] = float_to_half(NAN);
        }
        std::array<std::array<float, 4>, 12> frames{};
        for (uint32_t p = 0; p < 3; ++p)
            frames[p * 4] = {.25f * (p + 1), .5f * (p + 1), .75f * (p + 1), NAN};
        auto means = upload(xyz), sh0 = upload(rgb), scale = upload(scales), quat = upload(rotation), opacity = upload(alpha), bounds = upload(frames);
        auto view = projection(32, 32);
        view.intrinsics = {40, 40, 16, 16};
        view.clip_scale = {.01f, 100, 1, .3f};
        auto output = Tensor::empty({size_t(count) * 64}, Device::GPU, DataType::UInt8);
        auto geometry = Tensor::empty({size_t(count) * 64}, Device::GPU, DataType::UInt8);
        const auto code = [](uint32_t n, uint32_t c) { return int8_t(int((n + c) % 3) * 127 - 127); };
        SplatProjector projector(GetParam());
        for (uint32_t rest : {3u, 8u, 15u}) {
            const uint32_t slots = (rest * 3 + 3) / 4, padded = (count + 31) / 32 * 32;
            std::vector<int8_t> codes(size_t(padded) * slots * 4);
            for (uint32_t n = 0; n < count; ++n)
                for (uint32_t c = 0; c < rest * 3; ++c)
                    codes[(size_t(n / 32) * slots * 32 + (c / 4) * 32 + n % 32) * 4 + c % 4] = code(n, c);
            auto rest_tensor = upload(codes);
            SplatSources source{&means, &scale, &quat, &opacity, &sh0, &rest_tensor, &bounds, nullptr,
                                count, rest, SplatShStorage::RadSigned8, true, 0, page};
            for (uint32_t degree = 0; degree <= 3; ++degree) {
                if ((degree + 1) * (degree + 1) - 1 > rest)
                    continue;
                for (auto primitive : {SplatPrimitive::Gaussian, SplatPrimitive::Points, SplatPrimitive::Discs, SplatPrimitive::Gut}) {
                    ASSERT_TRUE(projector.project(source, view, degree, primitive, false, output,
                                                  primitive == SplatPrimitive::Gut ? &geometry : nullptr));
                    const auto result = download<ProjectedSplat>(output, count);
                    const auto guts = primitive == SplatPrimitive::Gut ? download<GutSplat>(geometry, count) : std::vector<GutSplat>{};
                    for (uint32_t n = 0; n < count; ++n) {
                        ASSERT_GT(result[n].bounds[2], result[n].bounds[0]);
                        for (uint32_t c = 0; c < 3; ++c) {
                            const auto coefficient = [&](uint32_t k) {
                                const uint32_t component = k * 3 + c, band = component < 9 ? 0 : component < 24 ? 1
                                                                                                                : 2;
                                return double(code(n, component)) / 127 * frames[(n / page) * 4][band];
                            };
                            double expected = .5 + .2820947917738781 * half_to_float(rgb[n * 4 + c]);
                            if (degree >= 1)
                                expected += std::sqrt(3. / (4 * M_PI)) * coefficient(1);
                            if (degree >= 2)
                                expected += std::sqrt(5. / (4 * M_PI)) * coefficient(5);
                            if (degree >= 3)
                                expected += std::sqrt(7. / (4 * M_PI)) * coefficient(11);
                            EXPECT_NEAR(result[n].color[c], std::max(0., expected), 2e-6);
                        }
                        EXPECT_NEAR(result[n].conic_opacity[3], 1.f / (1 + std::exp(-1.f)), 1e-6);
                        if (primitive == SplatPrimitive::Gut) {
                            EXPECT_NEAR(guts[n].inverse0[0], std::exp(3.f), 1e-4);
                            EXPECT_NEAR(guts[n].inverse1[1], std::exp(3.f), 1e-4);
                        }
                    }
                }
            }

            if (rest == 15) {
                const std::array<uint32_t, 7> indices{130, 127, 64, 63, 32, 31, 0};
                const std::array<uint32_t, 7> logical_ids{1000, 900, 800, 700, 2, 1, 0};
                auto ids = upload(indices), logical = upload(logical_ids);
                SplatLodCut cut{&ids, &logical, nullptr, nullptr, nullptr, 7, false, 1001};
                ASSERT_TRUE(projector.project(source, view, 3, SplatPrimitive::Gaussian, false, output, nullptr, nullptr, &cut));
                auto selected = download<ProjectedSplat>(output, indices.size());
                for (size_t n = 0; n < indices.size(); ++n) {
                    const uint32_t physical = indices[n];
                    double expected = .5 + .2820947917738781 * half_to_float(rgb[physical * 4]);
                    for (const auto [k, basis] : std::array<std::pair<uint32_t, double>, 3>{{{1, std::sqrt(3. / (4 * M_PI))}, {5, std::sqrt(5. / (4 * M_PI))}, {11, std::sqrt(7. / (4 * M_PI))}}}) {
                        const uint32_t component = k * 3, band = component < 9 ? 0 : component < 24 ? 1
                                                                                                    : 2;
                        expected += basis * double(code(physical, component)) / 127 * frames[(physical / page) * 4][band];
                    }
                    EXPECT_NEAR(selected[n].color[0], std::max(0., expected), 2e-6);
                }
                const std::array<uint8_t, 3> deleted{1, 0, 0};
                auto mask = upload(deleted);
                source.deleted = &mask;
                source.deleted_count = 3;
                ASSERT_TRUE(projector.project(source, view, 3, SplatPrimitive::Gaussian, false, output, nullptr, nullptr, &cut));
                selected = download<ProjectedSplat>(output, indices.size());
                for (size_t n = 0; n < logical_ids.size(); ++n)
                    EXPECT_EQ(selected[n].bounds[2] > selected[n].bounds[0], logical_ids[n] != 0);
                cut.logical_count = 1000;
                ASSERT_TRUE(projector.project(source, view, 3, SplatPrimitive::Gaussian, false, output, nullptr, nullptr, &cut));
                EXPECT_EQ(download<ProjectedSplat>(output, indices.size())[0].bounds[2], 0u);
            }
        }
        for (uint32_t invalid : {0u, 63u}) {
            SplatSources invalid_source{&means, &scale, &quat, &opacity, &sh0, nullptr, &bounds, nullptr,
                                        count, 0, SplatShStorage::RadSigned8, true, 0, invalid};
            EXPECT_FALSE(projector.project(invalid_source, view, 0, SplatPrimitive::Gaussian, false, output));
        }
    }

    TEST_P(SplatRadContracts, ProductionQuantizerTailAndMutationOrderingFeedTensorProjector) {
        constexpr uint32_t n = 131, page_size = 64, capacity = 192;
        std::vector<float> xyz(n * 3), rgb(n * 3), scales(n * 3, -3), rotation(n * 4), opacity(n, 1), rest(n * 45);
        for (uint32_t s = 0; s < n; ++s) {
            xyz[s * 3 + 2] = 3;
            rotation[s * 4] = 1;
            for (uint32_t c = 0; c < 3; ++c)
                rgb[s * 3 + c] = float(int((s + c) % 7) - 3) / 8;
            for (uint32_t c = 0; c < 45; ++c)
                rest[(size_t(s) * 15 + c / 3) * 3 + c % 3] = float(int((s + c) % 13) - 6) * .01f;
        }
        for (int degree : {0, 3}) {
            SplatData model(degree, Tensor::from_vector(xyz, {n, 3}, Device::GPU), Tensor::from_vector(rgb, {n, 1, 3}, Device::GPU),
                            degree ? Tensor::from_vector(rest, {n, 15, 3}, Device::GPU) : Tensor{}, Tensor::from_vector(scales, {n, 3}, Device::GPU),
                            Tensor::from_vector(rotation, {n, 4}, Device::GPU), Tensor::from_vector(opacity, {n, 1}, Device::GPU), 1.f);
            if (degree)
                (void)model.apply_shN_value_quant();
            ASSERT_TRUE(!degree || model.shN_value_quantized());
            RadPagePool pool;
            pool.page_splats = page_size;
            pool.sh_slots = degree ? 12 : 0;
            const std::array<size_t, 7> sizes{capacity * 12, capacity * 8, size_t(capacity) * pool.sh_slots * 4,
                                              capacity * 8, capacity * 8, capacity * 2, 3 * 64};
            for (size_t r = 0; r < sizes.size(); ++r)
                if (sizes[r])
                    pool.regions[r] = Tensor::empty({sizes[r]}, Device::GPU, DataType::UInt8);
            for (uint32_t page = 0; page < 3; ++page) {
                RadPageSources page_source{model.means_raw(), model.sh0_raw(), degree ? model.shN_raw() : Tensor{}, model.rotation_raw(), model.scaling_raw(), model.opacity_raw(),
                                           degree ? model.shN_value_bounds() : Tensor{}, page * page_size, std::min(page_size, n - page * page_size), degree ? 15u : 0u, bool(degree)};
                rad_page_quantize(page_source, pool, page);
            }
            std::array<Tensor, 7> snapshot;
            for (size_t r = 0; r < sizes.size(); ++r)
                if (sizes[r])
                    snapshot[r] = pool.regions[r].clone();
            SplatSources source{&pool.regions[0], &pool.regions[4], &pool.regions[3], &pool.regions[5], &pool.regions[1],
                                degree ? &pool.regions[2] : nullptr, degree ? &pool.regions[6] : nullptr, nullptr,
                                capacity, degree ? 15u : 0u, SplatShStorage::RadSigned8, true, 0, page_size};
            auto output = Tensor::empty({size_t(capacity) * 64}, Device::GPU, DataType::UInt8);
            auto view = projection(32, 32);
            view.intrinsics = {40, 40, 16, 16};
            view.clip_scale = {.01f, 100, 1, .3f};
            SplatProjector projector(GetParam());
            ASSERT_TRUE(projector.project(source, view, degree, SplatPrimitive::Gaussian, false, output));
            pool.regions[1].zero_();
            const auto result = download<ProjectedSplat>(output, capacity);
            const auto dc = download<uint16_t>(snapshot[1], sizes[1] / 2);
            const auto codes = degree ? download<int8_t>(snapshot[2], sizes[2]) : std::vector<int8_t>{};
            const auto frames = degree ? download<float>(snapshot[6], sizes[6] / 4) : std::vector<float>{};
            for (uint32_t s = 0; s < capacity; ++s) {
                EXPECT_EQ(result[s].bounds[2] > result[s].bounds[0], s < n);
                if (s >= n)
                    continue;
                for (uint32_t c = 0; c < 3; ++c) {
                    double expected = .5 + .2820947917738781 * half_to_float(dc[s * 4 + c]);
                    if (degree)
                        for (const auto [k, basis] : std::array<std::pair<uint32_t, double>, 3>{{{1, std::sqrt(3. / (4 * M_PI))}, {5, std::sqrt(5. / (4 * M_PI))}, {11, std::sqrt(7. / (4 * M_PI))}}}) {
                            const uint32_t component = k * 3 + c, band = component < 9 ? 0 : component < 24 ? 1
                                                                                                            : 2;
                            const size_t at = (size_t(s / 32) * pool.sh_slots * 32 + (component / 4) * 32 + s % 32) * 4 + component % 4;
                            expected += basis * double(codes[at]) / 127 * frames[(s / page_size) * 16 + band];
                        }
                    EXPECT_NEAR(result[s].color[c], std::max(0., expected), 2e-6);
                }
            }
            EXPECT_NE(download<uint8_t>(pool.regions[1], sizes[1]), download<uint8_t>(snapshot[1], sizes[1]));
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SplatRadContracts, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
