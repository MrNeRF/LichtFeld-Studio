/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_contract_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
    using namespace lfs::core;
    using namespace lfs::rendering;
    using namespace lfs::test::splat;

    class SplatProjectionContracts : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (backend_unavailable_or_cuda(GetParam()))
                GTEST_SKIP();
            scope_ = std::make_unique<GpuBackendScope>(GetParam());
        }
        std::unique_ptr<GpuBackendScope> scope_;
    };

    TEST_P(SplatProjectionContracts, StorageCodecsDegreesAndDispatchBoundariesMatchAnalyticSh) {
        SplatProjector projector(GetParam());
        for (uint32_t n : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 511u}) {
            SCOPED_TRACE(n);
            std::vector<float> xyz(n * 3), scales(n * 3, -3), rotations(n * 4), opacity(n, 5), dc(n * 3);
            for (uint32_t i = 0; i < n; ++i) {
                xyz[i * 3 + 2] = 3;
                rotations[i * 4] = 1;
            }
            auto means = upload(xyz), scale = upload(scales), rotation = upload(rotations), alpha = upload(opacity), sh0 = upload(dc);
            std::vector<float> bounds(((n + 255) / 256) * 2);
            for (size_t b = 0; b < bounds.size() / 2; ++b) {
                bounds[b * 2] = -.25f - float(b) / 8;
                bounds[b * 2 + 1] = .25f + float(b) / 4;
            }
            auto sh_bounds = upload(bounds);
            auto output = Tensor::empty({size_t(n) * 64}, Device::GPU, DataType::UInt8);
            for (uint32_t rest : {3u, 8u, 15u}) {
                const size_t padded = (n + 31) / 32 * 32, slots = (rest * 3 + 3) / 4;
                for (auto storage : {SplatShStorage::CanonicalFloat32, SplatShStorage::SwizzledFloat32,
                                     SplatShStorage::SwizzledFloat16, SplatShStorage::Q16}) {
                    SCOPED_TRACE(uint32_t(storage));
                    std::vector<float> canonical(size_t(n) * rest * 3), swizzled(padded * slots * 4);
                    std::vector<uint16_t> halves(swizzled.size()), codes(padded * rest * 3);
                    for (uint32_t i = 0; i < n; ++i)
                        for (uint32_t c = 0; c < rest * 3; ++c) {
                            const uint16_t q = (i + c) % 3 == 0 ? 0 : (i + c) % 3 == 1 ? 65535
                                                                                       : 32768;
                            codes[size_t(i / 32) * rest * 3 * 32 + c * 32 + i % 32] = q;
                            const float value = storage == SplatShStorage::Q16
                                                    ? std::fma(bounds[(i / 256) * 2 + 1] - bounds[(i / 256) * 2], float(q) / 65535, bounds[(i / 256) * 2])
                                                    : float(int((i + c) % 9) - 4) / 32;
                            canonical[(size_t(i) * rest * 3) + c] = value;
                            const size_t at = (size_t(i / 32) * slots * 32 + (c / 4) * 32 + i % 32) * 4 + c % 4;
                            swizzled[at] = value;
                            halves[at] = float_to_half(value);
                        }
                    auto rest_tensor = storage == SplatShStorage::CanonicalFloat32  ? upload(canonical)
                                       : storage == SplatShStorage::SwizzledFloat32 ? upload(swizzled)
                                       : storage == SplatShStorage::SwizzledFloat16 ? upload(halves)
                                                                                    : upload(codes);
                    SplatSources source{&means, &scale, &rotation, &alpha, &sh0, &rest_tensor,
                                        storage == SplatShStorage::Q16 ? &sh_bounds : nullptr,
                                        nullptr, n, rest, storage};
                    for (uint32_t degree = 0; degree <= 3; ++degree) {
                        if ((degree + 1) * (degree + 1) - 1 > rest)
                            continue;
                        ASSERT_TRUE(projector.project(source, projection(), degree, SplatPrimitive::Gaussian, false, output));
                        const auto result = download<ProjectedSplat>(output, n);
                        for (uint32_t i = 0; i < n; ++i) {
                            ASSERT_GT(result[i].bounds[2], result[i].bounds[0]);
                            EXPECT_NEAR(result[i].mean_depth[0], 127.5f, 1e-5);
                            for (uint32_t c = 0; c < 3; ++c) {
                                const auto value = [&](uint32_t k) { return canonical[(size_t(i) * rest + k) * 3 + c]; };
                                double expected = .5;
                                if (degree >= 1)
                                    expected += std::sqrt(3. / (4 * M_PI)) * value(1);
                                if (degree >= 2)
                                    expected += std::sqrt(5. / (4 * M_PI)) * value(5);
                                if (degree >= 3)
                                    expected += std::sqrt(7. / (4 * M_PI)) * value(11);
                                EXPECT_NEAR(result[i].color[c], std::max(0., expected), 2e-6);
                            }
                        }
                    }
                }
            }
        }
    }

    TEST_P(SplatProjectionContracts, PointsTransformsObjectsClippingDeletionAndExtentClamp) {
        constexpr uint32_t n = 33;
        std::vector<float> xyz(n * 3), scales(n * 3, -3), rotations(n * 4), opacity(n, 5), dc(n * 3);
        for (uint32_t i = 0; i < n; ++i) {
            xyz[i * 3 + 2] = 3;
            rotations[i * 4] = 1;
        }
        auto means = upload(xyz), scale = upload(scales), rotation = upload(rotations), alpha = upload(opacity), sh0 = upload(dc);
        auto output = Tensor::empty({size_t(n) * 64}, Device::GPU, DataType::UInt8);
        SplatSources source{&means, nullptr, nullptr, &alpha, &sh0, nullptr, nullptr, nullptr, n};
        SplatProjector projector(GetParam());
        ASSERT_TRUE(projector.project(source, projection(), 0, SplatPrimitive::Points, false, output));
        auto result = download<ProjectedSplat>(output, n);
        EXPECT_EQ(result[0].mean_depth[3], 2);

        auto clipped = projection();
        clipped.clip_scale[0] = 4;
        ASSERT_TRUE(projector.project(source, clipped, 0, SplatPrimitive::Points, false, output));
        result = download<ProjectedSplat>(output, n);
        for (const auto& s : result)
            EXPECT_EQ(s.bounds[2], 0u);

        auto moved = projection();
        moved.model_to_world[12] = .3f;
        ASSERT_TRUE(projector.project(source, moved, 0, SplatPrimitive::Points, false, output));
        result = download<ProjectedSplat>(output, n);
        EXPECT_NEAR(result[0].mean_depth[0], 147.5f, 1e-4);
        EXPECT_NEAR(result[0].color[3], 9.09f, 1e-4);
        std::vector<float> radial(n);
        for (uint32_t i = 0; i < n; ++i)
            radial[i] = result[i].color[3];
        for (float angle : {.13f, -.11f, .21f}) {
            auto rotated = moved;
            const float c = std::cos(angle), s = std::sin(angle);
            rotated.world_to_camera[0] = c;
            rotated.world_to_camera[2] = -s;
            rotated.world_to_camera[8] = s;
            rotated.world_to_camera[10] = c;
            ASSERT_TRUE(projector.project(source, rotated, 0, SplatPrimitive::Points, false, output));
            result = download<ProjectedSplat>(output, n);
            for (uint32_t i = 0; i < n; ++i)
                if (radial[i] && result[i].bounds[2] > result[i].bounds[0])
                    EXPECT_EQ(result[i].color[3], radial[i]);
        }
        EXPECT_EQ(download<float>(means, xyz.size())[0], 0);

        auto ortho = moved;
        ortho.extent[2] = 1;
        ASSERT_TRUE(projector.project(source, ortho, 0, SplatPrimitive::Points, false, output));
        EXPECT_NEAR(download<ProjectedSplat>(output, n)[0].mean_depth[0], 187.5f, 1e-4);

        std::array<SceneObject, 2> objects{};
        objects[0].model_to_world = objects[1].model_to_world = identity();
        objects[0].model_to_world[12] = .6f;
        objects[0].flags = {1, 0, 0, 0};
        std::vector<uint32_t> object_ids(n);
        for (uint32_t i = 0; i < n; ++i)
            object_ids[i] = i % 3;
        auto ids = upload(object_ids);
        source.objects = std::as_bytes(std::span(objects));
        source.object_indices = &ids;
        ASSERT_TRUE(projector.project(source, projection(), 0, SplatPrimitive::Points, false, output));
        result = download<ProjectedSplat>(output, n);
        for (uint32_t i = 0; i < n; ++i)
            if (i % 3 == 0)
                EXPECT_NEAR(result[i].mean_depth[0], 167.5f, 1e-4);
            else
                EXPECT_EQ(result[i].bounds[2], 0u);

        source.objects = {};
        source.object_indices = nullptr;
        source.scales = &means;
        source.rotations = &rotation;
        ASSERT_TRUE(projector.project(source, projection(), 0, SplatPrimitive::Gaussian, false, output));
        result = download<ProjectedSplat>(output, n);
        EXPECT_LE(result[0].mean_depth[3], 512.501f);
        EXPECT_GT(result[0].mean_depth[3], 0);
        std::vector<uint8_t> deleted(n, 1);
        auto deletion = upload(deleted);
        source.deleted = &deletion;
        ASSERT_TRUE(projector.project(source, projection(), 0, SplatPrimitive::Points, false, output));
        for (const auto& s : download<ProjectedSplat>(output, n))
            EXPECT_EQ(s.bounds[2], 0u);
    }

    TEST_P(SplatProjectionContracts, FullCameraCenterAdmissionMatchesIndependentBounds) {
        SplatProjector projector(GetParam());
        for (bool orthographic : {false, true}) {
            constexpr uint32_t n = 12;
            auto camera = projection(80, 80);
            camera.extent = {80, 80, orthographic ? 1u : 0u, 0};
            camera.intrinsics = {64, 64, 40, 40};
            camera.panorama = {80, 80, 0, 0};
            camera.display[3] = 1;
            std::array<float, n * 3> means{}, scales{}, dc{};
            std::array<float, n * 4> rotations{};
            std::array<float, n> opacity{};
            const std::array<float, 6> xs{-17, -16, -15, 95, 96, 97};
            for (uint32_t i = 0; i < n; ++i) {
                const float px = i < 6 ? xs[i] : 40, py = i >= 6 ? xs[i - 6] : 40;
                const float factor = orthographic ? 1.f : 4.f;
                means[i * 3] = (px - 40) * factor / 64;
                means[i * 3 + 1] = (py - 40) * factor / 64;
                means[i * 3 + 2] = 4;
                for (uint32_t c = 0; c < 3; ++c)
                    scales[i * 3 + c] = std::log(4.f);
                rotations[i * 4] = 1;
                opacity[i] = 3;
            }
            auto m = upload(means), s = upload(scales), q = upload(rotations), a = upload(opacity), d = upload(dc);
            SplatSources source{&m, &s, &q, &a, &d, nullptr, nullptr, nullptr, n};
            auto output = Tensor::empty({size_t(n) * 64}, Device::GPU, DataType::UInt8);
            for (bool cropped : {false, true}) {
                auto p = camera;
                if (cropped) {
                    p.extent[0] = 33;
                    p.extent[1] = 27;
                    p.intrinsics[2] -= 43;
                    p.intrinsics[3] -= 29;
                    p.panorama[2] = 43;
                    p.panorama[3] = 29;
                }
                ASSERT_TRUE(projector.project(source, p, 0, SplatPrimitive::Gaussian, false, output));
                const auto result = download<ProjectedSplat>(output, n);
                for (uint32_t i = 0; i < n; ++i) {
                    const float px = i < 6 ? xs[i] : 40, py = i >= 6 ? xs[i - 6] : 40;
                    EXPECT_EQ(result[i].bounds[2] > result[i].bounds[0], px >= -16 && px < 96 && py >= -16 && py < 96);
                }
            }
        }
    }

    TEST_P(SplatProjectionContracts, SphericalUtGaussianJacobianAndLodMatchIndependentOracles) {
        SplatProjector projector(GetParam());
        auto panorama = projection(128, 96);
        panorama.extent = {128, 96, 2, 0};
        panorama.panorama = {128, 96, 0, 0};
        const std::array<float, 3> logs{std::log(.05f), std::log(.07f), std::log(.08f)}, dc{};
        const std::array<float, 4> quaternion{1, 0, 0, 0};
        const std::array<float, 1> logit{2};
        auto s = upload(logs), q = upload(quaternion), a = upload(logit), d = upload(dc);
        auto projected = Tensor::empty({64}, Device::GPU, DataType::UInt8), gut = Tensor::empty({64}, Device::GPU, DataType::UInt8);
        const float lambda = .1f * .1f * 3.f - 3.f, denominator = 3.f + lambda;
        const double sigma_scale = std::sqrt(denominator), central = lambda / denominator, weight = 1. / (2 * denominator);
        for (const std::array<float, 3> mean : {std::array<float, 3>{1, .5f, 3}, {-1, -.5f, -3}, {.01f, .2f, -3}, {-.01f, .2f, -3}, {3, .1f, .2f}, {.1f, 3, .1f}}) {
            SCOPED_TRACE(testing::Message() << "mean=" << mean[0] << ',' << mean[1] << ',' << mean[2]);
            auto m = upload(mean);
            SplatSources source{&m, &s, &q, &a, &d, nullptr, nullptr, nullptr, 1};
            ASSERT_TRUE(projector.project(source, panorama, 0, SplatPrimitive::Gut, false, projected, &gut));
            std::array<std::array<double, 2>, 7> image{};
            for (size_t n = 0; n < image.size(); ++n) {
                std::array<double, 3> point{mean[0], mean[1], mean[2]};
                if (n)
                    point[(n - 1) % 3] += (n < 4 ? 1 : -1) * sigma_scale * std::exp(double(logs[(n - 1) % 3]));
                const double norm = std::hypot(point[0], point[1], point[2]);
                image[n] = {(std::atan2(point[0], point[2]) / (2 * M_PI) + .5) * 128, (std::asin(point[1] / norm) / M_PI + .5) * 96};
                if (n)
                    image[n][0] -= 128 * std::round((image[n][0] - image[0][0]) / 128);
            }
            std::array<double, 2> expected{};
            for (size_t n = 0; n < image.size(); ++n)
                for (size_t c = 0; c < 2; ++c)
                    expected[c] += (n ? weight : central) * image[n][c];
            expected[0] -= 128 * std::floor(expected[0] / 128);
            const auto result = download_one<ProjectedSplat>(projected);
            ASSERT_GT(result.bounds[2], result.bounds[0]);
            EXPECT_NEAR(result.mean_depth[0], expected[0] - .5, .005);
            EXPECT_NEAR(result.mean_depth[1], expected[1] - .5, .005);
            EXPECT_NEAR(result.mean_depth[2], std::hypot(mean[0], mean[1], mean[2]), 1e-5);
        }

        const std::array<float, 3> centered{0, 0, 3};
        auto m = upload(centered);
        SplatSources source{&m, &s, &q, &a, &d, nullptr, nullptr, nullptr, 1};
        ASSERT_TRUE(projector.project(source, panorama, 0, SplatPrimitive::Gaussian, false, projected));
        auto result = download_one<ProjectedSplat>(projected);
        const double vx = std::pow(128. * std::exp(double(logs[0])) / (6 * M_PI), 2) + panorama.clip_scale[3];
        const double vy = std::pow(96. * std::exp(double(logs[1])) / (3 * M_PI), 2) + panorama.clip_scale[3];
        EXPECT_NEAR(result.mean_depth[0], 63.5f, 1e-5);
        EXPECT_NEAR(result.mean_depth[1], 47.5f, 1e-5);
        EXPECT_NEAR(result.conic_opacity[0], 1 / vx, 1e-5);
        EXPECT_NEAR(result.conic_opacity[2], 1 / vy, 1e-5);
        EXPECT_NEAR(result.conic_opacity[1], 0, 1e-6);
        EXPECT_NEAR(result.mean_depth[2], 3, 1e-5);

        const std::array<float, 3> nonfinite{NAN, 0, 3};
        auto bad_m = upload(nonfinite);
        source.means = &bad_m;
        ASSERT_TRUE(projector.project(source, panorama, 0, SplatPrimitive::Gut, false, projected, &gut));
        EXPECT_EQ(download_one<ProjectedSplat>(projected).bounds[2], 0u);

        source.means = &m;
        const std::array<uint32_t, 2> cut_ids{0, 0xffffffffu}, logical{0, 0}, levels{1, 0};
        const std::array<float, 2> weights{.25f, 1};
        auto ci = upload(cut_ids), li = upload(logical), lv = upload(levels), wt = upload(weights);
        SplatLodCut cut{&ci, &li, &lv, &wt, nullptr, 2, true, 0};
        auto cut_output = Tensor::empty({128}, Device::GPU, DataType::UInt8);
        auto camera = projection();
        ASSERT_TRUE(projector.project(source, camera, 0, SplatPrimitive::Gaussian, false, cut_output, nullptr, nullptr, &cut));
        auto selected = download<ProjectedSplat>(cut_output, 2);
        EXPECT_GT(selected[0].bounds[2], 0u);
        EXPECT_EQ(selected[1].bounds[2], 0u);
        EXPECT_NEAR(selected[0].conic_opacity[3], .25f / (1 + std::exp(-2.f)), 1e-5);
        EXPECT_EQ(selected[0].color[0], 0);
        EXPECT_NEAR(selected[0].color[1], .5f, 1e-5);
        EXPECT_EQ(selected[0].color[2], 0);
        camera.display[2] = 1;
        ASSERT_TRUE(projector.project(source, camera, 0, SplatPrimitive::Gaussian, false, cut_output, nullptr, nullptr, &cut));
        selected = download<ProjectedSplat>(cut_output, 2);
        const double rx = std::pow(double(camera.intrinsics[0]) * std::exp(double(logs[0])) / 3, 2);
        const double ry = std::pow(double(camera.intrinsics[1]) * std::exp(double(logs[1])) / 3, 2);
        EXPECT_NEAR(selected[0].conic_opacity[3], 1.25 * std::sqrt(rx * ry / ((rx + .3) * (ry + .3))), 1e-5);
    }

    TEST_P(SplatProjectionContracts, ExportBandPreservesFullCameraCovariance) {
        SplatProjector projector(GetParam());
        auto means = upload(std::array<float, 3>{0, 0, 3});
        auto scales = upload(std::array<float, 3>{std::log(.7f), std::log(.7f), std::log(.7f)});
        auto rotation = upload(std::array<float, 4>{1, 0, 0, 0});
        auto opacity = upload(std::array<float, 1>{0});
        auto sh0 = upload(std::array<float, 3>{0, 0, 0});
        SplatSources source{&means, &scales, &rotation, &opacity, &sh0, nullptr, nullptr, nullptr, 1, 0, SplatShStorage::CanonicalFloat32};
        auto output = Tensor::empty({64}, Device::GPU, DataType::UInt8);
        auto full = projection();
        full.panorama = {256, 256, 0, 0};
        ASSERT_TRUE(projector.project(source, full, 0, SplatPrimitive::Gaussian, false, output));
        const auto reference = download_one<ProjectedSplat>(output);
        auto band = full;
        band.extent[1] = 64;
        band.intrinsics[3] -= 192;
        band.panorama[3] = 192;
        ASSERT_TRUE(projector.project(source, band, 0, SplatPrimitive::Gaussian, false, output));
        const auto cropped = download_one<ProjectedSplat>(output);
        ASSERT_GT(cropped.bounds[3], cropped.bounds[1]);
        for (size_t c = 0; c < 4; ++c)
            EXPECT_NEAR(cropped.conic_opacity[c], reference.conic_opacity[c], 1e-7);
        EXPECT_NEAR(cropped.mean_depth[0], reference.mean_depth[0], 1e-5);
        EXPECT_NEAR(cropped.mean_depth[1] + 192, reference.mean_depth[1], 1e-5);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SplatProjectionContracts, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
