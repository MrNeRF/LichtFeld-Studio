/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_contract_test_utils.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    using namespace lfs::rendering;
    using namespace lfs::test::splat;

    class SplatSparkGutContracts : public testing::TestWithParam<GpuBackend> {};

    TEST_P(SplatSparkGutContracts, RayDensityAndLodWeightMatchIndependentOracle) {
        if (backend_unavailable_or_cuda(GetParam())) GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        const std::array<float, 3> means{0, 0, 3}, scales{std::log(.18f), std::log(.18f), std::log(.18f)};
        const std::array<float, 4> rotation{1, 0, 0, 0};
        const std::array<float, 3> sh0{(.8f - .5f) / .28209479177387814f, (.4f - .5f) / .28209479177387814f, (.2f - .5f) / .28209479177387814f};
        auto xyz = upload(means), scale = upload(scales), quat = upload(rotation), rgb = upload(sh0);
        const std::array<uint32_t, 1> source_id{0}, logical_id{127};
        auto indices = upload(source_id), logical = upload(logical_id);
        SplatProjector projector(GetParam());
        for (uint32_t camera_model : {0u, 1u, 2u}) {
            const uint32_t width = camera_model == 2 ? 193 : 97, height = 97;
            auto camera = projection(width, height);
            camera.intrinsics = {64, 64, .5f * width, .5f * height};
            camera.clip_scale = {.01f, 100, 1, .3f};
            camera.extent = {width, height, camera_model, 0};
            camera.display[2] = 1;
            camera.panorama = {float(width), float(height), 0, 0};
            SplatRasterizer rasterizer(GetParam());
            ASSERT_TRUE(rasterizer.reserve(1, width, height, 256));
            auto projected = Tensor::empty({64}, Device::GPU, DataType::UInt8);
            auto gut = Tensor::empty({64}, Device::GPU, DataType::UInt8);
            for (bool mip : {false, true})
                for (float output_scale : {1.f, 2.f})
                    for (float opacity : {.3f, 1.f, 1.5f, 3.f, 5.f})
                        for (float weight : {1.f, .6f, .15f}) {
                            SCOPED_TRACE(testing::Message() << "camera=" << camera_model << " mip=" << mip << " scale=" << output_scale << " opacity=" << opacity << " weight=" << weight);
                            camera.extent[3] = mip;
                            camera.rasterization[0] = output_scale;
                            camera.clip_scale[3] = mip ? .1f : .3f;
                            const std::array<float, 1> encoded{opacity > 1 ? (opacity + 3) / 4 : opacity}, weights{weight};
                            auto alpha = upload(encoded), lod_weight = upload(weights);
                            SplatSources source{&xyz, &scale, &quat, &alpha, &rgb, nullptr, nullptr, nullptr, 1};
                            SplatLodCut cut{&indices, &logical, nullptr, &lod_weight, nullptr, 1, false, 128};
                            ASSERT_TRUE(projector.project(source, camera, 0, SplatPrimitive::Gut, false, projected, &gut, nullptr, &cut));
                            auto raster = raster_parameters(1, width, height, SplatRasterMode::Gut, 256, 16 | 8);
                            raster.intrinsics = camera.intrinsics; raster.clip = camera.clip_scale; raster.camera = camera.extent; raster.panorama = camera.panorama;
                            const SplatRasterLogical logical_map{&logical, 128};
                            ASSERT_TRUE(rasterizer.rasterize(projected, &gut, 1, SplatRasterMode::Gut, raster, nullptr, &logical_map));
                            const auto actual = readback(rasterizer, width, height);
                            ASSERT_EQ(actual.status.error, 0u);
                            const double effective = double(opacity) * weight;
                            const double power = effective > 1 ? .5 * std::pow(std::sqrt(8.) + .7 * (std::min(effective, 5.) - 1), 2)
                                                               : std::max(4., std::log(std::max(effective, .5 / 255) * 510));
                            const double density = effective > 1 ? std::exp((effective * effective - 1) / std::exp(1.)) : 0;
                            for (uint32_t y = 0; y < height; ++y)
                                for (uint32_t x = 0; x < width; ++x) {
                                    double distance2;
                                    bool forward = true;
                                    if (camera_model == 1) {
                                        const double px = (x + .5 - .5 * width) / 64, py = (y + .5 - .5 * height) / 64;
                                        distance2 = (px * px + py * py) / (.18 * .18);
                                    } else {
                                        double z;
                                        if (camera_model == 2) {
                                            const double longitude = ((x + .5) / width - .5) * 2 * M_PI;
                                            const double latitude = ((y + .5) / height - .5) * M_PI;
                                            z = std::cos(latitude) * std::cos(longitude); forward = z > 0;
                                        } else {
                                            const double px = (x + .5 - .5 * width) / 64, py = (y + .5 - .5 * height) / 64;
                                            z = 1 / std::sqrt(1 + px * px + py * py);
                                        }
                                        distance2 = 9 * (1 - z * z) / (.18 * .18);
                                    }
                                    const double value = std::exp(-.5 * distance2);
                                    double expected = !forward || .5 * distance2 > power ? 0 : density ? -std::expm1(density * std::log1p(-value)) : effective * value;
                                    expected = std::min(expected, double(.999f));
                                    if (expected < .5 / 255) expected = 0;
                                    const size_t at = (size_t(y) * width + x) * 4;
                                    EXPECT_NEAR(half_to_float(actual.color[at + 3]), expected, .0022) << "pixel=" << x << ',' << y << " q=" << distance2;
                                    EXPECT_NEAR(half_to_float(actual.color[at]), expected * .8, .0022);
                                    EXPECT_NEAR(half_to_float(actual.color[at + 1]), expected * .4, .0022);
                                    EXPECT_NEAR(half_to_float(actual.color[at + 2]), expected * .2, .0022);
                                }
                        }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SplatSparkGutContracts, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });
} // namespace
