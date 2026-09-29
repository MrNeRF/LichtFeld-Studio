/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_elapsed.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include <chrono>
#include <iostream>
#if defined(LFS_TEST_TENSOR_VULKAN)
#include "lfs/training/ops/gsplat_vulkan.hpp"
#endif
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs;
    using core::Device;
    using core::GpuBackend;
    using core::Tensor;
    namespace ops = gpu_ops;
    struct Result {
        std::vector<float> image, alpha, depth, densification, scores, share;
        std::array<std::vector<float>, 6> gradients;
        std::vector<ops::AdamSlot> requested;
    };
    struct CameraCase {
        core::CameraModelType model = core::CameraModelType::PINHOLE;
        bool distorted = false;
        ops::ShStorage storage = ops::ShStorage::Float32;
        size_t count = 5;
        bool background_image = false;
        bool transform = false;
        bool saturated = false;
        float mean_z_offset = 0.f;
        bool depth_gradient = false;
    };
    Result render(GpuBackend backend, ops::GsplatRenderMode mode = ops::GsplatRenderMode::RGB,
                  bool invisible = false, bool tile = false, bool backward = false, unsigned degree = 0, bool error_map = true, CameraCase camera = {}) {
        core::GpuBackendScope scope(backend);
        const auto* table = training::training_ops(backend).gsplat;
        ops::GsplatSaved saved;
#if defined(LFS_TEST_TENSOR_VULKAN)
        saved.backend = backend == GpuBackend::Vulkan ? training::vulkan::gsplat_create() : table->create();
#else
        saved.backend = table->create();
#endif
        const size_t n = camera.count;
        auto make = [&](const std::vector<float>& pattern, size_t columns) {
            std::vector<float> data(n * columns);
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = pattern[i % pattern.size()];
            return Tensor::from_vector(data, {n, columns}, Device::GPU);
        };
        auto means = make(std::vector<float>{-.12f, .05f, 3.f, .18f, -.09f, 3.3f, .01f, .1f, 2.8f, 0.f, 0.f, -2.f, 40.f, 0.f, 3.f}, 3);
        if (camera.mean_z_offset != 0.f) {
            auto values = means.to_vector();
            values[2] += camera.mean_z_offset;
            means = Tensor::from_vector(values, {n, 3}, Device::GPU);
        }
        if (invisible)
            means = Tensor::full({n, 3}, -10.f, Device::GPU);
        auto scales = make(std::vector<float>{-1.8f, -1.3f, -1.6f, -1.3f, -2.f, -1.1f, -1.6f, -1.5f, -1.7f, -2.f, -2.f, -2.f, -2.f, -2.f, -2.f}, 3);
        auto quats = make(std::vector<float>{1.f, .2f, -.1f, .3f, .9f, -.1f, .3f, .2f, 1.f, .05f, .2f, -.15f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f}, 4);
        auto opacity = Tensor::full({n, 1}, camera.saturated ? 12.f : 1.2f, Device::GPU);
        auto dc = make(std::vector<float>{.1f, .3f, -.4f, .9f, -.6f, .2f, -.3f, .4f, .6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, 3).reshape({int(n), 1, 3});
        auto view = Tensor::eye(4, Device::GPU), bg = Tensor::full({3}, .15f, Device::GPU);
        if (camera.transform)
            view = Tensor::from_vector(std::vector<float>{.9950042f, 0.f, .09983342f, .05f, 0.f, 1.f, 0.f, -.04f, -.09983342f, 0.f, .9950042f, .08f, 0.f, 0.f, 0.f, 1.f}, {4, 4}, Device::GPU);
        Tensor empty, image, alpha, depth, normal, bounds, radial, tangential, bg_image;
        if (camera.distorted) {
            if (camera.model == core::CameraModelType::PINHOLE) {
                radial = Tensor::from_vector(std::vector<float>{.02f, -.003f, .0004f, .001f, .0001f, 0.f}, {6}, Device::CPU);
                tangential = Tensor::from_vector(std::vector<float>{.001f, -.002f}, {2}, Device::CPU);
            } else {
                radial = Tensor::from_vector(std::vector<float>{.01f, -.002f, .0003f, -.00001f}, {4}, Device::CPU);
                tangential = Tensor::from_vector(std::vector<float>{.001f, -.002f, .0005f, -.0003f}, {4}, Device::CPU);
            }
        }
        if (camera.background_image)
            bg_image = Tensor::full({3, tile ? 17ul : 19ul, tile ? 13ul : 17ul}, .23f, Device::GPU);
        const size_t rest_cells = core::sh_swizzled_float_count(n, 15);
        std::vector<float> rest_values(rest_cells);
        for (size_t i = 0; i < rest_values.size(); ++i)
            rest_values[i] = .04f * std::sin(float(i) * .17f);
        auto rest = Tensor::from_vector(rest_values, {rest_cells}, Device::GPU);
        if (camera.storage == ops::ShStorage::IeeeFloat16) {
            rest = rest.to(core::DataType::Float16);
            // CUDA Gsplat accepts Float32 and Q16 only. Use the same rounded
            // values in Float32 as the oracle for Vulkan IEEE-half reads.
            if (backend == GpuBackend::CUDA)
                rest = rest.to(core::DataType::Float32);
        }
        if (camera.storage == ops::ShStorage::Q16) {
            auto codes = Tensor::empty({core::sh_value_quant::sh_value_u16_count(n, 15)}, Device::GPU, core::DataType::Float16);
            bounds = Tensor::empty({2 * core::sh_value_quant::n_bounds_for_prims(n)}, Device::GPU);
            training::training_ops(backend).sh->encode_q16(rest, codes, bounds, n, 15, 0, 0);
            rest = codes;
            if (backend == GpuBackend::CUDA) {
                // Decode the oracle with the explicit CUDA family: the CUDA
                // raster service otherwise selects SH from the process default.
                auto decoded = Tensor::empty({rest_cells}, Device::GPU);
                training::training_ops(backend).sh->decode_q16(codes, bounds, decoded, n, 15);
                rest = decoded;
                bounds = {};
            }
        }
        ops::GsplatParams p{.full_image = {19, 17}, .intrinsics = {18.f, 19.f, 8.5f, 9.5f}, .sh = {.storage = backend == GpuBackend::CUDA ? ops::ShStorage::Float32 : camera.storage, .active_bases = (degree + 1) * (degree + 1), .layout_bases = 16}, .camera_model = camera.model, .render_mode = mode, .scaling_modifier = camera.transform ? 1.3f : 1.f, .antialiased = camera.distorted};
        if (tile) {
            p.tile_x = 2;
            p.tile_y = 1;
            p.tile_w = 13;
            p.tile_h = 17;
        }
        ops::SplatInputs inputs{means, scales, quats, opacity, dc, rest, bounds};
        ops::RasterResult status;
#if defined(LFS_TEST_TENSOR_VULKAN)
        if (backend == GpuBackend::Vulkan)
            status = training::vulkan::gsplat_forward(saved, inputs, view, radial, tangential, bg, bg_image, p, {image, alpha, depth, normal});
        else
#endif
            status = table->forward(saved, inputs, view, radial, tangential, bg, bg_image, p, {image, alpha, depth, normal});
        EXPECT_EQ(status.code, ops::RasterResult::Code::Success) << status.message;
        EXPECT_FALSE(normal.is_valid());
        Result result;
        if (image.is_valid())
            result.image = image.to_vector();
        if (alpha.is_valid())
            result.alpha = alpha.to_vector();
        if (depth.is_valid())
            result.depth = depth.to_vector();
        if (backward) {
            size_t height = tile ? 17 : 19, width = tile ? 13 : 17;
            std::vector<float> upstream(3 * height * width), alpha_upstream(height * width);
            for (size_t i = 0; i < upstream.size(); ++i)
                upstream[i] = .015f * std::sin(float(i) * .23f);
            for (size_t i = 0; i < alpha_upstream.size(); ++i)
                alpha_upstream[i] = .01f * std::cos(float(i) * .31f);
            auto gi = Tensor::from_vector(upstream, {3, height, width}, Device::GPU);
            auto ga = Tensor::from_vector(alpha_upstream, {1, height, width}, Device::GPU);
            if (camera.depth_gradient) {
                const size_t channels = mode == ops::GsplatRenderMode::D || mode == ops::GsplatRenderMode::ED ? 1 : 4;
                std::vector<float> depth_upstream(channels * height * width, 0.f);
                depth_upstream[(channels - 1) * height * width + (height / 2) * width + width / 2] = 1.f;
                gi = Tensor::from_vector(depth_upstream, {channels, height, width}, Device::GPU);
                ga.zero_();
            }
            struct Slots {
                std::array<Tensor, 6> tensors;
                std::vector<ops::AdamSlot> requested;
            } slots;
            slots.tensors = {Tensor::full({n, 3}, .001f, Device::GPU), Tensor::full({n, 3}, .002f, Device::GPU),
                             Tensor::full({n, 4}, .003f, Device::GPU), Tensor::full({n, 1}, .004f, Device::GPU),
                             Tensor::full({n, 1, 3}, .005f, Device::GPU), Tensor::full({rest_cells}, .006f, Device::GPU)};
            ops::GsplatGradients outputs{&slots, [](void* owner, ops::AdamSlot slot) -> Tensor& {
                                             auto& s = *static_cast<Slots*>(owner);
                                             s.requested.push_back(slot);
                                             return s.tensors[size_t(slot)];
                                         }};
            auto dens = Tensor::full({2, n}, .01f, Device::GPU), scores = Tensor::full({n}, .02f, Device::GPU), shares = Tensor::zeros({n}, Device::GPU);
            auto errors = error_map ? Tensor::full({height, width}, .3f, Device::GPU) : Tensor{};
            auto edges = Tensor::full({height, width}, .7f, Device::GPU);
            const auto before = means.to_vector();
            table->backward(saved, gi, ga, outputs, dens, errors, edges, scores, shares);
            EXPECT_EQ(means.to_vector(), before);
            for (size_t i = 0; i < 6; ++i)
                result.gradients[i] = slots.tensors[i].to_vector();
            result.requested = slots.requested;
            result.densification = dens.to_vector();
            result.scores = scores.to_vector();
            result.share = shares.to_vector();
        }
#if defined(LFS_TEST_TENSOR_VULKAN)
        if (backend == GpuBackend::Vulkan)
            training::vulkan::gsplat_release(saved);
        else
#endif
            table->release(saved);
        return result;
    }
    void near(const std::vector<float>& a, const std::vector<float>& b) {
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i)
            ASSERT_NEAR(a[i], b[i], 1e-4f + 1e-4f * std::abs(b[i])) << "element " << i;
    }
    void compare(const Result& actual, const Result& expected) {
        near(actual.image, expected.image);
        near(actual.alpha, expected.alpha);
        near(actual.depth, expected.depth);
        for (size_t i = 0; i < 6; ++i) {
            SCOPED_TRACE("slot=" + std::to_string(i));
            near(actual.gradients[i], expected.gradients[i]);
        }
        {
            SCOPED_TRACE("densification");
            near(actual.densification, expected.densification);
        }
        {
            SCOPED_TRACE("edge scores");
            near(actual.scores, expected.scores);
        }
        {
            SCOPED_TRACE("screen share");
            near(actual.share, expected.share);
        }
        EXPECT_EQ(actual.requested, expected.requested);
    }
    TEST(VulkanGsplat, PinholeForwardMatchesCuda) {
        if (!core::gpu_backend_available(GpuBackend::Vulkan) || !core::gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        for (bool tile : {false, true})
            for (bool invisible : {false, true}) {
                auto expected = render(GpuBackend::CUDA, ops::GsplatRenderMode::RGB, invisible, tile);
                auto actual = render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB, invisible, tile);
                near(actual.image, expected.image);
                near(actual.alpha, expected.alpha);
            }
    }
    TEST(VulkanGsplat, DepthModesPreserveRgbAndNormalizeDepth) {
        if (!core::gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        auto rgb = render(GpuBackend::Vulkan);
        auto both = render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB_D);
        auto normalized = render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB_ED);
        auto depth = render(GpuBackend::Vulkan, ops::GsplatRenderMode::D);
        auto expected = render(GpuBackend::Vulkan, ops::GsplatRenderMode::ED);
        near(rgb.image, both.image);
        near(rgb.alpha, both.alpha);
        near(both.depth, depth.depth);
        near(normalized.depth, expected.depth);
        ASSERT_EQ(both.depth.size(), both.alpha.size());
        for (size_t i = 0; i < both.depth.size(); ++i) {
            EXPECT_TRUE(std::isfinite(both.depth[i]));
            EXPECT_NEAR(normalized.depth[i], both.depth[i] / std::max(both.alpha[i], 1e-10f), 1e-5f);
        }
    }
    TEST(VulkanGsplat, SingleGaussianDepthHasKnownWorldValue) {
        if (!core::gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const CameraCase camera{.count = 1};
        auto accumulated = render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB_D, false, false, false, 0, true, camera);
        auto normalized = render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB_ED, false, false, false, 0, true, camera);
        size_t covered = 0;
        for (size_t i = 0; i < accumulated.alpha.size(); ++i) {
            EXPECT_NEAR(accumulated.depth[i], 3.f * accumulated.alpha[i], 1e-5f);
            if (accumulated.alpha[i] > .01f) {
                EXPECT_NEAR(normalized.depth[i], 3.f, 1e-5f);
                ++covered;
            }
        }
        EXPECT_GT(covered, 10u);
    }
    TEST(VulkanGsplat, DepthBackwardMatchesFiniteDifference) {
        if (!core::gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        constexpr float step = .005f;
        constexpr size_t pixel = (19 / 2) * 17 + 17 / 2;
        for (const auto mode : {ops::GsplatRenderMode::D, ops::GsplatRenderMode::ED, ops::GsplatRenderMode::RGB_D, ops::GsplatRenderMode::RGB_ED}) {
            for (bool transform : {false, true}) {
                SCOPED_TRACE("mode=" + std::to_string(int(mode)) + " transform=" + std::to_string(transform));
                CameraCase camera{.count = 1, .transform = transform, .depth_gradient = true};
                auto analytic = render(GpuBackend::Vulkan, mode, false, false, true, 0, false, camera);
                camera.mean_z_offset = step;
                auto plus = render(GpuBackend::Vulkan, mode, false, false, false, 0, false, camera);
                camera.mean_z_offset = -step;
                auto minus = render(GpuBackend::Vulkan, mode, false, false, false, 0, false, camera);
                const float numerical = (plus.depth[pixel] - minus.depth[pixel]) / (2 * step);
                EXPECT_NEAR(analytic.gradients[0][2] - .001f, numerical, .002f);
            }
        }
    }
    TEST(VulkanGsplat, NonzeroBackwardMatchesCuda) {
        if (!core::gpu_backend_available(GpuBackend::Vulkan) || !core::gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        for (unsigned degree : {0u, 1u, 2u, 3u})
            for (bool errors : {false, true})
                for (bool invisible : {false, true}) {
                    SCOPED_TRACE("degree=" + std::to_string(degree) + " errors=" + std::to_string(errors) + " invisible=" + std::to_string(invisible));
                    auto expected = render(GpuBackend::CUDA, ops::GsplatRenderMode::RGB, invisible, true, true, degree, errors);
                    auto actual = render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB, invisible, true, true, degree, errors);
                    near(actual.image, expected.image);
                    near(actual.alpha, expected.alpha);
                    for (size_t i = 0; i < 6; ++i) {
                        SCOPED_TRACE("slot=" + std::to_string(i));
                        near(actual.gradients[i], expected.gradients[i]);
                    }
                    near(actual.densification, expected.densification);
                    near(actual.scores, expected.scores);
                    near(actual.share, expected.share);
                    EXPECT_EQ(actual.requested, expected.requested);
                }
    }
    TEST(VulkanGsplat, CameraModelsAndDistortion) {
        if (!core::gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        for (auto model : {core::CameraModelType::PINHOLE, core::CameraModelType::FISHEYE, core::CameraModelType::EQUIRECTANGULAR, core::CameraModelType::THIN_PRISM_FISHEYE})
            for (bool distorted : {false, true})
                for (bool tile : {false, true}) {
                    SCOPED_TRACE("model=" + std::to_string(int(model)) + " distortion=" + std::to_string(distorted) + " tile=" + std::to_string(tile));
                    CameraCase camera{.model = model, .distorted = distorted, .background_image = true, .transform = true};
                    auto actual = render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB, false, tile, true, 3, true, camera);
                    for (auto value : actual.image)
                        ASSERT_TRUE(std::isfinite(value));
                    if (core::gpu_backend_available(GpuBackend::CUDA))
                        compare(actual, render(GpuBackend::CUDA, ops::GsplatRenderMode::RGB, false, tile, true, 3, true, camera));
                }
    }
    TEST(VulkanGsplat, PackedShTailsAndSaturation) {
        if (!core::gpu_backend_available(GpuBackend::Vulkan) || !core::gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        for (auto storage : {ops::ShStorage::Float32, ops::ShStorage::IeeeFloat16, ops::ShStorage::Q16})
            for (size_t n : {31u, 32u, 33u, 255u, 256u, 257u}) {
                SCOPED_TRACE("storage=" + std::to_string(int(storage)) + " count=" + std::to_string(n));
                CameraCase camera{.storage = storage, .count = n, .saturated = n == 33};
                compare(render(GpuBackend::Vulkan, ops::GsplatRenderMode::RGB, false, true, true, 3, true, camera), render(GpuBackend::CUDA, ops::GsplatRenderMode::RGB, false, true, true, 3, true, camera));
            }
    }
    TEST(VulkanGsplat, DISABLED_MillionRowTiming) {
        constexpr size_t n = 1'000'000, w = 256, h = 256;
        for (auto backend : {GpuBackend::CUDA, GpuBackend::Vulkan}) {
            ASSERT_TRUE(core::gpu_backend_available(backend));
            core::GpuBackendScope scope(backend);
            const auto* table = training::training_ops(backend).gsplat;
            ASSERT_NE(table, nullptr);
            std::vector<float> positions(n * 3), quaternions(n * 4, 0.f);
            for (size_t i = 0; i < n; ++i) {
                positions[i * 3] = (float(i % 1000) / 999.f - .5f) * 8.f;
                positions[i * 3 + 1] = (float(i / 1000) / 999.f - .5f) * 8.f;
                positions[i * 3 + 2] = 3.f + .2f * float(i % 37) / 36.f;
                quaternions[i * 4] = 1.f;
            }
            auto means = Tensor::from_vector(positions, {n, 3}, Device::GPU), quats = Tensor::from_vector(quaternions, {n, 4}, Device::GPU);
            auto scales = Tensor::full({n, 3}, -4.f, Device::GPU), opacity = Tensor::full({n, 1}, .2f, Device::GPU), dc = Tensor::full({n, 1, 3}, .1f, Device::GPU);
            auto rest = Tensor::zeros({384}, Device::GPU), view = Tensor::eye(4, Device::GPU), bg = Tensor::full({3}, .1f, Device::GPU);
            Tensor empty, image, alpha, depth, normal;
            ops::SplatInputs inputs{means, scales, quats, opacity, dc, rest, empty};
            ops::GsplatParams p{.full_image = {int(h), int(w)}, .intrinsics = {128.f, 128.f, 128.f, 128.f}};
            auto gi = Tensor::full({3, h, w}, .001f, Device::GPU), ga = Tensor::full({1, h, w}, .002f, Device::GPU);
            std::array<Tensor, 6> grads{Tensor::zeros_like(means), Tensor::zeros_like(scales), Tensor::zeros_like(quats), Tensor::zeros_like(opacity), Tensor::zeros_like(dc), Tensor::zeros_like(rest)};
            ops::GsplatGradients outputs{&grads, [](void* owner, ops::AdamSlot slot) -> Tensor& { return (*static_cast<std::array<Tensor, 6>*>(owner))[size_t(slot)]; }};
            ops::GsplatSaved saved{table->create()};
            double forward = 0, backward = 0, wall = 0;
            for (int iteration = 0; iteration < 4; ++iteration) {
                for (auto& g : grads)
                    g.zero_();
                core::GpuElapsed timer(backend, 3);
                ASSERT_TRUE(timer.ready());
                auto target = core::TensorExecutionTarget::current();
                ASSERT_TRUE(timer.wait_queue(target));
                auto start = std::chrono::steady_clock::now();
                ASSERT_TRUE(timer.mark(0, target));
                ASSERT_EQ(table->forward(saved, inputs, view, empty, empty, bg, empty, p, {image, alpha, depth, normal}).code, ops::RasterResult::Code::Success);
                ASSERT_TRUE(timer.mark(1, target));
                table->backward(saved, gi, ga, outputs, empty, empty, empty, empty, empty);
                ASSERT_TRUE(timer.mark(2, target));
                ASSERT_TRUE(timer.wait_event(2));
                if (iteration) {
                    forward += timer.milliseconds(0, 1).value();
                    backward += timer.milliseconds(1, 2).value();
                    wall += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                }
            }
            std::cout << "Gsplat 1M " << (backend == GpuBackend::CUDA ? "CUDA" : "Vulkan") << " forward_gpu_ms=" << forward / 3
                      << " backward_gpu_ms=" << backward / 3 << " frame_wall_ms=" << wall / 3 << std::endl;
            table->release(saved);
        }
    }
} // namespace
