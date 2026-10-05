/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_vignette.hpp"
#include "program_contract.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs::core;
    using M = GpuKernelModule;
    struct Params {
        uint64_t first = 0, second = 0;
        uint32_t count = 0;
        float scale = 0, bias = 0;
        uint32_t padding = 0;
    };
    class Programs : public testing::TestWithParam<GpuBackend> {};

    TEST_P(Programs, SameSlangComputeMatchesCpuAndTensorTimeline) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        auto module = std::move(*loaded);
        std::vector<float> values(1031);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = float(i) / 16 - 20;
        auto input = Tensor::from_blob(values.data(), {values.size()}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto output = Tensor::zeros({values.size()}, Device::GPU);
        Params params{.count = uint32_t(values.size()), .scale = 2, .bias = 3};
        const std::array bindings{M::Binding{0, &input}, M::Binding{8, &output, M::Access::ReadWrite}};
        auto dispatched = module->dispatch({.function = "transform", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}, .groups = {M::groups_for(values.size(), 64), 1, 1}});
        ASSERT_TRUE(dispatched) << dispatched.error().detail();
        // Immediate module destruction and tensor consumption must be safe.
        module.reset();
        auto host = output.to(Device::CPU);
        for (size_t i = 0; i < values.size(); ++i)
            EXPECT_FLOAT_EQ(host.ptr<float>()[i], values[i] * 2 + 3) << i;
    }

    TEST_P(Programs, DependentDispatchChainStaysOrdered) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        constexpr size_t kCount = 4099;
        constexpr int kSteps = 256;
        std::array buffers{Tensor::zeros({kCount}, Device::GPU), Tensor::zeros({kCount}, Device::GPU)};
        const Params params{.count = uint32_t(kCount), .scale = 1, .bias = 1};
        for (int step = 0; step < kSteps; ++step) {
            const std::array bindings{M::Binding{0, &buffers[step % 2]},
                                      M::Binding{8, &buffers[(step + 1) % 2], M::Access::ReadWrite}};
            auto dispatched = (*loaded)->dispatch({.function = "transform",
                                                   .arguments = {std::as_bytes(std::span(&params, 1)), bindings},
                                                   .groups = {M::groups_for(kCount, 64), 1, 1}});
            ASSERT_TRUE(dispatched) << step << ": " << dispatched.error().detail();
        }
        // Each step adds one to the previous step's output.
        auto host = buffers[kSteps % 2].to(Device::CPU);
        for (size_t i = 0; i < kCount; ++i)
            ASSERT_EQ(host.ptr<float>()[i], float(kSteps)) << i;
    }

    TEST_P(Programs, SameSlangRasterBlendDepthAndLoadMatchCpuPixels) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        auto module = std::move(*loaded);
        if (GetParam() == GpuBackend::CUDA) {
            auto result = module->draw({});
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Unsupported);
            return;
        }
        // Integer-aligned right triangles avoid ambiguous sample coverage.
        constexpr size_t width = 64, height = 48;
        std::array<std::array<float, 4>, 9> positions{};
        std::array<std::array<float, 4>, 9> colors{};
        const std::array<float, 3> triangle_depths{0.6f, 0.4f, 0.7f};
        const std::array<std::array<float, 4>, 3> triangle_colors{{{0.25f, 0.5f, 1, 0.25f}, {1, 0.5f, 0.25f, 0.5f}, {0, 1, 0, 1}}};
        for (size_t triangle = 0; triangle < 3; ++triangle) {
            positions[triangle * 3] = {-0.75f, 0.75f, triangle_depths[triangle], 1};
            positions[triangle * 3 + 1] = {0.75f, 0.75f, triangle_depths[triangle], 1};
            positions[triangle * 3 + 2] = {-0.75f, -0.75f, triangle_depths[triangle], 1};
            for (size_t vertex = 0; vertex < 3; ++vertex)
                colors[triangle * 3 + vertex] = triangle_colors[triangle];
        }
        auto vertices = Tensor::from_blob(positions.data(), {9, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto vertex_colors = Tensor::from_blob(colors.data(), {9, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto color = Tensor::zeros({height, width, 4}, Device::GPU);
        auto depth = Tensor::zeros({height, width}, Device::GPU);
        const Params params{};
        const std::array bindings{M::Binding{0, &vertices}, M::Binding{8, &vertex_colors}};
        M::Draw draw{.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}, .color = &color, .depth = &depth, .vertex_count = 9, .blend = M::Blend::StraightAlpha, .clear_color = true, .color_clear = {0.125f, 0.25f, 0.5f, 1}, .clear_depth = true};
        auto result = module->draw(draw);
        ASSERT_TRUE(result) << result.error().detail();
        // Same depth must fail LESS; this also tests preserving attachments.
        draw.clear_color = draw.clear_depth = false;
        result = module->draw(draw);
        ASSERT_TRUE(result) << result.error().detail();
        module.reset();
        auto pixels = color.to(Device::CPU), depths = depth.to(Device::CPU);
        size_t covered = 0;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const float a = (float(x) + 0.5f - 8) / 48;
                const float b = (float(y) + 0.5f - 6) / 36;
                const bool inside = a >= 0 && b >= 0 && a + b < 1;
                covered += inside;
                const std::array<float, 4> expected = inside ? std::array{0.578125f, 0.40625f, 0.4375f, 1.0f} : draw.color_clear;
                for (size_t c = 0; c < 4; ++c)
                    EXPECT_NEAR(pixels.ptr<float>()[(y * width + x) * 4 + c], expected[c], 2e-6f) << x << ',' << y << ',' << c;
                EXPECT_NEAR(depths.ptr<float>()[y * width + x], inside ? 0.4f : 1.0f, 2e-6f) << x << ',' << y;
            }
        }
        EXPECT_GT(covered, 800u);
    }

    TEST_P(Programs, Rgba8RasterWithoutDepthMatchesCpuPixels) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        if (!(*loaded)->supports_raster())
            GTEST_SKIP() << "Backend is compute-only";
        constexpr size_t width = 79, height = 53;
        std::array<std::array<float, 4>, 6> positions{{{-1, 1, 0.5f, 1}, {1, 1, 0.5f, 1}, {-1, -1, 0.5f, 1}, {-1, -1, 0.5f, 1}, {1, 1, 0.5f, 1}, {1, -1, 0.5f, 1}}};
        std::array<std::array<float, 4>, 6> colors{};
        colors.fill({0.25f, 0.5f, 0.75f, 1});
        auto vertices = Tensor::from_blob(positions.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto vertex_colors = Tensor::from_blob(colors.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto color = Tensor::zeros({height, width, 4}, Device::GPU, DataType::UInt8);
        const Params params{};
        const std::array bindings{M::Binding{0, &vertices}, M::Binding{8, &vertex_colors}};
        constexpr M::Scissor scissor{7, 5, 61, 37};
        auto result = (*loaded)->draw({.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}, .color = &color, .vertex_count = 6, .scissor = scissor, .clear_color = true});
        ASSERT_TRUE(result) << result.error().detail();
        loaded->reset();
        const auto pixels = color.to(Device::CPU);
        const std::array<int, 4> expected{64, 128, 191, 255};
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                const bool inside = x >= scissor.x && x < scissor.x + scissor.width &&
                                    y >= scissor.y && y < scissor.y + scissor.height;
                for (size_t channel = 0; channel < 4; ++channel)
                    EXPECT_NEAR(int(pixels.ptr<uint8_t>()[(y * width + x) * 4 + channel]),
                                inside ? expected[channel] : 0, 1)
                        << x << ',' << y << ',' << channel;
            }
    }

    TEST_P(Programs, BatchedDrawsMatchSequentialDraws) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        if (!(*loaded)->supports_raster())
            GTEST_SKIP() << "Backend is compute-only";
        constexpr size_t width = 67, height = 41;
        std::array<std::array<float, 4>, 6> positions{{{-1, 1, 0.5f, 1}, {1, 1, 0.5f, 1}, {-1, -1, 0.5f, 1}, {-1, -1, 0.5f, 1}, {1, 1, 0.5f, 1}, {1, -1, 0.5f, 1}}};
        std::array<std::array<float, 4>, 6> first_colors{}, second_colors{};
        first_colors.fill({0.25f, 0.5f, 0.75f, 0.5f});
        second_colors.fill({1, 0.25f, 0, 0.75f});
        auto vertices = Tensor::from_blob(positions.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto first_tensor = Tensor::from_blob(first_colors.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto second_tensor = Tensor::from_blob(second_colors.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        const Params params{};
        const std::array first_bindings{M::Binding{0, &vertices}, M::Binding{8, &first_tensor}};
        const std::array second_bindings{M::Binding{0, &vertices}, M::Binding{8, &second_tensor}};
        const auto draws = [&](Tensor& target) {
            return std::array{
                M::Draw{.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), first_bindings}, .color = &target, .vertex_count = 6, .blend = M::Blend::StraightAlpha, .clear_color = true, .color_clear = {0, 0, 0.25f, 1}},
                M::Draw{.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), second_bindings}, .color = &target, .vertex_count = 6, .scissor = M::Scissor{5, 3, 31, 17}, .blend = M::Blend::StraightAlpha},
                M::Draw{.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), first_bindings}, .color = &target, .vertex_count = 6, .scissor = M::Scissor{20, 10, 40, 25}, .blend = M::Blend::PremultipliedAlpha},
                M::Draw{.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), second_bindings}, .color = &target, .vertex_count = 6, .scissor = M::Scissor{0, 0, 0, 0}},
            };
        };
        auto batched = Tensor::zeros({height, width, 4}, Device::GPU, DataType::UInt8);
        auto sequential = Tensor::zeros({height, width, 4}, Device::GPU, DataType::UInt8);
        const auto batch = draws(batched);
        auto result = (*loaded)->draw_batch(batch);
        ASSERT_TRUE(result) << result.error().detail();
        for (const auto& draw : draws(sequential)) {
            result = (*loaded)->draw(draw);
            ASSERT_TRUE(result) << result.error().detail();
        }
        // Clearing after the first draw is a contract violation, not a reorder.
        auto invalid = draws(batched);
        invalid[1].clear_color = true;
        result = (*loaded)->draw_batch(invalid);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
        loaded->reset();
        const auto a = batched.to(Device::CPU), b = sequential.to(Device::CPU);
        size_t differing = 0;
        for (size_t i = 0; i < a.numel(); ++i)
            differing += a.ptr<uint8_t>()[i] != b.ptr<uint8_t>()[i];
        EXPECT_EQ(differing, 0u);
        // The second draw's scissor region blends over the first.
        const auto pixel = [&](size_t x, size_t y, size_t c) { return int(a.ptr<uint8_t>()[(y * width + x) * 4 + c]); };
        EXPECT_NE(pixel(10, 8, 0), pixel(1, 1, 0));
        EXPECT_EQ(pixel(1, 1, 0), pixel(width - 1, 1, 0));
    }

    TEST_P(Programs, ViewportMapsNdcToItsPixelRect) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        if (!(*loaded)->supports_raster())
            GTEST_SKIP() << "Backend is compute-only";
        constexpr size_t width = 64, height = 48;
        // The upper-left NDC quadrant must land in the viewport's top-left quarter.
        std::array<std::array<float, 4>, 6> positions{{{-1, 1, 0.5f, 1}, {0, 1, 0.5f, 1}, {-1, 0, 0.5f, 1}, {-1, 0, 0.5f, 1}, {0, 1, 0.5f, 1}, {0, 0, 0.5f, 1}}};
        std::array<std::array<float, 4>, 6> colors{};
        colors.fill({1, 1, 1, 1});
        auto vertices = Tensor::from_blob(positions.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto vertex_colors = Tensor::from_blob(colors.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto color = Tensor::zeros({height, width, 4}, Device::GPU, DataType::UInt8);
        const Params params{};
        const std::array bindings{M::Binding{0, &vertices}, M::Binding{8, &vertex_colors}};
        constexpr M::Viewport viewport{8, 4, 40, 32};
        auto result = (*loaded)->draw({.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}, .color = &color, .vertex_count = 6, .viewport = viewport, .clear_color = true});
        ASSERT_TRUE(result) << result.error().detail();
        auto invalid = (*loaded)->draw({.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}, .color = &color, .vertex_count = 6, .viewport = M::Viewport{0, 0, 0, 10}});
        ASSERT_FALSE(invalid);
        EXPECT_EQ(invalid.error().code(), lfs::ErrorCode::InvalidArgument);
        loaded->reset();
        const auto pixels = color.to(Device::CPU);
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                const bool inside = x >= 8 && x < 28 && y >= 4 && y < 20;
                EXPECT_EQ(int(pixels.ptr<uint8_t>()[(y * width + x) * 4 + 3]), inside ? 255 : 0) << x << ',' << y;
            }
    }

    TEST_P(Programs, CullingTreatsCounterClockwiseNdcAsFront) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        if (!(*loaded)->supports_raster())
            GTEST_SKIP() << "Backend is compute-only";
        constexpr size_t width = 32, height = 32;
        // Left triangle counter-clockwise, right triangle clockwise (NDC +Y up).
        std::array<std::array<float, 4>, 6> positions{{{-0.9f, -0.5f, 0.5f, 1}, {-0.1f, -0.5f, 0.5f, 1}, {-0.5f, 0.5f, 0.5f, 1},
                                                       {0.1f, -0.5f, 0.5f, 1}, {0.5f, 0.5f, 0.5f, 1}, {0.9f, -0.5f, 0.5f, 1}}};
        std::array<std::array<float, 4>, 6> colors{};
        colors.fill({1, 1, 1, 1});
        auto vertices = Tensor::from_blob(positions.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto vertex_colors = Tensor::from_blob(colors.data(), {6, 4}, Device::CPU, DataType::Float32).to(Device::GPU);
        const Params params{};
        const std::array bindings{M::Binding{0, &vertices}, M::Binding{8, &vertex_colors}};
        const auto coverage = [&](M::Cull cull) {
            auto color = Tensor::zeros({height, width, 4}, Device::GPU, DataType::UInt8);
            auto result = (*loaded)->draw({.vertex = "vertexMain", .fragment = "fragmentMain", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}, .color = &color, .vertex_count = 6, .cull = cull, .clear_color = true});
            EXPECT_TRUE(result) << result.error().detail();
            const auto pixels = color.to(Device::CPU);
            // Sample the centroid row of each triangle: left half and right half.
            const auto alpha = [&](size_t x) { return int(pixels.ptr<uint8_t>()[(18 * width + x) * 4 + 3]); };
            return std::pair{alpha(8) > 0, alpha(24) > 0};
        };
        EXPECT_EQ(coverage(M::Cull::None), std::pair(true, true));
        EXPECT_EQ(coverage(M::Cull::Back), std::pair(true, false));
        EXPECT_EQ(coverage(M::Cull::Front), std::pair(false, true));
        loaded->reset();
    }

    TEST_P(Programs, InvalidBindingsReturnTypedErrors) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        auto loaded = M::load(program_contract_entries(), GetParam());
        ASSERT_TRUE(loaded) << loaded.error().detail();
        const Params params{};
        const std::array bindings{M::Binding{1, nullptr}};
        auto result = (*loaded)->dispatch({.function = "transform", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}});
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
        auto wrong_group = (*loaded)->dispatch({.function = "transform", .group = {32, 1, 1}});
        ASSERT_FALSE(wrong_group);
        EXPECT_EQ(wrong_group.error().code(), lfs::ErrorCode::InvalidArgument);
        auto missing_tensors = (*loaded)->dispatch({.function = "transform", .arguments = {std::as_bytes(std::span(&params, 1)), {}}});
        ASSERT_FALSE(missing_tensors);
        EXPECT_EQ(missing_tensors.error().code(), lfs::ErrorCode::InvalidArgument);
    }

    TEST_P(Programs, VignetteMatchesReferenceFormula) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP();
        GpuBackendScope scope(GetParam());
        for (const auto size : {std::array{79u, 53u}, std::array{128u, 96u}}) {
            constexpr float intensity = 0.35f, radius = 0.75f, softness = 0.5f;
            auto result = vignette_image(size[0], size[1], intensity, radius, softness);
            ASSERT_TRUE(result) << result.error().detail();
            auto host = result->to(Device::CPU);
            const float fade = (1 - radius) * 0.5f * std::min(size[0], size[1]);
            for (unsigned y = 0; y < size[1]; ++y)
                for (unsigned x = 0; x < size[0]; ++x) {
                    const float dx = std::max(std::abs(float(x) + 0.5f - size[0] * 0.5f) - (size[0] * 0.5f - fade), 0.0f);
                    const float dy = std::max(std::abs(float(y) + 0.5f - size[1] * 0.5f) - (size[1] * 0.5f - fade), 0.0f);
                    const float visible = std::clamp(1 - std::sqrt(dx * dx + dy * dy) / fade, 0.0f, 1.0f);
                    const float smoothed = visible * visible * (3 - 2 * visible);
                    const float alpha = intensity * (1 - (visible * (1 - softness) + smoothed * softness));
                    const auto index = (y * size[0] + x) * 4;
                    for (size_t c = 0; c < 3; ++c)
                        EXPECT_EQ(host.ptr<float>()[index + c], 0) << x << ',' << y;
                    EXPECT_NEAR(host.ptr<float>()[index + 3], alpha, 2e-6f) << x << ',' << y;
                }
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, Programs, testing::ValuesIn(kCompiledGpuBackends),
                             [](const auto& info) { return gpu_backend_name(info.param); });
} // namespace
