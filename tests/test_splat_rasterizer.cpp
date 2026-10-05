/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Every blend variant and the present pass load and run on each tensor
// backend, and source-sorted frames reproduce the full-sort image exactly.
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "splat_rasterizer.hpp"

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::GpuBackendScope;
    using lfs::core::Tensor;
    using lfs::rendering::SplatPresentParameters;
    using lfs::rendering::SplatRasterizer;
    using lfs::rendering::SplatRasterMode;
    using lfs::rendering::SplatRasterParameters;

    constexpr uint32_t kWidth = 160, kHeight = 96;

    struct Splat {
        std::array<float, 4> mean_depth, conic_opacity, color;
        std::array<uint32_t, 4> bounds;
    };
    static_assert(sizeof(Splat) == 64);

    Tensor upload(const void* data, size_t bytes) {
        return Tensor::from_blob(const_cast<void*>(data), {bytes}, Device::CPU, DataType::UInt8).to(Device::GPU);
    }
    template <class T>
    std::vector<T> download(const Tensor& tensor, size_t count) {
        auto host = tensor.to(Device::CPU);
        std::vector<T> result(count);
        std::memcpy(result.data(), host.data_ptr(), count * sizeof(T));
        return result;
    }

    // Opaque-ish Gaussians, enough of them for source sorting.
    std::vector<Splat> make_splats(uint32_t count) {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> x(0, kWidth), y(0, kHeight), depth(1, 20), tint(.1f, .9f);
        std::vector<Splat> splats(count);
        for (auto& s : splats) {
            const float cx = x(rng), cy = y(rng), d = depth(rng);
            s.mean_depth = {cx, cy, d, 12};
            s.conic_opacity = {.04f, 0, .04f, .7f};
            s.color = {tint(rng), tint(rng), tint(rng), d * d};
            const auto lo = [](float v) { return uint32_t(std::max(0.f, v - 12)); };
            s.bounds = {lo(cx), lo(cy), uint32_t(cx + 12), uint32_t(cy + 12)};
        }
        return splats;
    }

    SplatRasterParameters make_parameters(uint32_t count, SplatRasterMode mode, uint32_t flags) {
        SplatRasterParameters r;
        r.count = count;
        r.width = kWidth;
        r.height = kHeight;
        r.columns = (kWidth + 15) / 16;
        r.tiles = r.columns * ((kHeight + 15) / 16);
        r.capacity = 1'000'000;
        r.mode = uint32_t(mode);
        r.flags = flags;
        r.background = {.1f, .2f, .3f, (flags & 16384u) ? 0.f : 1.f};
        r.intrinsics = {100, 100, kWidth / 2.f, kHeight / 2.f};
        r.clip = {.01f, 1000, 1, .3f};
        r.camera = {kWidth, kHeight, 0, 0};
        return r;
    }

    class SplatRasterizing : public testing::TestWithParam<GpuBackend> {};

    TEST_P(SplatRasterizing, EveryVariantRunsAndPresents) {
        if (!lfs::core::gpu_backend_available(GetParam()) || GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        const auto splats = make_splats(600);
        const auto projected = upload(splats.data(), splats.size() * sizeof(Splat));
        // Zero 3DGUT geometry: culled by the blend's support tests, still compiled and dispatched.
        const auto gut = Tensor::zeros({splats.size() * 64}, Device::GPU, DataType::UInt8);
        struct Variant {
            SplatRasterMode mode;
            uint32_t flags;
        };
        for (const auto [mode, flags] : {Variant{SplatRasterMode::Gaussian, 128}, Variant{SplatRasterMode::Gaussian, 128 | 4096},
                                         Variant{SplatRasterMode::Gaussian, 4096}, Variant{SplatRasterMode::Gaussian, 128 | 2048},
                                         Variant{SplatRasterMode::Gaussian, 128 | 16384}, Variant{SplatRasterMode::Points, 4096},
                                         Variant{SplatRasterMode::Discs, 4096}, Variant{SplatRasterMode::Gut, 128 | 4096},
                                         Variant{SplatRasterMode::Gut, 4096}}) {
            SplatRasterizer rasterizer(GetParam());
            ASSERT_TRUE(rasterizer.reserve(uint32_t(splats.size()), kWidth, kHeight, 1'000'000));
            auto rasterized = rasterizer.rasterize(projected, mode == SplatRasterMode::Gut ? &gut : nullptr, uint32_t(splats.size()), mode,
                                                   make_parameters(uint32_t(splats.size()), mode, flags));
            ASSERT_TRUE(rasterized) << rasterized.error().detail() << " mode=" << uint32_t(mode) << " flags=" << flags;
            auto rgba = Tensor::empty({kHeight, kWidth, 4}, Device::GPU, DataType::UInt8);
            auto depth = Tensor::empty({kHeight, kWidth}, Device::GPU, DataType::Float32);
            SplatPresentParameters present;
            present.transparent = (flags & 16384u) ? 1u : 0u;
            present.background = {.1f, .2f, .3f, 1};
            present.extent = {kWidth, kHeight, kWidth, kHeight};
            auto shown = rasterizer.present(present, rgba, depth);
            ASSERT_TRUE(shown) << shown.error().detail();
            EXPECT_EQ(download<uint32_t>(rasterizer.status(), 6)[2], 0u) << "mode=" << uint32_t(mode) << " flags=" << flags;
            if (mode == SplatRasterMode::Gaussian && !present.transparent) {
                const auto pixels = download<uint8_t>(rgba, size_t(kWidth) * kHeight * 4);
                size_t opaque = 0;
                for (size_t i = 3; i < pixels.size(); i += 4)
                    opaque += pixels[i] == 255;
                EXPECT_EQ(opaque, size_t(kWidth) * kHeight) << "flags=" << flags;
            }
        }
    }

    // A completed dense frame of the same source count switches to source
    // sorting; the image must not change.
    TEST_P(SplatRasterizing, SourceSortedFrameMatchesFirstFrame) {
        if (!lfs::core::gpu_backend_available(GetParam()) || GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        const auto splats = make_splats(6000);
        const auto count = uint32_t(splats.size());
        const auto projected = upload(splats.data(), splats.size() * sizeof(Splat));
        SplatRasterizer rasterizer(GetParam());
        ASSERT_TRUE(rasterizer.reserve(count, kWidth, kHeight, 1'000'000));
        const auto parameters = make_parameters(count, SplatRasterMode::Gaussian, 128 | 4096);
        std::vector<std::vector<uint16_t>> frames;
        for (int frame = 0; frame < 3; ++frame) {
            auto rasterized = rasterizer.rasterize(projected, nullptr, count, SplatRasterMode::Gaussian, parameters);
            ASSERT_TRUE(rasterized) << rasterized.error().detail();
            frames.push_back(download<uint16_t>(rasterizer.color(), size_t(kWidth) * kHeight * 4));
        }
        EXPECT_EQ(frames[0], frames[2]);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SplatRasterizing, testing::ValuesIn(lfs::core::kCompiledGpuBackends),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });
} // namespace
