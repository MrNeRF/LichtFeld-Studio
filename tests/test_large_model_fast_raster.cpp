/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The rasterizers' forward passes (FastGS, and gsplat for --gut training) read each visible splat's SH rest from the
// training store: Q16 cells through a 64-bit start per Gaussian, fp32 and f16 float4 slots through a float index that
// passes 2^32 at row 89,478,464.
// A model of 95,444,224 Gaussians at SH degree 3 renders a 64x64 frame in which only six splats are visible: two
// below every wrap, two past the float4 wrap and two past the Q16 wrap; every other Gaussian sits behind the camera.
// A model of the compact blocks (large_model_test_support.hpp), with the same six splats and the same SH bytes, is
// the reference: the frames must match bit for bit. An SH index computed in 32 bits reads another row's
// coefficients for the far splats and changes their colour.
//
// Tiers by device memory: see large_model_test_support.hpp. Each case's memory is in its skip message.

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "lfs/training/ops/gsplat.hpp"
#include "lfs/training/ops/raster.hpp"
#include "lfs/training/ops/registry.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <tuple>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::ShFormat;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace ops = lfs::gpu_ops;

    constexpr int kSize = 64;
    constexpr float kFocal = 57.6f, kCenter = 32.0f, kDepth = 4.0f;

    // Where each visible splat lands in the frame, and its row in the big model.
    struct Visible {
        size_t row;
        float u, v;
    };
    constexpr std::array<Visible, 6> kVisible{{{5, 12, 14},
                                               {89'478'420, 32, 14},
                                               {89'478'470, 52, 14},
                                               {89'478'500, 12, 44},
                                               {95'443'720, 32, 44},
                                               {95'444'100, 52, 44}}};
    static_assert(kVisible[2].row >= lm::kFloat4WrapRow && kVisible[4].row >= lm::kQ16WrapRow);

    // Small SH values a function of the row and cell, (value - 1024) / 8192: exact in fp32 and f16, colours away
    // from their clamps.
    float shading(const size_t row, const uint32_t cell) { return (lm::value(row, cell) - 1024.0f) / 8192.0f; }

    ops::ShStorage storage_of(const ShFormat format) {
        return format == ShFormat::Q16       ? ops::ShStorage::Q16
               : format == ShFormat::Float16 ? ops::ShStorage::IeeeFloat16
                                             : ops::ShStorage::Float32;
    }

    // Parameters, the forward's per-Gaussian buffers (about 24 values) and the screen share: 156 bytes a Gaussian.
    size_t frame_bytes(const ShFormat format, const size_t rows) {
        return rows * 156 + lm::store_bytes(format, rows);
    }

    Tensor host_tensor(const std::vector<float>& values, const lfs::core::TensorShape& shape) {
        Tensor host = Tensor::empty(shape, Device::CPU, DataType::Float32);
        std::memcpy(host.data_ptr(), values.data(), values.size() * sizeof(float));
        return host;
    }

    struct Rendered {
        std::vector<float> image, alpha, depth;
    };

    enum class Rasterizer { Fast,
                            Gsplat };

    class LargeModelFastRaster : public ::testing::TestWithParam<std::tuple<GpuBackend, Rasterizer, ShFormat>> {
    protected:
        void SetUp() override {
            const auto [backend, rasterizer, format] = GetParam();
            if (const auto reason = lm::skip_reason(frame_bytes(format, lm::kRows) + frame_bytes(format, lm::kCompactRows)))
                GTEST_SKIP() << *reason;
            if (!lfs::core::gpu_backend_available(backend))
                GTEST_SKIP() << lfs::core::gpu_backend_name(backend) << " device unavailable";
            const auto& table = lfs::training::training_ops(backend);
            if (rasterizer == Rasterizer::Fast ? table.fast == nullptr : table.gsplat == nullptr)
                GTEST_SKIP() << "the rasterizer's slot is empty";
            session_.emplace(backend);
            ASSERT_TRUE(session_->switched());
        }

        void TearDown() override { session_.reset(); }

        // Renders a model of `rows` Gaussians whose SH rest is `sh` and whose visible splats are the rows in
        // `visible_rows`; every other Gaussian sits behind the camera.
        static Rendered render(const size_t rows, const lm::Store& sh, const std::array<size_t, 6>& visible_rows) {
            [[maybe_unused]] const auto [backend, rasterizer, format] = GetParam();
            Tensor means = Tensor::full({rows, 3}, -10.0f, Device::GPU);
            Tensor scales = Tensor::full({rows, 3}, -2.5f, Device::GPU);
            Tensor rotations = Tensor::full({rows, 4}, 0.5f, Device::GPU);
            Tensor opacities = Tensor::full({rows, 1}, 2.0f, Device::GPU);
            Tensor sh0 = Tensor::full({rows, 1, 3}, 0.1f, Device::GPU);
            for (size_t k = 0; k < kVisible.size(); ++k) {
                const size_t m = visible_rows[k];
                const auto f = static_cast<float>(k);
                means.slice(0, m, m + 1).copy_(host_tensor({(kVisible[k].u - kCenter) * kDepth / kFocal,
                                                            (kVisible[k].v - kCenter) * kDepth / kFocal, 0.01f * f},
                                                           {1, 3}).gpu());
                rotations.slice(0, m, m + 1).copy_(host_tensor({1.0f, 0.0f, 0.0f, 0.0f}, {1, 4}).gpu());
                sh0.slice(0, m, m + 1).copy_(host_tensor({0.1f * f, -0.05f * f, 0.02f * f}, {1, 1, 3}).gpu());
            }
            const std::vector<float> view_values{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, kDepth, 0, 0, 0, 1};
            const Tensor camera = host_tensor({0, 0, -kDepth}, {3}).gpu();
            const Tensor background = host_tensor({0, 0, 0}, {3}).gpu();
            const Tensor none;
            const ops::ShParams sh_params{.storage = storage_of(sh.format), .active_bases = 16, .layout_bases = 16};
            const ops::SplatInputs splats{.means = means, .raw_scales = scales, .raw_rotations = rotations,
                                          .raw_opacities = opacities, .sh0 = sh0, .shN = sh.values,
                                          .sh_value_bounds = sh.format == ShFormat::Q16 ? sh.bounds : none};
            Tensor image, alpha, depth, normal;
            ops::RasterResult result;
            if (rasterizer == Rasterizer::Fast) {
                const ops::FastRasterOps& table = *lfs::training::training_ops(backend).fast;
                const ops::FastParams params{
                    .full_image = {kSize, kSize},
                    .intrinsics = {kFocal, kFocal, kCenter, kCenter},
                    .sh = sh_params,
                    .render_depth = true,
                };
                Tensor share = Tensor::zeros({rows}, Device::GPU);
                ops::FastSaved saved;
                saved.backend = table.create();
                result = table.forward(saved, splats, host_tensor(view_values, {1, 4, 4}).gpu(), camera, background,
                                       none, params, {.image = image, .alpha = alpha, .depth = depth, .normal = normal},
                                       share);
                EXPECT_EQ(result.code, ops::RasterResult::Code::Success) << result.message;
                Rendered out;
                if (result.code == ops::RasterResult::Code::Success)
                    out = {lm::host_copy<float>(image), lm::host_copy<float>(alpha), lm::host_copy<float>(depth)};
                table.release(saved);
                return out;
            }
            const ops::GsplatRasterOps& table = *lfs::training::training_ops(backend).gsplat;
            const ops::GsplatParams params{
                .full_image = {kSize, kSize},
                .intrinsics = {kFocal, kFocal, kCenter, kCenter},
                .sh = sh_params,
                .render_mode = ops::GsplatRenderMode::RGB_D,
            };
            ops::GsplatSaved saved;
            saved.backend = table.create();
            result = table.forward(saved, splats, host_tensor(view_values, {4, 4}).gpu(), none, none, background, none,
                                   params, {.image = image, .alpha = alpha, .depth = depth, .normal = normal});
            EXPECT_EQ(result.code, ops::RasterResult::Code::Success) << result.message;
            Rendered out;
            if (result.code == ops::RasterResult::Code::Success)
                out = {lm::host_copy<float>(image), lm::host_copy<float>(alpha),
                       depth.is_valid() ? lm::host_copy<float>(depth) : std::vector<float>{}};
            table.release(saved);
            return out;
        }

        static void expect_same(const std::vector<float>& actual, const std::vector<float>& expected,
                                const std::string& what) {
            ASSERT_EQ(actual.size(), expected.size()) << what;
            size_t wrong = 0, first = 0;
            for (size_t i = 0; i < actual.size(); ++i)
                if (std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0 && wrong++ == 0)
                    first = i;
            EXPECT_EQ(wrong, 0u) << what << ": " << wrong << " of " << actual.size() << " values differ; first at "
                                 << first << ": " << actual[first] << " instead of " << expected[first];
        }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(LargeModelFastRaster, ForwardReadsShPastTheWrap) {
        const ShFormat format = std::get<2>(GetParam());
        const lm::Store compact = lm::compact_blocks(format, lm::kRows, shading);
        std::array<size_t, 6> big_rows{}, compact_rows{};
        for (size_t k = 0; k < kVisible.size(); ++k) {
            big_rows[k] = kVisible[k].row;
            compact_rows[k] = lm::compact_row(kVisible[k].row);
        }

        const lm::Store small{.format = format, .rows = lm::kCompactRows, .values = compact.values.gpu(),
                              .bounds = format == ShFormat::Q16 ? compact.bounds.gpu() : Tensor{}};
        const Rendered reference = render(lm::kCompactRows, small, compact_rows);
        ASSERT_EQ(reference.alpha.size(), size_t{kSize} * kSize) << "the reference frame did not render";
        // Not a vacuous frame: every visible splat covers its pixel.
        for (const auto& visible : kVisible)
            EXPECT_GT(reference.alpha[static_cast<size_t>(visible.v) * kSize + static_cast<size_t>(visible.u)], 0.5f)
                << "the splat of row " << visible.row << " does not cover its pixel";

        auto store = lm::zero_store(format, lm::kRows, Device::GPU);
        if (!store)
            GTEST_SKIP() << "this device can't allocate or clear " << lm::kRows << " rows of " << lm::format_name(format);
        lm::place_blocks(*store, compact);
        const Rendered actual = render(lm::kRows, *store, big_rows);
        const std::string what = std::string(lm::format_name(format)) + " SH, " + std::to_string(lm::kRows) + " Gaussians";
        expect_same(actual.image, reference.image, what + ", image");
        expect_same(actual.alpha, reference.alpha, what + ", alpha");
        expect_same(actual.depth, reference.depth, what + ", depth");
    }

    // gsplat takes fp32 or Q16 SH rest.
    std::vector<std::tuple<GpuBackend, Rasterizer, ShFormat>> frame_cases() {
        std::vector<std::tuple<GpuBackend, Rasterizer, ShFormat>> cases;
        for (const GpuBackend backend : {GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan}) {
            for (const ShFormat format : {ShFormat::Q16, ShFormat::Float16, ShFormat::Float32})
                cases.emplace_back(backend, Rasterizer::Fast, format);
            for (const ShFormat format : {ShFormat::Q16, ShFormat::Float32})
                cases.emplace_back(backend, Rasterizer::Gsplat, format);
        }
        return cases;
    }

    INSTANTIATE_TEST_SUITE_P(Backends, LargeModelFastRaster, testing::ValuesIn(frame_cases()), [](const auto& info) {
        return std::string(lfs::core::gpu_backend_name(std::get<0>(info.param))) +
               (std::get<1>(info.param) == Rasterizer::Fast ? "_Fast_" : "_Gsplat_") +
               lm::format_name(std::get<2>(info.param));
    });

} // namespace
