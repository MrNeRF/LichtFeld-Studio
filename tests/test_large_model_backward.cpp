/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The rasterizers' backward passes with a few visible splats far into a large model. FastGS keeps 12 helper
// gradient floats a Gaussian and gsplat 14; a row start computed in 32 bits (primitive x 12, g x 14) wraps at
// 357,913,942 and 306,783,379 Gaussians, and the splat there then adds its gradients into rows 0 and 1 and reads them
// back from there. On Vulkan on NVIDIA a buffer past 4 GiB has the same problem at a smaller N, by the record size.
//
// One body serves every case: the model's size comes from the environment (LFS_TEST_LARGE_FAST_ROWS and
// LFS_TEST_LARGE_GSPLAT_ROWS, by default just past Metal's wraps above), and the visible splats are chosen from it:
// rows 0, 1 and 2, the first two rows past each known wrap point below the size (the two above, and the first records
// past 4 GiB at 72, 64, 56 and 16 bytes a Gaussian), the last two rows, and any rows listed in
// LFS_TEST_LARGE_VISIBLE_ROWS (comma separated). Every other Gaussian sits behind the camera. The forward's frame and
// one backward's raw gradients are compared with the same visible splats alone, in a small model: the frames bit for
// bit, the gradients within the blend backward's atomic reordering. A wrong record address draws a far splat from
// another record; a wrong row start mixes the gradients of the splats it aliases; sampled invisible rows must get no
// gradient.
//
// Memory: 280 bytes a Gaussian by the test's own count. At the default sizes that is 100 GB (FastGS) and 86 GB
// (gsplat), tier 2 (LFS_TEST_HUGE_SH=1; on a shared machine, in an exclusive hold of the GPU lock); at the 61M, 68M
// and 78M of Vulkan's 4 GiB points it is 17 to 22 GB, tier 1 (LFS_TEST_LARGE_SH=1).

#include "large_model_test_support.hpp"

#include "core/tensor_backend.hpp"
#include "lfs/training/ops/domain_types.hpp"
#include "lfs/training/ops/gsplat.hpp"
#include "lfs/training/ops/raster.hpp"
#include "lfs/training/ops/registry.hpp"

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace lm = lfs::test::large_model;
    namespace ops = lfs::gpu_ops;

    enum class Rasterizer { Fast,
                            Gsplat };

    constexpr int kSize = 96;
    constexpr float kFocal = 86.4f, kCenter = 48.0f, kDepth = 4.0f;

    // The first Gaussian whose record of `bytes` starts past 2^32 bytes, or whose row of `floats` floats passes 2^32
    // floats.
    constexpr size_t past_bytes(const size_t bytes) { return ((size_t{1} << 32) + bytes - 1) / bytes; }
    constexpr size_t past_floats(const size_t floats) { return ((size_t{1} << 32) + floats - 1) / floats; }
    static_assert(past_floats(12) == 357'913'942 && past_floats(14) == 306'783'379);

    size_t rows_from(const char* const variable, const size_t fallback) {
        if (const char* value = std::getenv(variable))
            return static_cast<size_t>(std::strtoull(value, nullptr, 10));
        return fallback;
    }

    std::vector<size_t> visible_rows(const size_t rows) {
        std::set<size_t> picked{0, 1, 2, rows - 2, rows - 1};
        for (const size_t row : {past_floats(12), past_floats(14), past_bytes(72), past_bytes(64), past_bytes(56),
                                 past_bytes(16)})
            if (row + 3 < rows) {
                picked.insert(row);
                picked.insert(row + 1);
            }
        if (const char* listed = std::getenv("LFS_TEST_LARGE_VISIBLE_ROWS")) {
            for (const char* p = listed; *p;) {
                char* end = nullptr;
                const size_t row = static_cast<size_t>(std::strtoull(p, &end, 10));
                if (end == p)
                    break;
                if (row < rows)
                    picked.insert(row);
                p = *end == ',' ? end + 1 : end;
            }
        }
        return {picked.begin(), picked.end()};
    }

    Tensor host_tensor(const std::vector<float>& values, const lfs::core::TensorShape& shape) {
        Tensor host = Tensor::empty(shape, Device::CPU, DataType::Float32);
        std::memcpy(host.data_ptr(), values.data(), values.size() * sizeof(float));
        return host;
    }

    // The parameters of a model of `rows` Gaussians at SH degree 0, all behind the camera but the splats at
    // `visible` (row k of the list at grid cell k of the frame).
    struct Model {
        Tensor means, scales, rotations, opacities, sh0;
    };

    Model make_model(const size_t rows, const std::vector<size_t>& visible) {
        Model m{Tensor::full({rows, 3}, -10.0f, Device::GPU), Tensor::full({rows, 3}, -2.3f, Device::GPU),
                Tensor::full({rows, 4}, 0.5f, Device::GPU), Tensor::full({rows, 1}, 1.0f, Device::GPU),
                Tensor::full({rows, 1, 3}, 0.1f, Device::GPU)};
        for (size_t k = 0; k < visible.size(); ++k) {
            const size_t row = visible[k];
            const float u = 10.0f + 19.0f * static_cast<float>(k % 5), v = 10.0f + 19.0f * static_cast<float>(k / 5);
            const auto f = static_cast<float>(k);
            m.means.slice(0, row, row + 1)
                .copy_(host_tensor({(u - kCenter) * kDepth / kFocal, (v - kCenter) * kDepth / kFocal, 0.02f * f}, {1, 3}).gpu());
            m.scales.slice(0, row, row + 1).copy_(host_tensor({-2.3f, -2.4f + 0.01f * f, -2.5f}, {1, 3}).gpu());
            const float angle = 0.1f * f;
            m.rotations.slice(0, row, row + 1)
                .copy_(host_tensor({std::cos(angle), 0.0f, 0.0f, std::sin(angle)}, {1, 4}).gpu());
            m.opacities.slice(0, row, row + 1).copy_(host_tensor({0.5f + 0.1f * f}, {1, 1}).gpu());
            m.sh0.slice(0, row, row + 1).copy_(host_tensor({0.1f * f, -0.05f * f, 0.2f}, {1, 1, 3}).gpu());
        }
        return m;
    }

    // An image gradient a function of the pixel.
    Tensor image_gradient() {
        std::vector<float> values(3 * kSize * kSize);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(static_cast<int>(i * 7919 % 2003) - 1001) / 1001.0f;
        return host_tensor(values, {3, kSize, kSize}).gpu();
    }

    // The forward's image and alpha and the raw gradients of the five parameter groups after one backward: means,
    // scales, rotations, opacities, sh0.
    struct Gradients {
        std::array<Tensor, 5> groups;
        std::vector<float> image, alpha;
        Tensor& operator[](const size_t i) { return groups[i]; }
        const Tensor& operator[](const size_t i) const { return groups[i]; }
    };

    Gradients run(const Rasterizer rasterizer, const GpuBackend backend, Model& m, const size_t rows) {
        const std::vector<float> view{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, kDepth, 0, 0, 0, 1};
        const Tensor camera = host_tensor({0, 0, -kDepth}, {3}).gpu();
        const Tensor background = host_tensor({0, 0, 0}, {3}).gpu();
        const Tensor none;
        // SH degree 0 has no rest coefficients; the model then holds a valid empty tensor, and CUDA's forwards
        // require a valid one.
        const Tensor no_sh_rest = Tensor::zeros({0}, Device::GPU);
        const ops::ShParams sh{.storage = ops::ShStorage::Float32, .active_bases = 1, .layout_bases = 1};
        const ops::SplatInputs splats{.means = m.means, .raw_scales = m.scales, .raw_rotations = m.rotations,
                                      .raw_opacities = m.opacities, .sh0 = m.sh0, .shN = no_sh_rest, .sh_value_bounds = none};
        Gradients g{{Tensor::zeros({rows, 3}, Device::GPU), Tensor::zeros({rows, 3}, Device::GPU),
                     Tensor::zeros({rows, 4}, Device::GPU), Tensor::zeros({rows, 1}, Device::GPU),
                     Tensor::zeros({rows, 1, 3}, Device::GPU)}};
        Tensor image, alpha, depth, normal;
        if (rasterizer == Rasterizer::Fast) {
            const ops::FastRasterOps& table = *lfs::training::training_ops(backend).fast;
            const ops::FastParams params{.full_image = {kSize, kSize},
                                         .intrinsics = {kFocal, kFocal, kCenter, kCenter},
                                         .sh = sh,
                                         .render_depth = false};
            Tensor share = Tensor::zeros({rows}, Device::GPU);
            ops::FastSaved saved;
            saved.backend = table.create();
            const auto result = table.forward(saved, splats, host_tensor(view, {1, 4, 4}).gpu(), camera, background,
                                              none, params, {.image = image, .alpha = alpha, .depth = depth, .normal = normal},
                                              share);
            EXPECT_EQ(result.code, ops::RasterResult::Code::Success) << result.message;
            if (result.code != ops::RasterResult::Code::Success) {
                table.release(saved);
                return g;
            }
            g.image = lm::host_copy<float>(image);
            g.alpha = lm::host_copy<float>(alpha);
            // Gradient capture: every group disabled, its raw gradient added into `gradient`, the parameters and the
            // (empty) moments untouched.
            Tensor dummy_moments = Tensor::zeros({4}, Device::GPU, DataType::UInt8);
            Tensor dummy_bounds = Tensor::zeros({1, 4}, Device::GPU);
            Tensor unused;
            std::array<Tensor*, 5> parameters{&m.means, &m.scales, &m.rotations, &m.opacities, &m.sh0};
            Tensor empty_parameter;
            const auto group = [&](const size_t i) {
                return ops::BackwardAdamParam{.parameter = i < 5 ? *parameters[i] : empty_parameter,
                                              .packed_moments = dummy_moments,
                                              .joint_bounds = dummy_bounds,
                                              .sh_value_bounds = unused,
                                              .frozen_mask = none,
                                              .crop_damping_mask = none,
                                              .screen_share = none,
                                              .joint_bits = 0,
                                              .primitives = static_cast<int>(rows),
                                              .elements = i < 5 ? static_cast<int>(std::min<size_t>(parameters[i]->numel(), 0x7fffffff)) : 0,
                                              .attributes = i == 2 ? 4 : i == 3 ? 1 : 3,
                                              .enabled = false,
                                              .gradient = i < 5 ? g[i] : Tensor{}};
            };
            Tensor scale_loss, opacity_loss;
            const ops::BackwardAdam adam{.groups = {group(0), group(1), group(2), group(3), group(4), group(5)},
                                         .scale_reg_loss = scale_loss,
                                         .opacity_reg_loss = opacity_loss};
            table.backward(saved, {.image = image_gradient(), .alpha = none, .depth = none, .normal = none}, unused,
                           none, none, unused, adam, DensificationType::None);
            table.release(saved);
            return g;
        }
        const ops::GsplatRasterOps& table = *lfs::training::training_ops(backend).gsplat;
        const ops::GsplatParams params{.full_image = {kSize, kSize},
                                       .intrinsics = {kFocal, kFocal, kCenter, kCenter},
                                       .sh = sh,
                                       .render_mode = ops::GsplatRenderMode::RGB};
        ops::GsplatSaved saved;
        saved.backend = table.create();
        const auto result = table.forward(saved, splats, host_tensor(view, {4, 4}).gpu(), none, none, background, none,
                                          params, {.image = image, .alpha = alpha, .depth = depth, .normal = normal});
        EXPECT_EQ(result.code, ops::RasterResult::Code::Success) << result.message;
        if (result.code != ops::RasterResult::Code::Success) {
            table.release(saved);
            return g;
        }
        g.image = lm::host_copy<float>(image);
        g.alpha = lm::host_copy<float>(alpha);
        Tensor rest;
        std::array<Tensor*, 6> slots{&g[0], &g[1], &g[2], &g[3], &g[4], &rest};
        const ops::GsplatGradients gradients{&slots, [](void* owner, ops::AdamSlot slot) -> Tensor& {
                                                 return *(*static_cast<std::array<Tensor*, 6>*>(owner))[static_cast<size_t>(slot)];
                                             }};
        Tensor densification = Tensor::zeros({2, rows}, Device::GPU);
        Tensor edge_scores = Tensor::zeros({rows}, Device::GPU);
        Tensor share = Tensor::zeros({rows}, Device::GPU);
        table.backward(saved, image_gradient(), Tensor::zeros({1, kSize, kSize}, Device::GPU), gradients, densification,
                       none, none, edge_scores, share);
        table.release(saved);
        return g;
    }

    class LargeModelBackward : public ::testing::TestWithParam<std::tuple<GpuBackend, Rasterizer>> {
    protected:
        void SetUp() override {
            const auto [backend, rasterizer] = GetParam();
            rows_ = rasterizer == Rasterizer::Fast ? rows_from("LFS_TEST_LARGE_FAST_ROWS", past_floats(12) + 74)
                                                   : rows_from("LFS_TEST_LARGE_GSPLAT_ROWS", past_floats(14) + 13);
            ASSERT_GE(rows_, size_t{64});
            // Parameters and captured gradients (14 floats each), the forward's buffers (about 24 values), the
            // backward's gradient rows (12 or 14) and the screen share and densification rows.
            if (const auto reason = lm::skip_reason(rows_ * (28 + 24 + 14 + 4) * sizeof(float)))
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

        size_t rows_ = 0;
        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(LargeModelBackward, GradientsOfFarSplatsMatchTheSameSplatsAlone) {
        const auto [backend, rasterizer] = GetParam();
        const std::vector<size_t> visible = visible_rows(rows_);
        ASSERT_LE(visible.size(), size_t{25});

        std::vector<size_t> small_rows(visible.size());
        for (size_t k = 0; k < visible.size(); ++k)
            small_rows[k] = k;
        Model small = make_model(visible.size(), small_rows);
        const Gradients expected = run(rasterizer, backend, small, visible.size());

        Model big = make_model(rows_, visible);
        const Gradients actual = run(rasterizer, backend, big, rows_);

        // The forward: the frames bit for bit (a wrong record address draws a far splat from another record).
        const auto same_frame = [](const std::vector<float>& a, const std::vector<float>& b) {
            return a.size() == b.size() && !a.empty() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };
        EXPECT_TRUE(same_frame(actual.image, expected.image)) << "the image differs from the same splats alone";
        EXPECT_TRUE(same_frame(actual.alpha, expected.alpha)) << "the alpha differs from the same splats alone";

        static constexpr std::array<const char*, 5> kNames{"means", "scales", "rotations", "opacities", "sh0"};
        static constexpr std::array<size_t, 5> kWidths{3, 3, 4, 1, 3};
        for (size_t group = 0; group < 5; ++group) {
            const size_t w = kWidths[group];
            const auto want = lm::host_copy<float>(expected[group]);
            double scale = 0;
            for (const float v : want)
                scale = std::max(scale, std::abs(double(v)));
            EXPECT_GT(scale, 0.0) << kNames[group] << ": the small model's gradients are all zero";
            const auto flat = actual[group].flatten();
            for (size_t k = 0; k < visible.size(); ++k) {
                const auto got = lm::host_copy<float>(flat.slice(0, visible[k] * w, (visible[k] + 1) * w));
                for (size_t c = 0; c < w; ++c)
                    EXPECT_NEAR(got[c], want[k * w + c], 1e-4 * scale + 1e-6)
                        << kNames[group] << " gradient of row " << visible[k] << ", component " << c;
            }
            // Rows near the visible ones that no splat covers get no gradient at all.
            for (const size_t first : {size_t{3}, rows_ - 12}) {
                const auto quiet = lm::host_copy<float>(flat.slice(0, first * w, (first + 8) * w));
                for (size_t i = 0; i < quiet.size(); ++i) {
                    const size_t row = first + i / w;
                    if (std::find(visible.begin(), visible.end(), row) == visible.end())
                        EXPECT_EQ(quiet[i], 0.0f) << kNames[group] << " gradient of the invisible row " << row;
                }
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        Backends, LargeModelBackward,
        testing::Combine(testing::Values(GpuBackend::CUDA, GpuBackend::Metal, GpuBackend::Vulkan),
                         testing::Values(Rasterizer::Fast, Rasterizer::Gsplat)),
        [](const auto& info) {
            return std::string(lfs::core::gpu_backend_name(std::get<0>(info.param))) +
                   (std::get<1>(info.param) == Rasterizer::Fast ? "_Fast" : "_Gsplat");
        });

} // namespace
