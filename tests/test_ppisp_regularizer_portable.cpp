/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// PPISP regularizer loss and gradients on Metal and Vulkan against the host
// computation they replaced (which read the parameters back every step).

#include "components/ppisp.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "cuda_backend_test.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace {
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    using lfs::training::PPISP;

    constexpr int kCameras = 2;
    constexpr int kFrames = 5;

    float smooth_l1(const float x, const float beta) {
        const float a = std::abs(x);
        return a < beta ? 0.5f * x * x / beta : a - 0.5f * beta;
    }
    float smooth_l1_grad(const float x, const float beta) {
        return std::abs(x) < beta ? x / beta : (x > 0.0f ? 1.0f : -1.0f);
    }

    struct Reference {
        float loss = 0.0f;
        std::vector<float> exposure, vignetting, color, crf;
    };

    // Transliteration of the former host-side PPISP::reg_loss_gpu / reg_backward.
    Reference reference(const lfs::training::PPISPConfig& c, const std::vector<float>& e,
                        const std::vector<float>& v, const std::vector<float>& col,
                        const std::vector<float>& crf, const std::vector<float>& pinv) {
        Reference r;
        r.exposure.assign(e.size(), 0.0f);
        r.vignetting.assign(v.size(), 0.0f);
        r.color.assign(col.size(), 0.0f);
        r.crf.assign(crf.size(), 0.0f);
        float mean = 0.0f;
        for (const float x : e)
            mean += x;
        mean /= kFrames;
        r.loss += c.exposure_mean * smooth_l1(mean, 0.1f);
        for (auto& g : r.exposure)
            g += c.exposure_mean * smooth_l1_grad(mean, 0.1f) / kFrames;
        for (int cam = 0; cam < kCameras; ++cam) {
            for (int ch = 0; ch < 3; ++ch) {
                const int b = cam * 15 + ch * 5;
                r.loss += c.vig_center * (v[b] * v[b] + v[b + 1] * v[b + 1]) / (kCameras * 3);
                r.vignetting[b] += c.vig_center * 2.0f / (kCameras * 3) * v[b];
                r.vignetting[b + 1] += c.vig_center * 2.0f / (kCameras * 3) * v[b + 1];
                for (int a = 0; a < 3; ++a) {
                    if (v[b + 2 + a] > 0.0f) {
                        r.loss += c.vig_non_pos * v[b + 2 + a] / (kCameras * 9);
                        r.vignetting[b + 2 + a] += c.vig_non_pos / (kCameras * 9);
                    }
                }
            }
            for (int p = 0; p < 5; ++p) {
                const float m = (v[cam * 15 + p] + v[cam * 15 + 5 + p] + v[cam * 15 + 10 + p]) / 3.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    const float d = v[cam * 15 + ch * 5 + p] - m;
                    r.loss += c.vig_channel * d * d / 3.0f / (kCameras * 5);
                    r.vignetting[cam * 15 + ch * 5 + p] += c.vig_channel / (kCameras * 5) * 2.0f * d / 3.0f;
                }
            }
            for (int p = 0; p < 4; ++p) {
                const float m = (crf[cam * 12 + p] + crf[cam * 12 + 4 + p] + crf[cam * 12 + 8 + p]) / 3.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    const float d = crf[cam * 12 + ch * 4 + p] - m;
                    r.loss += c.crf_channel * d * d / 3.0f / (kCameras * 4);
                    r.crf[cam * 12 + ch * 4 + p] += c.crf_channel / (kCameras * 4) * 2.0f * d / 3.0f;
                }
            }
        }
        std::array<float, 8> offsets{};
        for (int f = 0; f < kFrames; ++f)
            for (int j = 0; j < 8; ++j)
                for (int k = 0; k < 8; ++k)
                    offsets[j] += col[f * 8 + k] * pinv[k * 8 + j] / kFrames;
        for (int j = 0; j < 8; ++j)
            r.loss += c.color_mean * smooth_l1(offsets[j], 0.005f) / 8.0f;
        for (int f = 0; f < kFrames; ++f)
            for (int k = 0; k < 8; ++k)
                for (int j = 0; j < 8; ++j)
                    r.color[f * 8 + k] +=
                        c.color_mean * smooth_l1_grad(offsets[j], 0.005f) / 8.0f * pinv[k * 8 + j] / kFrames;
        return r;
    }

    std::vector<float> pattern(const size_t n, const float scale, const int seed) {
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i)
            values[i] = scale * (static_cast<float>((i * 7919u + seed * 104729u) % 2003u) / 1001.0f - 1.0f);
        return values;
    }

    std::vector<float> host(const Tensor& t) { return t.cpu().to_vector(); }

    void set(Tensor target, const std::vector<float>& values) {
        target.copy_(Tensor::from_vector(values, {values.size()}, Device::GPU));
    }

    void expect_close(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& what) {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        for (size_t i = 0; i < actual.size(); ++i)
            EXPECT_NEAR(actual[i], expected[i], 1e-6f + 1e-5f * std::abs(expected[i])) << what << " " << i;
    }

    class PortablePpispRegularizer : public ::testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << lfs::core::gpu_backend_name(GetParam()) << " device unavailable";
            session_.emplace(GetParam());
            ASSERT_TRUE(session_->switched());
        }
        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
    };

    TEST_P(PortablePpispRegularizer, LossAndGradientsMatchHostReference) {
        lfs::training::PPISPConfig config;
        PPISP ppisp(1000, config);
        for (int f = 0; f < kFrames; ++f)
            ppisp.register_frame(f, f % kCameras);
        ppisp.finalize();

        // Exposure mean inside smooth-L1's quadratic zone, color offsets on both sides.
        const auto e = pattern(kFrames, 0.05f, 1);
        const auto v = pattern(kCameras * 15, 0.3f, 2);
        const auto col = pattern(kFrames * 8, 0.4f, 3);
        const auto crf = pattern(kCameras * 12, 0.8f, 4);
        set(ppisp.exposure_params(), e);
        set(ppisp.vignetting_params(), v);
        set(ppisp.color_params(), col);
        set(ppisp.crf_params(), crf);

        // pinv of the ZCA color basis, as in PPISP::init_color_pinv_block_diag.
        std::vector<float> pinv(64, 0.0f);
        const std::array<std::array<float, 4>, 4> blocks{{{0.0480542f, -0.0043631f, -0.0043631f, 0.0481283f},
                                                          {0.0580570f, -0.0179872f, -0.0179872f, 0.0431061f},
                                                          {0.0433336f, -0.0180537f, -0.0180537f, 0.0580500f},
                                                          {0.0128369f, -0.0034654f, -0.0034654f, 0.0128158f}}};
        for (int b = 0; b < 4; ++b) {
            pinv[(2 * b) * 8 + 2 * b] = blocks[b][0];
            pinv[(2 * b) * 8 + 2 * b + 1] = blocks[b][1];
            pinv[(2 * b + 1) * 8 + 2 * b] = blocks[b][2];
            pinv[(2 * b + 1) * 8 + 2 * b + 1] = blocks[b][3];
        }
        const Reference expected = reference(config, e, v, col, crf, pinv);

        const auto loss = host(ppisp.reg_loss_gpu());
        ASSERT_EQ(loss.size(), 1u);
        EXPECT_NEAR(loss[0], expected.loss, 1e-5f * std::abs(expected.loss));

        ppisp.zero_grad();
        ppisp.reg_backward();
        expect_close(host(ppisp.exposure_grad()), expected.exposure, "exposure grad");
        expect_close(host(ppisp.vignetting_grad()), expected.vignetting, "vignetting grad");
        expect_close(host(ppisp.color_grad()), expected.color, "color grad");
        expect_close(host(ppisp.crf_grad()), expected.crf, "crf grad");
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PortablePpispRegularizer, testing::Values(GpuBackend::Metal, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(lfs::core::gpu_backend_name(info.param)); });
} // namespace
