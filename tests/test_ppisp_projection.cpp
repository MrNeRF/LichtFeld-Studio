/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/ppisp.hpp"
#include "core/tensor.hpp"
#include "lfs/kernels/ppisp.cuh"
#include "tensor_hardening_test_utils.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::PPISP;

    class PPISPProjectionTest : public tensor_hardening::CudaTest {};

    std::vector<float> cpu_copy(const Tensor& tensor) {
        return tensor.cpu().contiguous().to_vector();
    }

    TEST_F(PPISPProjectionTest, ProjectMeanZerosExposureAndColorAndPreservesGaps) {
        PPISP ppisp(100);
        ppisp.register_frame(0, 1);
        ppisp.register_frame(1, 1);
        ppisp.register_frame(2, 2);
        ppisp.finalize();

        const auto vig_before = cpu_copy(ppisp.vignetting_params());
        const auto crf_before = cpu_copy(ppisp.crf_params());

        std::vector<float> exposure = {0.5f, -0.25f, 1.0f};
        std::vector<float> color(3 * 8);
        for (int f = 0; f < 3; ++f) {
            for (int c = 0; c < 8; ++c) {
                color[static_cast<size_t>(f * 8 + c)] = 0.1f * static_cast<float>(f + 1) +
                                                        0.01f * static_cast<float>(c);
            }
        }
        ppisp.exposure_params().copy_from(
            Tensor::from_vector(exposure, lfs::core::TensorShape({3}), Device::GPU));
        ppisp.color_params().copy_from(
            Tensor::from_vector(color, lfs::core::TensorShape({24}), Device::GPU));

        std::vector<float> exp_diffs = {exposure[1] - exposure[0], exposure[2] - exposure[1]};
        std::vector<float> color_diffs(2 * 8);
        for (int c = 0; c < 8; ++c) {
            color_diffs[static_cast<size_t>(c)] = color[static_cast<size_t>(8 + c)] - color[static_cast<size_t>(c)];
            color_diffs[static_cast<size_t>(8 + c)] =
                color[static_cast<size_t>(16 + c)] - color[static_cast<size_t>(8 + c)];
        }

        ppisp.project_mean();

        const auto exp_after = cpu_copy(ppisp.exposure_params());
        const auto color_after = cpu_copy(ppisp.color_params());
        float exp_mean = 0.0f;
        for (float v : exp_after)
            exp_mean += v;
        EXPECT_NEAR(exp_mean / 3.0f, 0.0f, 1e-6f);
        EXPECT_NEAR(exp_after[1] - exp_after[0], exp_diffs[0], 1e-6f);
        EXPECT_NEAR(exp_after[2] - exp_after[1], exp_diffs[1], 1e-6f);

        ASSERT_EQ(color_after.size(), 24u);
        for (int c = 0; c < 8; ++c) {
            float mean = 0.0f;
            for (int f = 0; f < 3; ++f)
                mean += color_after[static_cast<size_t>(f * 8 + c)];
            EXPECT_NEAR(mean / 3.0f, 0.0f, 1e-6f) << "color channel " << c;
            EXPECT_NEAR(color_after[static_cast<size_t>(8 + c)] - color_after[static_cast<size_t>(c)],
                        color_diffs[static_cast<size_t>(c)], 1e-6f);
            EXPECT_NEAR(color_after[static_cast<size_t>(16 + c)] - color_after[static_cast<size_t>(8 + c)],
                        color_diffs[static_cast<size_t>(8 + c)], 1e-6f);
        }

        const auto vig_after = cpu_copy(ppisp.vignetting_params());
        const auto crf_after = cpu_copy(ppisp.crf_params());
        ASSERT_EQ(vig_before.size(), vig_after.size());
        ASSERT_EQ(crf_before.size(), crf_after.size());
        for (size_t i = 0; i < vig_before.size(); ++i)
            EXPECT_EQ(vig_before[i], vig_after[i]);
        for (size_t i = 0; i < crf_before.size(); ++i)
            EXPECT_EQ(crf_before[i], crf_after[i]);
    }

    TEST_F(PPISPProjectionTest, MajorityCameraIdPicksMostRegisteredFrames) {
        PPISP ppisp(100);
        ppisp.register_frame(10, 7);
        ppisp.register_frame(11, 3);
        ppisp.register_frame(12, 3);
        ppisp.register_frame(13, 3);
        ppisp.register_frame(14, 7);
        ppisp.finalize();
        EXPECT_EQ(ppisp.majority_camera_id(), 3);
    }

    TEST_F(PPISPProjectionTest, MajorityCameraIdTieBreaksToSmallestId) {
        PPISP ppisp(100);
        ppisp.register_frame(1, 9);
        ppisp.register_frame(2, 4);
        ppisp.finalize();
        EXPECT_EQ(ppisp.majority_camera_id(), 4);
    }

    TEST_F(PPISPProjectionTest, FusedVignettingRegMatchesClosedForm) {
        constexpr int kCameras = 3;
        constexpr float kCenter = 0.02f;
        constexpr float kChannel = 0.1f;
        constexpr float kNonPos = 0.01f;
        std::vector<float> vig(static_cast<size_t>(kCameras) * 15);
        for (size_t i = 0; i < vig.size(); ++i) {
            vig[i] = 0.15f * std::sin(0.37f * static_cast<float>(i + 1)) - 0.04f;
        }
        auto vig_gpu = Tensor::from_vector(
            vig, lfs::core::TensorShape({vig.size()}), Device::GPU);
        auto grad_gpu = Tensor::zeros({vig.size()}, Device::GPU);
        auto loss_gpu = Tensor::zeros({1}, Device::GPU);

        lfs::training::kernels::launch_ppisp_vignetting_reg(
            vig_gpu.ptr<float>(), grad_gpu.ptr<float>(), loss_gpu.ptr<float>(),
            kCameras, kCenter, kChannel, kNonPos, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        float expected_loss = 0.0f;
        std::vector<float> expected_grad(vig.size(), 0.0f);
        const float inv_center = 1.0f / static_cast<float>(kCameras * 3);
        const float inv_non_pos = 1.0f / static_cast<float>(kCameras * 9);
        const float inv_channel = 1.0f / static_cast<float>(kCameras * 15);
        for (int cam = 0; cam < kCameras; ++cam) {
            for (int ch = 0; ch < 3; ++ch) {
                const size_t base = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5;
                const float cx = vig[base + 0];
                const float cy = vig[base + 1];
                expected_loss += kCenter * (cx * cx + cy * cy) * inv_center;
                expected_grad[base + 0] += kCenter * 2.0f * inv_center * cx;
                expected_grad[base + 1] += kCenter * 2.0f * inv_center * cy;
                for (int a = 0; a < 3; ++a) {
                    if (vig[base + 2 + static_cast<size_t>(a)] > 0.0f) {
                        expected_loss += kNonPos * vig[base + 2 + static_cast<size_t>(a)] * inv_non_pos;
                        expected_grad[base + 2 + static_cast<size_t>(a)] += kNonPos * inv_non_pos;
                    }
                }
            }
            for (int p = 0; p < 5; ++p) {
                float vals[3];
                size_t idxs[3];
                float mean = 0.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    idxs[ch] = static_cast<size_t>(cam) * 15 + static_cast<size_t>(ch) * 5 +
                               static_cast<size_t>(p);
                    vals[ch] = vig[idxs[ch]];
                    mean += vals[ch];
                }
                mean /= 3.0f;
                for (int ch = 0; ch < 3; ++ch) {
                    const float diff = vals[ch] - mean;
                    expected_loss += kChannel * diff * diff * inv_channel;
                    expected_grad[idxs[ch]] += kChannel * 2.0f * inv_channel * diff;
                }
            }
        }

        EXPECT_NEAR(cpu_copy(loss_gpu)[0], expected_loss, 1e-5f);
        const auto got_grad = cpu_copy(grad_gpu);
        ASSERT_EQ(got_grad.size(), expected_grad.size());
        for (size_t i = 0; i < expected_grad.size(); ++i) {
            EXPECT_NEAR(got_grad[i], expected_grad[i], 1e-5f) << "grad index " << i;
        }
    }

} // namespace
