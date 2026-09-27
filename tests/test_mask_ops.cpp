/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/kernels/mask_preprocess.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs::core;
    namespace ops = lfs::gpu_ops;
    namespace kernels = lfs::training::kernels;
    using MaskOpsTest = lfs::test::CudaBackendTest;

    std::vector<uint8_t> bytes(const Tensor& t) {
        const auto cpu = t.cpu().contiguous();
        const auto* p = static_cast<const uint8_t*>(cpu.data_ptr());
        return {p, p + cpu.bytes()};
    }

    TEST_F(MaskOpsTest, MatchesDirectLaunchers) {
        constexpr size_t h = 33, w = 35;
        std::vector<uint8_t> values(h * w);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<uint8_t>((i * 79 + 31) % 256);
        }
        auto cpu_mask = Tensor::empty({h, w}, Device::CPU, DataType::UInt8);
        for (size_t i = 0; i < values.size(); ++i) {
            cpu_mask.ptr<uint8_t>()[i] = values[i];
        }
        const auto u8 = cpu_mask.gpu();
        const auto alpha = Tensor::full({h, w}, 0.43f, Device::GPU);
        const auto& table = *lfs::training::training_ops(GpuBackend::CUDA).masks;
        for (const auto dtype : {DataType::UInt8, DataType::Float32, DataType::Bool}) {
            const auto mask = dtype == DataType::Float32 ? u8.to(dtype) / 255.0f : u8.to(dtype);
            for (const bool use_roi : {false, true}) {
                const auto roi = use_roi ? Tensor::full({h, w}, 0.37f, Device::GPU) : Tensor{};
                const auto* rp = use_roi ? roi.ptr<float>() : nullptr;
                auto actual = Tensor::full({h, w}, -9.f, Device::GPU);
                auto expected = actual.clone();
                auto temp = Tensor::zeros({1024}, Device::GPU);
                auto direct_temp = temp.clone();
                auto loss = Tensor::full({1}, -9.f, Device::GPU);
                auto direct_loss = loss.clone();
                for (const auto mode : {ops::MaskPhotoMode::BinaryGt0, ops::MaskPhotoMode::SegmentAndIgnore}) {
                    table.photometric_weight(mask, roi, actual, mode);
                    if (dtype == DataType::Float32) {
                        kernels::launch_fuse_photometric_mask_weight_f32(mask.ptr<float>(), rp, expected.ptr<float>(), h, w, mode);
                    } else {
                        kernels::launch_fuse_photometric_mask_weight_u8(mask.ptr<uint8_t>(), rp, expected.ptr<float>(), h, w, mode);
                    }
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    EXPECT_EQ(bytes(actual), bytes(expected));
                }
                for (const auto mode : {ops::MaskOpacityMode::BinaryGt0, ops::MaskOpacityMode::SegmentAndIgnore}) {
                    table.opacity_penalty(alpha, mask, roi, actual, temp, loss, mode, 1.7f, 0.13f);
                    if (dtype == DataType::Float32) {
                        kernels::launch_fuse_mask_opacity_penalty_f32(alpha.ptr<float>(), mask.ptr<float>(), rp,
                                                                      expected.ptr<float>(), direct_temp.ptr<float>(), direct_loss.ptr<float>(), h, w, 1.7f, 0.13f, mode);
                    } else {
                        kernels::launch_fuse_mask_opacity_penalty_u8(alpha.ptr<float>(), mask.ptr<uint8_t>(), rp,
                                                                     expected.ptr<float>(), direct_temp.ptr<float>(), direct_loss.ptr<float>(), h, w, 1.7f, 0.13f, mode);
                    }
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    EXPECT_EQ(bytes(actual), bytes(expected));
                    EXPECT_EQ(bytes(loss), bytes(direct_loss));
                    EXPECT_EQ(bytes(temp), bytes(direct_temp));
                }
                table.alpha_consistency(alpha, mask, roi, actual, temp, loss, 10.f);
                if (dtype == DataType::Float32) {
                    kernels::launch_fuse_alpha_consistent_f32(alpha.ptr<float>(), mask.ptr<float>(), rp,
                                                              expected.ptr<float>(), direct_temp.ptr<float>(), direct_loss.ptr<float>(), h, w, 10.f);
                } else {
                    kernels::launch_fuse_alpha_consistent_u8(alpha.ptr<float>(), mask.ptr<uint8_t>(), rp,
                                                             expected.ptr<float>(), direct_temp.ptr<float>(), direct_loss.ptr<float>(), h, w, 10.f);
                }
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                EXPECT_EQ(bytes(actual), bytes(expected));
                EXPECT_EQ(bytes(loss), bytes(direct_loss));
                EXPECT_EQ(bytes(temp), bytes(direct_temp));
            }
        }
    }

    TEST_F(MaskOpsTest, RequiresCudaFamiliesForEnabledLosses) {
        using namespace lfs::training;
        param::TrainingParameters p;
        p.optimization.scale_reg = 0.f;
        p.optimization.opacity_reg = 0.f;
        p.optimization.enable_sparsity = false;
        p.optimization.mask_mode = param::MaskMode::None;
        EXPECT_FALSE(required_training_families(p, {}).test(static_cast<size_t>(Family::Masks)));
        EXPECT_FALSE(required_training_families(p, {}).test(static_cast<size_t>(Family::ExtraLoss)));
        for (const auto mode : {param::MaskMode::Segment, param::MaskMode::Ignore,
                                param::MaskMode::SegmentAndIgnore, param::MaskMode::AlphaConsistent}) {
            p.optimization.mask_mode = mode;
            EXPECT_TRUE(required_training_families(p, {}).test(static_cast<size_t>(Family::Masks)));
        }
        for (int i = 0; i < 3; ++i) {
            p.optimization.scale_reg = i == 0 ? 0.1f : 0.f;
            p.optimization.opacity_reg = i == 1 ? 0.1f : 0.f;
            p.optimization.enable_sparsity = i == 2;
            EXPECT_TRUE(required_training_families(p, {}).test(static_cast<size_t>(Family::ExtraLoss)));
        }
        FamilySet families;
        families.set(static_cast<size_t>(Family::Masks));
        families.set(static_cast<size_t>(Family::ExtraLoss));
        EXPECT_TRUE(missing_training_families(training_ops(GpuBackend::CUDA), families).empty());
        for (const auto backend : {GpuBackend::Vulkan, GpuBackend::Metal}) {
            EXPECT_EQ(training_ops(backend).masks, nullptr);
            EXPECT_EQ(training_ops(backend).extra_loss, nullptr);
            EXPECT_EQ(missing_training_families(training_ops(backend), families),
                      (std::vector<std::string_view>{"Masks", "ExtraLoss"}));
        }
    }
} // namespace
