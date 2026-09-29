/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/kernels/bilateral_grid.cuh"

#include <cstdint>
#include <vector>

namespace {
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace ops = lfs::gpu_ops;
    namespace kernels = lfs::training::kernels;
    using BilateralOpsTest = lfs::test::CudaBackendTest;

    Tensor pattern(const lfs::core::TensorShape& shape, const int seed = 7) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>((i * 13 + seed) % 31 + 1) / 1024.f;
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    std::vector<uint8_t> bytes(const Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.bytes()};
    }

    void same(const Tensor& actual, const Tensor& expected) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(bytes(actual), bytes(expected));
    }

    const ops::BilateralOps& table() {
        return *lfs::training::training_ops(GpuBackend::CUDA).bilateral;
    }

    TEST_F(BilateralOpsTest, SliceForwardAndBackwardMatchLaunchers) {
        // One nonzero gradient pixel removes atomic ordering from the byte oracle.
        for (const bool chroma : {false, true}) {
            for (const bool chw : {false, true}) {
                for (const bool warp : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << chroma << ' ' << chw << ' ' << warp);
                    const size_t channels = chroma ? 9 : 12;
                    auto grid = pattern({1, channels, 2, 2, 2});
                    auto rgb = pattern(chw ? lfs::core::TensorShape{3, 5, 7} : lfs::core::TensorShape{5, 7, 3}, 11);
                    auto offset = pattern({channels}, 17);
                    auto output = Tensor::zeros(rgb.shape(), Device::GPU);
                    auto expected = output.clone();
                    std::vector<float> gradient(rgb.numel(), 0.f);
                    for (size_t c = 0; c < 3; ++c)
                        gradient[chw ? c * 35 + 17 : 17 * 3 + c] = static_cast<float>(c + 1) / 8.f;
                    auto grad_output = Tensor::from_vector(gradient, rgb.shape(), Device::GPU);
                    auto grad_grid = Tensor::zeros(grid.shape(), Device::GPU);
                    auto direct_grid = grad_grid.clone();
                    auto grad_rgb = output.clone();
                    auto direct_rgb = output.clone();
                    const ops::GridSliceParams params{chw ? ops::Layout::CHW : ops::Layout::HWC, chroma ? ops::GridTransform::ExposureChroma : ops::GridTransform::Affine, warp};
                    const auto forward = chroma
                                             ? (chw ? kernels::launch_bilateral_grid_slice_forward_exposure_chroma_chw : kernels::launch_bilateral_grid_slice_forward_exposure_chroma)
                                             : (chw ? kernels::launch_bilateral_grid_slice_forward_chw : kernels::launch_bilateral_grid_slice_forward);
                    const auto backward = chroma
                                              ? (chw ? kernels::launch_bilateral_grid_slice_backward_exposure_chroma_chw : kernels::launch_bilateral_grid_slice_backward_exposure_chroma)
                                              : (chw ? kernels::launch_bilateral_grid_slice_backward_chw : kernels::launch_bilateral_grid_slice_backward);
                    const auto before = bytes(output);
                    table().slice_forward(grid, rgb, offset, output, params);
                    forward(grid.ptr<float>(), rgb.ptr<float>(), expected.ptr<float>(), 2, 2, 2, 5, 7, offset.ptr<float>(), nullptr);
                    same(output, expected);
                    EXPECT_NE(bytes(output), before);
                    table().slice_backward(grid, rgb, grad_output, offset, grad_grid, grad_rgb, params);
                    backward(grid.ptr<float>(), rgb.ptr<float>(), grad_output.ptr<float>(), direct_grid.ptr<float>(), direct_rgb.ptr<float>(), 2, 2, 2, 5, 7, offset.ptr<float>(), nullptr, warp);
                    same(grad_grid, direct_grid);
                    same(grad_rgb, direct_rgb);
                    EXPECT_NE(bytes(grad_rgb), before);
                }
            }
        }
    }

    TEST_F(BilateralOpsTest, TotalVariationMatchesLaunchers) {
        auto grid = pattern({2, 9, 3, 4, 5});
        auto loss = Tensor::zeros({1}, Device::GPU);
        auto direct_loss = loss.clone();
        auto temp = Tensor::empty({2048}, Device::GPU);
        auto direct_temp = Tensor::empty({2048}, Device::GPU);
        auto gradient = pattern(grid.shape(), 11);
        auto direct_gradient = gradient.clone();
        const auto before = bytes(gradient);
        table().tv_forward(grid, loss, temp, 7);
        kernels::launch_bilateral_grid_tv_forward(grid.ptr<float>(), direct_loss.ptr<float>(), direct_temp.ptr<float>(), 2, 9, 3, 4, 5, 7, nullptr);
        same(loss, direct_loss);
        EXPECT_GT(loss.cpu().item<float>(), 0.f);
        table().tv_backward(grid, gradient, 0.25f, 7);
        kernels::launch_bilateral_grid_tv_backward(grid.ptr<float>(), 0.25f, direct_gradient.ptr<float>(), 2, 9, 3, 4, 5, 7, nullptr);
        same(gradient, direct_gradient);
        EXPECT_NE(bytes(gradient), before);
    }

    TEST_F(BilateralOpsTest, ProjectionAndOffsetMatchLaunchers) {
        for (const int per_image : {0, 1}) {
            auto grid = pattern({2, 12, 2, 3, 4});
            auto direct_grid = grid.clone();
            auto mean = pattern({per_image ? size_t{24} : size_t{12}}, 11);
            auto identity = pattern({12}, 19);
            const auto before = bytes(grid);
            table().project_mean(grid, mean, identity, per_image);
            kernels::launch_bilateral_grid_project_mean(direct_grid.ptr<float>(), mean.ptr<float>(), identity.ptr<float>(), 2, 12, 2, 3, 4, per_image, nullptr);
            same(grid, direct_grid);
            EXPECT_NE(bytes(grid), before);
        }
        auto sum = pattern({12});
        auto offset = pattern({12}, 11);
        auto direct_sum = sum.clone();
        auto direct_offset = offset.clone();
        auto identity = pattern({12}, 19);
        auto old_mean = pattern({12}, 23);
        auto new_mean = pattern({12}, 29);
        const auto before = bytes(offset);
        table().update_offset(sum, offset, identity, old_mean, new_mean, 24.f, 1.f / 48.f);
        kernels::launch_bilateral_grid_update_shared_offset(direct_sum.ptr<float>(), direct_offset.ptr<float>(), identity.ptr<float>(), old_mean.ptr<float>(), new_mean.ptr<float>(), 12, 24.f, 1.f / 48.f, nullptr);
        same(sum, direct_sum);
        same(offset, direct_offset);
        EXPECT_NE(bytes(offset), before);
    }

    TEST_F(BilateralOpsTest, AdamAndMomentScalingMatchLaunchers) {
        auto grid = pattern({1, 12, 2, 3, 4});
        auto m1 = pattern(grid.shape(), 11);
        auto m2 = pattern(grid.shape(), 17);
        auto gradient = pattern(grid.shape(), 19);
        auto direct_grid = grid.clone();
        auto direct_m1 = m1.clone();
        auto direct_m2 = m2.clone();
        const auto before = bytes(grid);
        const ops::AdamUpdateParams params{0.002f, 0.9f, 0.999f, 10.f, 31.622776f, 1e-15f};
        table().adam(grid, m1, m2, gradient, params);
        kernels::launch_bilateral_grid_adam_update(direct_grid.ptr<float>(), direct_m1.ptr<float>(), direct_m2.ptr<float>(), gradient.ptr<float>(), 288, params.lr, params.beta1, params.beta2, params.bc1_rcp, params.bc2_sqrt_rcp, params.eps, nullptr);
        same(grid, direct_grid);
        same(m1, direct_m1);
        same(m2, direct_m2);
        EXPECT_NE(bytes(grid), before);
        const auto moments_before = bytes(m1);
        table().scale_moments(m1, m2, 0.81f, 0.998001f);
        kernels::launch_bilateral_grid_scale_moments(direct_m1.ptr<float>(), direct_m2.ptr<float>(), 288, 0.81f, 0.998001f, nullptr);
        same(m1, direct_m1);
        same(m2, direct_m2);
        EXPECT_NE(bytes(m1), moments_before);
    }

    TEST_F(BilateralOpsTest, RegistryRequiresBilateralOnlyWhenEnabled) {
        using namespace lfs::training;
        lfs::core::param::TrainingParameters params;
        EXPECT_FALSE(required_training_families(params, {}).test(static_cast<size_t>(Family::Bilateral)));
        params.optimization.use_bilateral_grid = true;
        EXPECT_TRUE(required_training_families(params, {}).test(static_cast<size_t>(Family::Bilateral)));
        params.optimization.use_bilateral_grid = false;
        params.optimization.use_exposure_correction = true;
        EXPECT_TRUE(required_training_families(params, {}).test(static_cast<size_t>(Family::Bilateral)));
        FamilySet required;
        required.set(static_cast<size_t>(Family::Bilateral));
        EXPECT_TRUE(missing_training_families(training_ops(GpuBackend::CUDA), required).empty());
        EXPECT_NE(training_ops(GpuBackend::Vulkan).bilateral, nullptr);
        EXPECT_TRUE(missing_training_families(training_ops(GpuBackend::Vulkan), required).empty());
        EXPECT_EQ(training_ops(GpuBackend::Metal).bilateral, nullptr);
        EXPECT_EQ(missing_training_families(training_ops(GpuBackend::Metal), required), std::vector<std::string_view>{"Bilateral"});
        auto missing = training_ops(GpuBackend::CUDA);
        missing.bilateral = nullptr;
        EXPECT_EQ(missing_training_families(missing, required), std::vector<std::string_view>{"Bilateral"});
    }
} // namespace
