/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/kernels/l1_loss.cuh"
#include "lfs/kernels/ssim.cuh"
#include "lfs/training/ops/photometric_cuda.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/metrics/metrics.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;

    Tensor fixed_image(const int n, const int c, const int h, const int w, const float bias) {
        auto cpu = Tensor::empty(
            {static_cast<size_t>(n), static_cast<size_t>(c), static_cast<size_t>(h), static_cast<size_t>(w)},
            Device::CPU);
        float* values = cpu.ptr<float>();
        for (int batch = 0; batch < n; ++batch) {
            for (int channel = 0; channel < c; ++channel) {
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        values[((batch * c + channel) * h + y) * w + x] =
                            bias + 0.01f * static_cast<float>(channel) +
                            0.001f * static_cast<float>(x + y) +
                            0.0001f * static_cast<float>(batch);
                    }
                }
            }
        }
        return cpu.to(Device::GPU);
    }

    Tensor fixed_mask(const int h, const int w) {
        auto cpu = Tensor::empty({static_cast<size_t>(h), static_cast<size_t>(w)}, Device::CPU);
        float* values = cpu.ptr<float>();
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                values[y * w + x] = ((x + y) % 5 == 0) ? 0.0f : 1.0f;
            }
        }
        return cpu.to(Device::GPU);
    }

    void expect_identical(const Tensor& actual, const Tensor& reference) {
        ASSERT_TRUE(actual.is_valid());
        ASSERT_TRUE(reference.is_valid());
        ASSERT_EQ(actual.shape(), reference.shape());
        ASSERT_EQ(actual.dtype(), reference.dtype());
        const auto actual_cpu = actual.cpu().contiguous();
        const auto reference_cpu = reference.cpu().contiguous();
        ASSERT_EQ(actual_cpu.bytes(), reference_cpu.bytes());
        EXPECT_EQ(0, std::memcmp(actual_cpu.ptr<float>(), reference_cpu.ptr<float>(), actual_cpu.bytes()));
    }

    // The outputs are views into the saved workspace, so the run keeps it alive.
    struct PhotoRun {
        lfs::gpu_ops::PhotoSaved saved;
        Tensor loss;
        Tensor grad_corrected;
        Tensor grad_raw;
    };

    PhotoRun evaluate_path(const lfs::gpu_ops::PhotoPath path, const float weight, const Tensor& corrected,
                           const Tensor& raw, const Tensor& target, const Tensor& mask) {
        const auto& ops = lfs::training::cuda_photometric_ops();
        PhotoRun run{.saved = {.backend = ops.create()}};
        ops.evaluate(
            run.saved, corrected, raw, target, mask,
            {.path = path, .ssim_weight = weight, .valid_padding = true},
            run.loss, run.grad_corrected, run.grad_raw);
        return run;
    }

} // namespace

TEST(TrainingOpsCapability, VulkanConfigurationIsRejectedBeforeAllocation) {
    const auto before = lfs::core::alloc_counter::snapshot();
    lfs::core::param::TrainingParameters params;
    const auto dependencies = lfs::training::training_loader_dependencies(params);
    const auto reason = lfs::training::unavailable_training_reason(
        params, lfs::core::GpuBackend::Vulkan, dependencies);
    ASSERT_TRUE(reason.has_value());
    EXPECT_EQ(
        reason->rfind("Vulkan training is unavailable for this configuration.\nMissing families: ", 0),
        0u);
    EXPECT_NE(reason->find("Photometric"), std::string::npos);
    EXPECT_NE(reason->find("Fast"), std::string::npos);
    EXPECT_NE(reason->find("Mrnf"), std::string::npos);
    EXPECT_EQ(reason->back(), '.');
    EXPECT_FALSE(lfs::training::unavailable_training_reason(
        params, lfs::core::GpuBackend::CUDA, dependencies));
    const auto metal = lfs::training::unavailable_training_reason(
        params, lfs::core::GpuBackend::Metal, dependencies);
    ASSERT_TRUE(metal.has_value());
    EXPECT_EQ(metal->rfind("Metal training is unavailable for this configuration.", 0), 0u);
    EXPECT_EQ(lfs::core::alloc_counter::delta_since(before), 0u);
}

TEST(TrainingOpsCapability, EvaluationWithoutPhotometricFamilyFailsBeforeAllocation) {
    const auto metal = lfs::training::unavailable_training_family(
        lfs::core::GpuBackend::Metal, lfs::training::Family::Photometric);
    ASSERT_TRUE(metal.has_value());
    EXPECT_EQ(
        *metal,
        "Metal training is unavailable for this configuration.\nMissing families: Photometric.");
    const auto vulkan = lfs::training::unavailable_training_family(
        lfs::core::GpuBackend::Vulkan, lfs::training::Family::Photometric);
    ASSERT_TRUE(vulkan.has_value());
    EXPECT_EQ(
        *vulkan,
        "Vulkan training is unavailable for this configuration.\nMissing families: Photometric.");
    EXPECT_FALSE(lfs::training::unavailable_training_family(
        lfs::core::GpuBackend::CUDA, lfs::training::Family::Photometric));

    const auto backend = lfs::core::default_gpu_backend();
    if (lfs::training::training_ops(backend).photometric != nullptr) {
        return;
    }
    const auto expected = lfs::training::unavailable_training_family(
        backend, lfs::training::Family::Photometric);
    ASSERT_TRUE(expected.has_value());

    const auto before = lfs::core::alloc_counter::snapshot();
    {
        lfs::training::SSIM ssim(true);
        const Tensor predicted = Tensor::zeros({1, 3, 4, 4}, Device::CPU);
        const Tensor target = Tensor::zeros({1, 3, 4, 4}, Device::CPU);
        try {
            (void)ssim.compute(predicted, target);
            FAIL() << "SSIM compute should reject a backend without Photometric";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()), *expected);
        }
    }
    {
        const auto output = std::filesystem::temp_directory_path() / "lfs-tbo7-missing-photometric";
        std::filesystem::remove_all(output);
        lfs::core::param::TrainingParameters params;
        params.optimization.enable_eval = true;
        params.dataset.output_path = output;
        lfs::training::MetricsEvaluator evaluator(params);
        lfs::core::SplatData splat;
        lfs::core::Tensor background;
        try {
            (void)evaluator.evaluate(1, splat, nullptr, background);
            FAIL() << "evaluation should reject a backend without Photometric";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()), *expected);
        }
        std::filesystem::remove_all(output);
    }
    EXPECT_EQ(lfs::core::alloc_counter::delta_since(before), 0u);
}

class PhotometricOpsBytes : public lfs::test::CudaBackendTest {};

TEST_F(PhotometricOpsBytes, FixedInputsMatchKernelBytes) {
    constexpr int H = 32;
    constexpr int W = 40;
    constexpr float kWeight = 0.2f;
    const Tensor corrected = fixed_image(1, 3, H, W, 0.2f);
    const Tensor raw = fixed_image(1, 3, H, W, 0.05f);
    const Tensor target = fixed_image(1, 3, H, W, 0.4f);
    const Tensor mask = fixed_mask(H, W);

    {
        const size_t count = corrected.numel();
        const size_t blocks = std::min((count + 255) / 256, size_t{1024});
        Tensor grad = Tensor::empty(corrected.shape(), Device::GPU);
        Tensor loss = Tensor::zeros({1}, Device::GPU);
        Tensor scratch = Tensor::empty({blocks}, Device::GPU);
        lfs::core::pin_operands({&corrected, &target, &grad, &loss, &scratch});
        lfs::training::kernels::launch_fused_l1_loss(
            corrected.ptr<float>(), target.ptr<float>(), grad.ptr<float>(), loss.ptr<float>(),
            scratch.ptr<float>(), count, nullptr);
        const PhotoRun ops = evaluate_path(lfs::gpu_ops::PhotoPath::L1, 0.0f, corrected, {}, target, {});
        expect_identical(ops.loss, loss);
        expect_identical(ops.grad_corrected, grad);
        EXPECT_FALSE(ops.grad_raw.is_valid());
    }

    {
        lfs::training::kernels::SSIMWorkspace workspace;
        auto [loss_reference, ctx] = lfs::training::kernels::ssim_forward(corrected, target, workspace, true);
        const Tensor one = Tensor::full({1}, 1.0f, Device::GPU);
        const Tensor loss = one - loss_reference;
        const Tensor grad = lfs::training::kernels::ssim_backward(ctx, workspace, -1.0f);
        const PhotoRun ops = evaluate_path(lfs::gpu_ops::PhotoPath::SSIM, 1.0f, corrected, {}, target, {});
        expect_identical(ops.loss, loss);
        expect_identical(ops.grad_corrected, grad);
    }

    {
        lfs::training::kernels::FusedL1SSIMWorkspace workspace;
        auto [loss, ctx] = lfs::training::kernels::fused_l1_ssim_forward(
            corrected, target, kWeight, workspace, true);
        const Tensor grad = lfs::training::kernels::fused_l1_ssim_backward(ctx, workspace);
        const PhotoRun ops = evaluate_path(lfs::gpu_ops::PhotoPath::Fused, kWeight, corrected, {}, target, {});
        expect_identical(ops.loss, loss);
        expect_identical(ops.grad_corrected, grad);
        const PhotoRun again = evaluate_path(lfs::gpu_ops::PhotoPath::Fused, kWeight, corrected, {}, target, {});
        expect_identical(again.loss, ops.loss);
        expect_identical(again.grad_corrected, ops.grad_corrected);
    }

    {
        lfs::training::kernels::DecoupledFusedL1SSIMWorkspace workspace;
        auto [loss, ctx] = lfs::training::kernels::decoupled_fused_l1_ssim_forward(
            corrected, raw, target, kWeight, workspace, true);
        const auto grads = lfs::training::kernels::decoupled_fused_l1_ssim_backward(ctx, workspace);
        const PhotoRun ops = evaluate_path(lfs::gpu_ops::PhotoPath::Decoupled, kWeight, corrected, raw, target, {});
        expect_identical(ops.loss, loss);
        expect_identical(ops.grad_corrected, grads.grad_corrected);
        expect_identical(ops.grad_raw, grads.grad_raw);
    }

    {
        lfs::training::kernels::MaskedFusedL1SSIMWorkspace workspace;
        auto [loss, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
            corrected, target, mask, kWeight, workspace);
        const Tensor grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, workspace);
        const PhotoRun ops = evaluate_path(lfs::gpu_ops::PhotoPath::MaskedFused, kWeight, corrected, {}, target, mask);
        expect_identical(ops.loss, loss);
        expect_identical(ops.grad_corrected, grad);
    }

    {
        lfs::training::kernels::MaskedDecoupledFusedL1SSIMWorkspace workspace;
        auto [loss, ctx] = lfs::training::kernels::masked_decoupled_fused_l1_ssim_forward(
            corrected, raw, target, mask, kWeight, workspace);
        const auto grads = lfs::training::kernels::masked_decoupled_fused_l1_ssim_backward(ctx, workspace);
        const PhotoRun ops =
            evaluate_path(lfs::gpu_ops::PhotoPath::MaskedDecoupled, kWeight, corrected, raw, target, mask);
        expect_identical(ops.loss, loss);
        expect_identical(ops.grad_corrected, grads.grad_corrected);
        expect_identical(ops.grad_raw, grads.grad_raw);
    }
}

TEST_F(PhotometricOpsBytes, MetricAndErrorMapMatchKernels) {
    constexpr int H = 32;
    constexpr int W = 40;
    const Tensor predicted = fixed_image(1, 3, H, W, 0.3f);
    const Tensor target = fixed_image(1, 3, H, W, 0.6f);
    const auto& ops = lfs::training::cuda_photometric_ops();
    lfs::gpu_ops::PhotoSaved saved{.backend = ops.create()};

    auto [reference_value, reference_ctx] = lfs::training::kernels::ssim_forward(predicted, target, true);
    (void)reference_ctx;
    const Tensor metric = ops.metric(saved, predicted, target, false, true);
    expect_identical(metric, reference_value);

    const auto reference_maps = lfs::training::kernels::ssim_forward_map(predicted, target, false);
    const Tensor mapped = ops.metric(saved, predicted, target, true, false);
    expect_identical(mapped, reference_maps.ssim_value);
    expect_identical(saved.ssim_map, reference_maps.ssim_map);
    expect_identical(saved.cs_map, reference_maps.cs_map);

    lfs::training::kernels::SSIMMapWorkspace workspace;
    Tensor reference_error;
    lfs::training::kernels::ssim_error_map_forward(predicted, target, workspace, reference_error, true);
    Tensor ops_error;
    ops.error_map(saved, predicted, target, ops_error, true);
    expect_identical(ops_error, reference_error);

    Tensor reference_from_map = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::GPU);
    Tensor ops_from_map = Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::GPU);
    lfs::training::kernels::launch_ssim_to_error_map(reference_maps.ssim_map, reference_from_map);
    ops.map_to_error(reference_maps.ssim_map, ops_from_map);
    expect_identical(ops_from_map, reference_from_map);
}
