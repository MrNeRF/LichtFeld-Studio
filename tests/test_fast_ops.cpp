/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "fast_raster_test_helpers.hpp"
#include "lfs/training/ops/fast_cuda.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/optimizer/adam_optimizer.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace {

    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    Camera make_camera(const int w, const int h) {
        std::vector<float> rotation = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> translation = {0, 0, 4};
        auto r = Tensor::from_blob(rotation.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::GPU);
        auto t = Tensor::from_blob(translation.data(), {3}, Device::CPU, DataType::Float32).to(Device::GPU);
        return Camera(r, t, 100.f, 100.f, w * 0.5f, h * 0.5f, Tensor(), Tensor(),
                      lfs::core::CameraModelType::PINHOLE, "test", "", std::filesystem::path{}, w, h, 0);
    }

    std::unique_ptr<SplatData> make_splat(const int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::GPU);
        auto cpu = means.to(Device::CPU);
        float* values = cpu.ptr<float>();
        for (int i = 0; i < n; ++i) {
            values[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
            values[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
            values[i * 3 + 2] = 0.0f;
        }
        means = cpu.to(Device::GPU);
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.5f, Device::GPU);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::GPU);
        auto scaling = Tensor::full({static_cast<size_t>(n), 3}, -2.0f, Device::GPU);
        std::vector<float> rotation(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rotation[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto quat = Tensor::from_blob(rotation.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                        .to(Device::GPU);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 2.0f, Device::GPU);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, quat, opacity, 1.0f);
    }

    void expect_same_bytes(const Tensor& actual, const Tensor& reference) {
        ASSERT_TRUE(actual.is_valid());
        ASSERT_TRUE(reference.is_valid());
        const auto actual_cpu = actual.cpu().contiguous();
        const auto reference_cpu = reference.cpu().contiguous();
        ASSERT_EQ(actual_cpu.bytes(), reference_cpu.bytes());
        EXPECT_EQ(0, std::memcmp(actual_cpu.data_ptr(), reference_cpu.data_ptr(), actual_cpu.bytes()));
    }

    float max_abs_diff(const Tensor& actual, const Tensor& reference) {
        const auto actual_cpu = actual.cpu().contiguous();
        const auto reference_cpu = reference.cpu().contiguous();
        EXPECT_EQ(actual_cpu.numel(), reference_cpu.numel());
        const float* a = actual_cpu.ptr<float>();
        const float* b = reference_cpu.ptr<float>();
        float diff = 0.f;
        for (size_t i = 0; i < actual_cpu.numel(); ++i) {
            diff = std::max(diff, std::fabs(a[i] - b[i]));
        }
        return diff;
    }

    class FastOpsRaster : public lfs::test::CudaBackendTest {
    protected:
        void SetUp() override {
            LFS_CUDA_BACKEND_OR_RETURN();
            bg_ = Tensor::zeros({3}, Device::GPU);
            camera_ = std::make_unique<Camera>(make_camera(32, 24));
            splat_ = make_splat(12);
        }

        void TearDown() override {
            if (IsSkipped()) {
                return;
            }
            splat_.reset();
            camera_.reset();
            lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
        }

        std::weak_ptr<Tensor> track_means_storage() {
            auto owner = std::make_shared<Tensor>(splat_->means());
            splat_->means() = Tensor::from_external_owner(
                owner->ptr<float>(), owner->shape(), Device::GPU, DataType::Float32, owner);
            return owner;
        }

        Tensor bg_;
        std::unique_ptr<Camera> camera_;
        std::unique_ptr<SplatData> splat_;
    };

} // namespace

TEST(TrainingOpsCapability, FastFamilyIsCudaOnly) {
    const auto& cuda = lfs::training::training_ops(lfs::core::GpuBackend::CUDA);
    const auto& vulkan = lfs::training::training_ops(lfs::core::GpuBackend::Vulkan);
    const auto& metal = lfs::training::training_ops(lfs::core::GpuBackend::Metal);
    EXPECT_NE(cuda.fast, nullptr);
    EXPECT_EQ(vulkan.fast, nullptr);
    EXPECT_EQ(metal.fast, nullptr);
    EXPECT_FALSE(lfs::training::unavailable_training_family(
        lfs::core::GpuBackend::CUDA, lfs::training::Family::Fast));
    const auto reason = lfs::training::unavailable_training_family(
        lfs::core::GpuBackend::Vulkan, lfs::training::Family::Fast);
    ASSERT_TRUE(reason.has_value());
    EXPECT_NE(reason->find("Fast"), std::string::npos);

    lfs::core::param::TrainingParameters params;
    params.optimization.set_raster_backend(lfs::core::param::RasterBackendId::ThreeDGS);
    const auto missing = lfs::training::unavailable_training_reason(
        params, lfs::core::GpuBackend::Vulkan, lfs::training::training_loader_dependencies(params));
    ASSERT_TRUE(missing.has_value());
    EXPECT_NE(missing->find("Fast"), std::string::npos);
}

TEST_F(FastOpsRaster, ForwardRepeatsExactly) {
    auto reference = lfs::training::fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(reference.has_value()) << std::string(reference.error().user_message());
    const auto reference_image = reference->first.image.clone();
    const auto reference_alpha = reference->first.alpha.clone();
    reference->second.release_forward_context();

    const auto& ops = lfs::training::cuda_fast_ops();
    lfs::gpu_ops::FastSaved saved{.backend = ops.create()};
    lfs::training::RenderOutput output;
    const auto result = lfs::training::fast_render(
        ops, saved, *camera_, *splat_, bg_, 0, 0, 0, 0, false, {}, false, true, output);
    ASSERT_EQ(result.code, lfs::gpu_ops::RasterResult::Code::Success);
    EXPECT_TRUE(result.has_work);
    expect_same_bytes(output.image, reference_image);
    expect_same_bytes(output.alpha, reference_alpha);
    ops.release(saved);
}

TEST_F(FastOpsRaster, ReleaseDropsModelStorage) {
    const auto storage = track_means_storage();
    auto forward = lfs::training::fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    splat_->means() = {};
    EXPECT_FALSE(storage.expired());
    forward->second.release_forward_context();
    EXPECT_TRUE(storage.expired());
}

TEST_F(FastOpsRaster, ZeroGradientLeavesParametersUnchanged) {
    const auto storage = track_means_storage();
    const auto means = splat_->means().clone();
    const auto scales = splat_->scaling_raw().clone();
    const auto opacity = splat_->opacity_raw().clone();
    lfs::training::AdamConfig config{.lr = 1e-3f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    lfs::training::AdamOptimizer ops_opt(*splat_, config);
    ops_opt.allocate_gradients();
    ops_opt.zero_grad(1);
    const auto& ops = lfs::training::cuda_fast_ops();
    lfs::gpu_ops::FastSaved saved{.backend = ops.create()};
    lfs::training::RenderOutput output;
    const auto forward = lfs::training::fast_render(
        ops, saved, *camera_, *splat_, bg_, 0, 0, 0, 0, false, {}, false, true, output);
    ASSERT_EQ(forward.code, lfs::gpu_ops::RasterResult::Code::Success);
    auto grad = Tensor::zeros_like(output.image);
    const auto prepared = ops_opt.prepare_fastgs_fused_adam(1, lfs::core::getCurrentCUDAStream());
    Tensor none;
    ops.backward(
        saved,
        {.image = grad, .alpha = none, .depth = none, .normal = none},
        splat_->_densification_info,
        none,
        none,
        none,
        prepared,
        DensificationType::None);
    if (lfs::training::fastgs_adam_enabled(prepared)) {
        ops_opt.commit_fastgs_fused_adam(1);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_same_bytes(splat_->means(), means);
    expect_same_bytes(splat_->scaling_raw(), scales);
    expect_same_bytes(splat_->opacity_raw(), opacity);
    splat_->means() = {};
    EXPECT_TRUE(storage.expired());
}

TEST_F(FastOpsRaster, NonzeroBackwardStaysFinite) {
    lfs::training::AdamConfig config{.lr = 1e-2f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    lfs::training::AdamOptimizer opt(*splat_, config);
    opt.allocate_gradients();
    opt.zero_grad(1);
    auto forward = lfs::training::fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value()) << std::string(forward.error().user_message());
    const auto before = splat_->means().clone();
    auto grad = Tensor::full_like(forward->first.image, 0.05f);
    lfs::training::fast_rasterize_backward(
        forward->second, grad, *splat_, opt, {}, {}, DensificationType::None, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto after = splat_->means();
    EXPECT_LT(max_abs_diff(after, before), 1.f);
    const auto cpu = after.cpu().contiguous();
    const float* values = cpu.ptr<float>();
    for (size_t i = 0; i < cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(values[i])) << "means[" << i << "]";
    }
}
