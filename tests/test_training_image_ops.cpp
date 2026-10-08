/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/training_image_cuda.hpp"
#include "lfs/training/ops/training_image_vulkan.hpp"
#include "training/kernels/camera_loss_heatmap.cuh"
#include "training/kernels/grad_alpha.hpp"
#include "training/kernels/image_kernels.hpp"
#include "training/kernels/roi_weight_map.hpp"
#include "training/losses/mask_loss.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace kernels = lfs::training::kernels;

    const auto& image_ops() {
        return *lfs::training::training_ops(lfs::core::GpuBackend::CUDA).training_image;
    }

    Tensor pattern(const lfs::core::TensorShape& shape) {
        std::vector<float> data(shape.elements());
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>((i * 7919 + 17) % 251) / 251.f;
        }
        return Tensor::from_vector(data, shape, Device::GPU);
    }

    std::vector<uint8_t> bytes(const Tensor& tensor) {
        const auto cpu = tensor.cpu().contiguous();
        const auto* data = static_cast<const uint8_t*>(cpu.data_ptr());
        return {data, data + cpu.bytes()};
    }

    void expect_same(const Tensor& actual, const Tensor& direct) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(bytes(actual), bytes(direct));
    }
} // namespace

TEST(TrainingImageRegistry, RequiresCudaTableForEveryTrainingConfiguration) {
    using namespace lfs::training;
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::CUDA).training_image, &cuda_training_image_ops());
    lfs::core::param::TrainingParameters params;
    EXPECT_TRUE(required_training_families(params, {}).test(static_cast<size_t>(Family::TrainingImage)));
    TrainingOps empty_cuda;
    FamilySet required;
    required.set(static_cast<size_t>(Family::TrainingImage));
    EXPECT_EQ(missing_training_families(empty_cuda, required), std::vector<std::string_view>{"TrainingImage"});
    for (const auto backend : {lfs::core::GpuBackend::Vulkan, lfs::core::GpuBackend::Metal}) {
        if (backend == lfs::core::GpuBackend::Vulkan) {
            EXPECT_EQ(training_ops(backend).training_image, &vulkan_training_image_ops());
            continue;
        }
        EXPECT_EQ(training_ops(backend).training_image, nullptr);
        const auto reason = unavailable_training_reason(params, backend, {});
        ASSERT_TRUE(reason.has_value());
        EXPECT_NE(reason->find("TrainingImage"), std::string::npos);
    }
}

class TrainingImageOpsBytes : public lfs::test::CudaBackendTest {
protected:
    void SetUp() override {
        LFS_CUDA_BACKEND_OR_RETURN();
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
        guard_.emplace(stream_);
    }
    void TearDown() override {
        if (stream_) {
            EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            guard_.reset();
            EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
        }
    }

private:
    cudaStream_t stream_ = nullptr;
    std::optional<lfs::core::CUDAStreamGuard> guard_;
};

TEST_F(TrainingImageOpsBytes, Heatmap) {
    auto loss = Tensor::full({1}, 0.375f, Device::GPU);
    auto latest = Tensor::full({7}, -1.f, Device::GPU);
    auto ema = latest.clone();
    auto direct_latest = latest.clone();
    auto direct_ema = ema.clone();
    const auto before = bytes(latest);
    for (int slot : {0, 6, 0}) {
        image_ops().heatmap(loss, latest, ema, slot, 0.2f);
        kernels::launch_update_camera_loss_heatmap(loss.ptr<float>(), slot, 0.2f,
                                                   direct_latest.ptr<float>(), direct_ema.ptr<float>(), 7,
                                                   lfs::core::getCurrentCUDAStream());
        loss.mul_(1.5f);
    }
    expect_same(latest, direct_latest);
    expect_same(ema, direct_ema);
    EXPECT_NE(bytes(latest), before);
}

TEST_F(TrainingImageOpsBytes, Roi) {
    const auto view = Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, Device::GPU);
    const auto position = Tensor::zeros({3}, Device::GPU);
    lfs::gpu_ops::RoiParams p{
        .image = {7, 9},
        .intrinsics = {6, 6, 4.5f, 3.5f},
        .world_to_cropbox = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.2f, 0.3f, 0, 1},
        .minimum = {-1, -1, 2},
        .maximum = {1, 1, 4},
        .outside_weight = 0.2f,
        .inverse = false};
    for (bool inverse : {false, true}) {
        p.inverse = inverse;
        auto actual = Tensor::full({7, 9}, -1.f, Device::GPU);
        auto direct = actual.clone();
        const auto before = bytes(actual);
        image_ops().roi(view, position, actual, p);
        kernels::launch_roi_weight_map(view.ptr<float>(), position.ptr<float>(), 6, 6, 4.5f, 3.5f, 9, 7,
                                       glm::make_mat4(p.world_to_cropbox.data()), glm::make_vec3(p.minimum.data()),
                                       glm::make_vec3(p.maximum.data()), inverse, 0.2f, direct.ptr<float>(),
                                       lfs::core::getCurrentCUDAStream());
        expect_same(actual, direct);
        EXPECT_NE(bytes(actual), before);
    }
}

TEST_F(TrainingImageOpsBytes, ResizeBackground) {
    const auto source = pattern({3, 13, 19});
    auto actual = Tensor::full({3, 7, 31}, -1.f, Device::GPU);
    auto direct = actual.clone();
    const auto before = bytes(actual);
    image_ops().resize_background(source, actual);
    kernels::launch_bilinear_resize_chw(source.ptr<float>(), direct.ptr<float>(), 3, 13, 19, 7, 31,
                                        lfs::core::getCurrentCUDAStream());
    expect_same(actual, direct);
    EXPECT_NE(bytes(actual), before);
}

TEST_F(TrainingImageOpsBytes, RandomBackground) {
    auto actual = Tensor::full({3, 17, 33}, -1.f, Device::GPU);
    auto direct = actual.clone();
    const auto before = bytes(actual);
    for (uint64_t seed : {0ULL, 42ULL, 0x100000001ULL}) {
        image_ops().random_background(actual, seed);
        kernels::launch_random_background(direct.ptr<float>(), 17, 33, seed, lfs::core::getCurrentCUDAStream());
        expect_same(actual, direct);
        EXPECT_NE(bytes(actual), before);
    }
}

TEST_F(TrainingImageOpsBytes, CannyFloatAndByte) {
    for (const auto dtype : {DataType::Float32, DataType::UInt8}) {
        auto source = pattern({3, 35, 37});
        if (dtype == DataType::UInt8) {
            source = (source * 255.f).to(dtype);
        }
        auto actual = Tensor::full({35, 37}, -1.f, Device::GPU);
        auto direct = actual.clone();
        const auto before = bytes(actual);
        image_ops().canny(source, actual);
        if (dtype == DataType::UInt8) {
            kernels::launch_fused_canny_edge_filter_chw(source.ptr<uint8_t>(), direct.ptr<float>(), 35, 37,
                                                        lfs::core::getCurrentCUDAStream());
        } else {
            kernels::launch_fused_canny_edge_filter_chw(source.ptr<float>(), direct.ptr<float>(), 35, 37,
                                                        lfs::core::getCurrentCUDAStream());
        }
        expect_same(actual, direct);
        EXPECT_NE(bytes(actual), before);
    }
}

TEST_F(TrainingImageOpsBytes, NormalizeScalarAndSkip) {
    for (float value : {2.f, 0.f, 1e-6f}) {
        auto actual = pattern({257});
        auto direct = actual.clone();
        auto scalar = Tensor::full({1}, value, Device::GPU);
        const auto before = bytes(actual);
        image_ops().normalize_scalar(actual, scalar, 1e-6f);
        kernels::launch_normalize_by_device_scalar(direct.ptr<float>(), direct.numel(), scalar.ptr<float>(), 1e-6f);
        expect_same(actual, direct);
        if (value > 1e-6f) {
            EXPECT_NE(bytes(actual), before);
        } else {
            EXPECT_EQ(bytes(actual), before);
        }
    }
}

TEST(TrainingImageEdgeMask, ExcludesPixelsBeforeNormalizingOnSelectedBackend) {
    constexpr size_t h = 35, w = 37;
    const auto image = pattern({3, h, w});
    const auto& ops = lfs::training::training_ops(lfs::core::default_gpu_backend());
    for (const auto dtype : {DataType::Float32, DataType::UInt8}) {
        for (const bool band : {false, true}) {
            std::vector<float> mask_values(h * w);
            for (size_t i = 0; i < mask_values.size(); ++i)
                mask_values[i] = i % 3 == 0 ? 0.f : (i % 3 == 1 ? 128.f : 255.f);
            auto mask = Tensor::from_vector(mask_values, {h, w}, Device::GPU);
            mask = dtype == DataType::UInt8 ? mask.to(dtype) : mask / 255.f;
            auto raw = Tensor::zeros({h, w}, Device::GPU);
            ops.training_image->canny(image, raw);
            auto expected = raw.cpu().to_vector();
            std::vector<float> positive;
            for (size_t i = 0; i < expected.size(); ++i) {
                if (mask_values[i] == 0.f || (band && mask_values[i] <= 250.f))
                    expected[i] = 0.f;
                if (expected[i] > 0.f)
                    positive.push_back(expected[i]);
            }
            ASSERT_FALSE(positive.empty());
            std::sort(positive.begin(), positive.end());
            const float median = positive[positive.size() / 2];
            auto actual = Tensor::zeros({h, w}, Device::GPU);
            lfs::training::losses::compute_edge_weight_map(image, mask, band, actual);
            const auto values = actual.cpu().to_vector();
            for (size_t i = 0; i < values.size(); ++i)
                EXPECT_NEAR(values[i], expected[i] / median, 1e-5f) << i;
        }
    }
}
