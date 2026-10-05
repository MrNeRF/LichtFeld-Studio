/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "lfs/training/ops/mrnf_cuda.hpp"
#include "lfs/training/ops/mrnf_vulkan.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/refine_scratch.hpp"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace kernels = lfs::training::mrnf_strategy;
    namespace ops = lfs::gpu_ops;

    constexpr uint64_t kSeed = 0x4d524e46ull;

    Tensor pattern(const lfs::core::TensorShape& shape, const float scale, const int seed) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i) {
            const auto k = static_cast<int>((i * 17u + static_cast<size_t>(seed) * 13u) % 97u);
            values[i] = scale * (static_cast<float>(k) / 48.f - 1.f);
        }
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor mask_of(const size_t n, const size_t period) {
        std::vector<bool> values(n);
        for (size_t i = 0; i < n; ++i) {
            values[i] = i % period == 0;
        }
        return Tensor::from_vector(values, {n}, Device::GPU);
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

    void changed(const Tensor& tensor, const std::vector<uint8_t>& before) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_NE(bytes(tensor), before);
    }

    void same_float(const float actual, const float expected) {
        uint32_t a = 0;
        uint32_t b = 0;
        std::memcpy(&a, &actual, sizeof(a));
        std::memcpy(&b, &expected, sizeof(b));
        EXPECT_EQ(a, b);
    }

    const ops::MrnfOps& cuda_ops() {
        return *lfs::training::training_ops(lfs::core::GpuBackend::CUDA).mrnf;
    }

    size_t positive_count(const Tensor& weights) {
        const auto host = weights.cpu();
        size_t count = 0;
        for (size_t i = 0; i < host.numel(); ++i) {
            count += host.ptr<float>()[i] > 0.f ? 1 : 0;
        }
        return count;
    }
} // namespace

TEST(MrnfOpsCapability, ProvidesCudaAndVulkanFamilies) {
    using namespace lfs::training;
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::CUDA).mrnf, &cuda_mrnf_ops());
    EXPECT_FALSE(unavailable_training_family(lfs::core::GpuBackend::CUDA, Family::Mrnf));
    FamilySet required;
    required.set(static_cast<size_t>(Family::Mrnf));
    EXPECT_EQ(missing_training_families(TrainingOps{}, required), std::vector<std::string_view>{"Mrnf"});
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::Vulkan).mrnf, &vulkan_mrnf_ops());
    for (auto backend : {lfs::core::GpuBackend::Metal}) {
        EXPECT_EQ(training_ops(backend).mrnf, nullptr);
        const auto reason = unavailable_training_family(backend, Family::Mrnf);
        ASSERT_TRUE(reason.has_value());
        EXPECT_NE(reason->find("Missing families: Mrnf"), std::string::npos);
        for (const auto* strategy : {"mrnf", "igs+"}) {
            lfs::core::param::TrainingParameters params;
            params.optimization.strategy = strategy;
            const auto training_reason = unavailable_training_reason(params, backend, {});
            ASSERT_TRUE(training_reason.has_value());
            EXPECT_NE(training_reason->find("Mrnf"), std::string::npos);
        }
    }
}

class MrnfOpsBytes : public lfs::test::CudaBackendTest {
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

TEST_F(MrnfOpsBytes, NoiseAndDecayMatchLaunchers) {
    constexpr size_t n = 257;
    auto opacity = pattern({n}, 1.f, 3) - 8.f;
    auto scales = pattern({n, 3}, 1.f, 5);
    auto visibility = pattern({n}, 1.f, 7).abs() + 1.f;
    for (bool masked : {false, true}) {
        auto means = pattern({n, 3}, 0.5f, 1);
        auto direct = means.clone();
        const auto before = bytes(means);
        Tensor frozen;
        if (masked) {
            frozen = mask_of(n, 4);
        }
        kernels::launch_mrnf_noise_injection(
            direct.ptr<float>(), opacity.ptr<float>(), visibility.ptr<float>(),
            masked ? frozen.ptr<bool>() : nullptr, masked ? n : 0,
            1.f, 4.f, 1.f, n, kSeed);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cuda_ops().noise(means, opacity, visibility, frozen, {.seed = kSeed, .lr_mean = 1.f, .noise_weight = 4.f, .median_scale = 1.f});
        same(means, direct);
        changed(means, before);

        auto raw = pattern({n}, 1.5f, 9);
        auto log_scales = pattern({n, 3}, 0.4f, 11);
        auto raw_b = raw.clone();
        auto scales_b = log_scales.clone();
        const auto raw_before = bytes(raw);
        kernels::launch_mrnf_decay(
            raw_b.ptr<float>(), scales_b.ptr<float>(),
            masked ? frozen.ptr<bool>() : nullptr, masked ? n : 0,
            0.02f, 0.01f, 0.4f, n);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cuda_ops().decay(raw, log_scales, frozen,
                         {.opacity_decay = 0.02f, .scale_decay = 0.01f, .train_t = 0.4f});
        same(raw, raw_b);
        same(log_scales, scales_b);
        changed(raw, raw_before);
    }
}

TEST_F(MrnfOpsBytes, BoundsMatchLauncher) {
    constexpr size_t n = 129;
    auto means = pattern({n, 3}, 3.f, 2);
    kernels::MRNFBounds direct{};
    kernels::launch_percentile_bounds(means.ptr<float>(), n, 0.8f, &direct);
    const auto actual = cuda_ops().percentile_bounds(means, 0.8f);
    for (int axis = 0; axis < 3; ++axis) {
        same_float(actual.center[axis], direct.center[axis]);
        same_float(actual.extent[axis], direct.extent[axis]);
    }
    same_float(actual.median_size, direct.median_size);
    same_float(actual.max_extent, direct.max_extent);
    EXPECT_GT(actual.max_extent, 0.f);
}

TEST_F(MrnfOpsBytes, GumbelMatchesLauncher) {
    constexpr size_t n = 257;
    auto weights = pattern({n}, 1.f, 8).abs();
    auto host = weights.cpu();
    for (size_t i = 0; i < n; i += 5) {
        host.ptr<float>()[i] = 0.f;
    }
    weights = host.gpu();
    const size_t nnz = positive_count(weights);
    const ops::GumbelParams cases[] = {
        {.seed = kSeed, .known_nnz = 0, .compact_sparse = true},
        {.seed = kSeed, .known_nnz = nnz, .compact_sparse = true},
        {.seed = kSeed + 9, .known_nnz = nnz, .compact_sparse = false},
    };
    for (const auto& params : cases) {
        for (bool use_scratch : {false, true}) {
            constexpr size_t k = 17;
            auto actual = Tensor::empty({k}, Device::GPU, DataType::Int64);
            auto direct = actual.clone();
            lfs::training::GumbelTopKScratch scratch_a;
            lfs::training::GumbelTopKScratch scratch_b;
            kernels::launch_gumbel_topk(
                weights.ptr<float>(), n, k, params.seed, direct.ptr<int64_t>(), nullptr,
                params.compact_sparse, use_scratch ? &scratch_b : nullptr, params.known_nnz);
            cuda_ops().gumbel(use_scratch ? &scratch_a : nullptr, weights, actual, params);
            same(actual, direct);
            changed(actual, bytes(Tensor::zeros({k}, Device::GPU, DataType::Int64)));
        }
    }
    auto all = Tensor::empty({n}, Device::GPU, DataType::Int64);
    auto all_direct = all.clone();
    kernels::launch_gumbel_topk(weights.ptr<float>(), n, n, kSeed, all_direct.ptr<int64_t>());
    cuda_ops().gumbel(nullptr, weights, all, {.seed = kSeed});
    same(all, all_direct);
}

TEST_F(MrnfOpsBytes, FoldErrorMatchesLauncher) {
    constexpr size_t n = 64;
    auto max_a = pattern({n}, 0.3f, 6);
    auto max_b = max_a.clone();
    auto err_a = pattern({2, n}, 0.8f, 7);
    auto err_b = err_a.clone();
    const auto err_before = bytes(err_a);
    kernels::launch_fold_densification_error_and_zero(max_b.ptr<float>(), err_b.ptr<float>(), n);
    cuda_ops().fold_error(max_a, err_a);
    same(max_a, max_b);
    same(err_a, err_b);
    changed(err_a, err_before);
}
