/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/tensor.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "lfs/training/ops/mrnf_cuda.hpp"
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

TEST(MrnfOpsCapability, OnlyCudaProvidesTheFamily) {
    using namespace lfs::training;
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::CUDA).mrnf, &cuda_mrnf_ops());
    EXPECT_FALSE(unavailable_training_family(lfs::core::GpuBackend::CUDA, Family::Mrnf));
    FamilySet required;
    required.set(static_cast<size_t>(Family::Mrnf));
    EXPECT_EQ(missing_training_families(TrainingOps{}, required), std::vector<std::string_view>{"Mrnf"});
    for (auto backend : {lfs::core::GpuBackend::Vulkan, lfs::core::GpuBackend::Metal}) {
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
        Tensor far;
        if (masked) {
            far = mask_of(n, 3);
        }
        kernels::launch_mrnf_decay(
            raw_b.ptr<float>(), scales_b.ptr<float>(),
            masked ? frozen.ptr<bool>() : nullptr, masked ? n : 0,
            masked ? far.ptr<bool>() : nullptr, masked ? n : 0,
            0.02f, 0.01f, masked ? 0.25f : 1.f, 0.4f, n);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cuda_ops().decay(raw, log_scales, frozen, far,
                         {.opacity_decay = 0.02f, .scale_decay = 0.01f, .far_decay_scale = masked ? 0.25f : 1.f, .train_t = 0.4f});
        same(raw, raw_b);
        same(log_scales, scales_b);
        changed(raw, raw_before);
    }
}

TEST_F(MrnfOpsBytes, BoundsAndMedianMatchLaunchers) {
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

    auto scales = pattern({n, 3}, 0.8f, 4);
    float direct_median = -1.f;
    bool direct_valid = false;
    kernels::launch_median_geomean_extent(scales.ptr<float>(), n, &direct_median, &direct_valid);
    const auto extent = cuda_ops().median_extent(scales);
    same_float(extent.value, direct_median);
    EXPECT_EQ(extent.valid, direct_valid);
    EXPECT_TRUE(extent.valid);
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

TEST_F(MrnfOpsBytes, FoldProjectAndErrorMatchLaunchers) {
    constexpr size_t n = 64;
    auto vis = pattern({n}, 1.f, 1).abs();
    auto weight = pattern({n}, 0.2f, 2);
    auto dens = pattern({2, n}, 0.5f, 3);
    auto ratio = pattern({n}, 0.1f, 4);
    auto vis_b = vis.clone();
    auto weight_b = weight.clone();
    auto dens_b = dens.clone();
    auto ratio_b = ratio.clone();
    const auto dens_before = bytes(dens);
    kernels::launch_fold_densification_and_zero(
        vis_b.ptr<float>(), weight_b.ptr<float>(), dens_b.ptr<float>(), n, nullptr, 2, ratio_b.ptr<float>(), 0.75f);
    cuda_ops().fold(vis, weight, dens, ratio, 0.75f);
    same(vis, vis_b);
    same(weight, weight_b);
    same(dens, dens_b);
    same(ratio, ratio_b);
    changed(dens, dens_before);
    Tensor no_ratio;
    kernels::launch_fold_densification_and_zero(
        vis_b.ptr<float>(), weight_b.ptr<float>(), dens_b.ptr<float>(), n, nullptr, 2, nullptr, 0.f);
    cuda_ops().fold(vis, weight, dens, no_ratio, 0.f);
    same(vis, vis_b);
    same(weight, weight_b);
    same(dens, dens_b);

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

    auto means = pattern({n, 3}, 1.f, 12);
    std::vector<float> view{
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 2.f,
        0.f, 0.f, 0.f, 1.f};
    auto w2c = Tensor::from_vector(view, {4, 4}, Device::GPU);
    auto means2d = Tensor::zeros({n, 2}, Device::GPU);
    auto radii = Tensor::zeros({n}, Device::GPU);
    auto means2d_b = means2d.clone();
    auto radii_b = radii.clone();
    const ops::ProjectParams project{.image = {.h = 8, .w = 12}, .intrinsics = {.fx = 20.f, .fy = 18.f, .cx = 6.f, .cy = 4.f}, .near_plane = 0.01f};
    kernels::launch_project_visible_centers(
        means.ptr<float>(), w2c.ptr<float>(), 20.f, 18.f, 6.f, 4.f, 12, 8, 0.01f,
        means2d_b.ptr<float>(), radii_b.ptr<float>(), n);
    cuda_ops().project_centers(means, w2c, means2d, radii, project);
    same(means2d, means2d_b);
    same(radii, radii_b);
    changed(means2d, bytes(Tensor::zeros({n, 2}, Device::GPU)));

    auto error = pattern({8, 12}, 1.f, 15).abs();
    auto scores = Tensor::zeros({n}, Device::GPU);
    auto scores_b = scores.clone();
    kernels::launch_gather_center_error(
        means2d_b.ptr<float>(), radii_b.ptr<float>(), error.ptr<float>(), 12, 8, scores_b.ptr<float>(), n);
    cuda_ops().gather_center_error(means2d, radii, error, scores);
    same(scores, scores_b);

    auto far = Tensor::zeros_bool({n}, Device::GPU);
    auto far_b = far.clone();
    kernels::launch_far_field_mask(means.ptr<float>(), 0.1f, -0.2f, 0.3f, 1.5f, far_b.ptr<bool>(), n);
    cuda_ops().far_mask(means, far, {0.1f, -0.2f, 0.3f}, 1.5f);
    same(far, far_b);
    changed(far, bytes(Tensor::zeros_bool({n}, Device::GPU)));
}

TEST_F(MrnfOpsBytes, SeedsMedianAndStarvationMatchLaunchers) {
    constexpr int height = 6;
    constexpr int width = 8;
    constexpr size_t hw = static_cast<size_t>(height * width);
    auto predicted = pattern({3, height, width}, 1.f, 1);
    auto target = pattern({3, height, width}, 0.7f, 2);
    auto error = Tensor::zeros({height, width}, Device::GPU);
    auto error_b = error.clone();
    kernels::launch_mean_abs_error_hw(
        predicted.ptr<float>(), target.ptr<float>(), 3, height, width, error_b.ptr<float>());
    cuda_ops().mean_abs_error(predicted, target, error);
    same(error, error_b);
    changed(error, bytes(Tensor::zeros({height, width}, Device::GPU)));

    auto alpha = pattern({hw}, 0.5f, 3).abs().clamp(0.f, 1.f);
    auto weights = error.clone().reshape({hw});
    auto weights_b = error_b.clone().reshape({hw});
    kernels::launch_seed_weights_from_error_alpha(
        weights_b.ptr<float>(), alpha.ptr<float>(), weights_b.ptr<float>(), hw);
    cuda_ops().seed_weights(weights, alpha, weights);
    same(weights, weights_b);

    constexpr size_t k = 5;
    std::vector<int64_t> pixels{1, 4, 7, 20, 40};
    auto cpu = Tensor::empty({k}, Device::CPU, DataType::Int64);
    for (size_t i = 0; i < k; ++i) {
        cpu.ptr<int64_t>()[i] = pixels[i];
    }
    auto indices = cpu.gpu();
    auto depth = pattern({hw}, 2.f, 6).abs();
    auto rgb = Tensor::zeros({k, 3}, Device::GPU);
    auto out_alpha = Tensor::zeros({k}, Device::GPU);
    auto out_depth = Tensor::zeros({k}, Device::GPU);
    auto rgb_b = rgb.clone();
    auto alpha_b = out_alpha.clone();
    auto depth_b = out_depth.clone();
    kernels::launch_gather_seed_payloads(
        indices.ptr<int64_t>(), k, hw, target.ptr<float>(), 3, alpha.ptr<float>(), depth.ptr<float>(),
        rgb_b.ptr<float>(), alpha_b.ptr<float>(), depth_b.ptr<float>());
    cuda_ops().gather_seeds(indices, target, alpha, depth, rgb, out_alpha, out_depth);
    same(rgb, rgb_b);
    same(out_alpha, alpha_b);
    same(out_depth, depth_b);
    changed(rgb, bytes(Tensor::zeros({k, 3}, Device::GPU)));

    constexpr size_t count = 33;
    auto values = pattern({count}, 3.f, 9).abs();
    const float direct = kernels::launch_sorted_median(values.ptr<float>(), values.numel());
    const float actual = cuda_ops().sorted_median(values);
    same_float(actual, direct);
    EXPECT_GT(actual, 0.f);

    auto starved = pattern({count}, 1.f, 10).abs();
    auto vis = pattern({count}, 2.f, 11).abs();
    auto starved_b = starved.clone();
    const auto starved_before = bytes(starved);
    kernels::launch_apply_explore_starvation_weights(starved_b.ptr<float>(), vis.ptr<float>(), vis.numel(), direct);
    cuda_ops().starvation_weights(starved, vis, actual);
    same(starved, starved_b);
    changed(starved, starved_before);
}
