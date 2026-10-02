/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/pruning_kernels.hpp"
#include "lfs/training/ops/refine_cuda.hpp"
#include "lfs/training/ops/refine_vulkan.hpp"
#include "lfs/training/ops/registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

TEST(RefineVulkanMedian, MatchesPositiveUpperMedianAcrossDistributions) {
    using namespace lfs::core;
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan unavailable";
    GpuBackendScope scope(GpuBackend::Vulkan);
    const auto* ops = lfs::training::training_ops(GpuBackend::Vulkan).refine;
    ASSERT_NE(ops, nullptr);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<std::vector<float>> cases{{0.f, -2.f, nan}, {3.f}, {nan, -2.f, 0.f, 1.f, 9.f, 4.f, 4.f, inf}, {inf, inf, 1.f, -inf}, {1e-12f, 2e-12f, -1e-12f}};
    for (size_t count : {255u, 256u, 257u, 4097u, 1048576u}) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = i % 13 == 0 ? nan : float(int((i * 37) % 997) - 100) / 64.f;
        cases.push_back(std::move(values));
    }
    for (auto values : cases) {
        auto tensor = Tensor::from_vector(values, {values.size()}, Device::GPU);
        ops->normalize_positive_median(tensor);
        const auto host = tensor.cpu();
        std::vector<float> positives;
        for (float& value : values) {
            if (std::isnan(value))
                value = 0.f;
            if (value > 0.f)
                positives.push_back(value);
        }
        float median = 0.f;
        if (!positives.empty()) {
            auto middle = positives.begin() + positives.size() / 2;
            std::nth_element(positives.begin(), middle, positives.end());
            median = *middle;
        }
        const float* actual = host.ptr<float>();
        for (size_t i = 0; i < values.size(); ++i) {
            const float expected = positives.empty() ? 0.f : values[i] / std::max(median, 1e-9f);
            if (std::isnan(expected)) {
                ASSERT_TRUE(std::isnan(actual[i])) << "index " << i;
            } else {
                ASSERT_FLOAT_EQ(actual[i], expected) << "count " << values.size() << ", index " << i;
            }
        }
    }
}

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    namespace kernels = lfs::training::kernels;
    namespace pruning = lfs::training::pruning;

    Tensor pattern(const lfs::core::TensorShape& shape, float offset = 0.f) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = offset + static_cast<float>((i * 17 + 3) % 101) / 101.f;
        return Tensor::from_vector(values, shape, Device::GPU);
    }

    Tensor indices(const std::vector<int64_t>& values) {
        auto cpu = Tensor::empty({values.size()}, Device::CPU, DataType::Int64);
        std::copy(values.begin(), values.end(), cpu.ptr<int64_t>());
        return cpu.gpu();
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

    const auto& cuda_ops() {
        return *lfs::training::training_ops(lfs::core::GpuBackend::CUDA).refine;
    }

    struct Rows {
        Tensor means, rotations, scales, sh0, opacity;
        static Rows make(size_t n, bool flat) {
            return {pattern({n, 3}), pattern({n, 4}, 0.1f), pattern({n, 3}, -2.f),
                    pattern(flat ? lfs::core::TensorShape{n, 3} : lfs::core::TensorShape{n, 1, 3}),
                    pattern(flat ? lfs::core::TensorShape{n} : lfs::core::TensorShape{n, 1}, -0.5f)};
        }
        Rows clone() const { return {means.clone(), rotations.clone(), scales.clone(), sh0.clone(), opacity.clone()}; }
        lfs::gpu_ops::RefineOutputs out() { return {means, rotations, scales, sh0, opacity}; }
        lfs::gpu_ops::RefineInputs in() const { return {means, rotations, scales, sh0, opacity}; }
        void expect_same(const Rows& expected) const {
            same(means, expected.means);
            same(rotations, expected.rotations);
            same(scales, expected.scales);
            same(sh0, expected.sh0);
            same(opacity, expected.opacity);
        }
    };
} // namespace

TEST(RefineOpsCapability, ProvidesCudaAndVulkanFamilies) {
    using namespace lfs::training;
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::CUDA).refine, &cuda_refine_ops());
    FamilySet required;
    required.set(static_cast<size_t>(Family::Refine));
    EXPECT_EQ(missing_training_families(TrainingOps{}, required), std::vector<std::string_view>{"Refine"});
    EXPECT_EQ(training_ops(lfs::core::GpuBackend::Vulkan).refine, &vulkan_refine_ops());
    {
        auto backend = lfs::core::GpuBackend::Metal;
        EXPECT_EQ(training_ops(backend).refine, nullptr);
        const auto reason = unavailable_training_family(backend, Family::Refine);
        ASSERT_TRUE(reason.has_value());
        EXPECT_NE(reason->find("Missing families: Refine"), std::string::npos);
    }
}

class RefineOpsBytes : public lfs::test::CudaBackendTest {
protected:
    void SetUp() override {
        LFS_CUDA_BACKEND_OR_RETURN();
        Tensor::manual_seed(42);
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
    cudaStream_t stream_ = nullptr;
    std::optional<lfs::core::CUDAStreamGuard> guard_;
};

TEST_F(RefineOpsBytes, SplitAndFillBothRowLayouts) {
    for (bool flat : {false, true}) {
        auto parents = Rows::make(257, flat);
        auto reference = parents.clone();
        auto children = Rows::make(3, flat);
        auto expected_children = children.clone();
        auto split_ids = indices({0, 128, 256});
        auto before = bytes(parents.means);
        cuda_ops().split(parents.out(), children.out(), split_ids);
        kernels::launch_long_axis_split_gaussians_inplace(
            reference.means.ptr<float>(), reference.rotations.ptr<float>(), reference.scales.ptr<float>(),
            reference.sh0.ptr<float>(), nullptr, reference.opacity.ptr<float>(),
            expected_children.means.ptr<float>(), expected_children.rotations.ptr<float>(), expected_children.scales.ptr<float>(),
            expected_children.sh0.ptr<float>(), nullptr, expected_children.opacity.ptr<float>(), split_ids.ptr<int64_t>(), 3, 0, stream_);
        parents.expect_same(reference);
        children.expect_same(expected_children);
        EXPECT_NE(bytes(parents.means), before);

        auto destinations = indices({2, 127, 254});
        auto free = Tensor::ones({257}, Device::GPU, DataType::Bool);
        auto expected_free = free.clone();
        before = bytes(parents.means);
        cuda_ops().fill_slots(destinations, children.in(), parents.out(), free);
        kernels::launch_fill_free_slots_fused(
            destinations.ptr<int64_t>(), 3, expected_children.means.ptr<float>(), expected_children.rotations.ptr<float>(),
            expected_children.scales.ptr<float>(), expected_children.sh0.ptr<float>(), expected_children.opacity.ptr<float>(),
            reference.means.ptr<float>(), reference.rotations.ptr<float>(), reference.scales.ptr<float>(),
            reference.sh0.ptr<float>(), reference.opacity.ptr<float>(), flat ? 0 : 1, expected_free.ptr<bool>(), 257, stream_);
        parents.expect_same(reference);
        same(free, expected_free);
        EXPECT_NE(bytes(parents.means), before);
        EXPECT_EQ(free.count_nonzero(), 254);
    }
}

TEST_F(RefineOpsBytes, CountsWithOptionalInputs) {
    auto b0 = pattern({257}) > 0.5f;
    auto b1 = pattern({17}) > 0.25f;
    auto f0 = pattern({513}, -0.5f);
    auto f1 = pattern({129}, -0.75f);
    for (bool optional : {false, true}) {
        auto actual = Tensor::full({4}, -1.f, Device::GPU, DataType::Int64);
        auto expected = actual.clone();
        cuda_ops().counts(b0, optional ? b1 : Tensor{}, f0, optional ? f1 : Tensor{}, actual);
        kernels::launch_packed_refine_counts(b0.ptr<bool>(), 257, optional ? b1.ptr<bool>() : nullptr, optional ? 17 : 0,
                                             f0.ptr<float>(), 513, optional ? f1.ptr<float>() : nullptr, optional ? 129 : 0,
                                             expected.ptr<int64_t>(), stream_);
        same(actual, expected);
        const auto host_counts = actual.cpu();
        EXPECT_GT(host_counts.ptr<int64_t>()[0], 0);
    }
}

TEST_F(RefineOpsBytes, CountsPreservePendingSnapshotsLikeDirectAccess) {
    auto actual = pattern({4097});
    auto expected = actual.clone();
    auto* actual_data = actual.ptr<float>();
    auto* expected_data = expected.ptr<float>();
    const auto actual_snapshot = actual.add(1.0f);
    const auto expected_snapshot = expected.add(1.0f);
    ASSERT_TRUE(actual_snapshot.is_deferred());
    ASSERT_TRUE(expected_snapshot.is_deferred());
    auto counts = Tensor::zeros({4}, Device::GPU, DataType::Int64);
    auto expected_counts = counts.clone();

    cuda_ops().counts({}, {}, actual, {}, counts);
    kernels::launch_packed_refine_counts(nullptr, 0, nullptr, 0, expected.ptr<float>(), expected.numel(),
                                         nullptr, 0, expected_counts.ptr<int64_t>(), stream_);
    same(counts, expected_counts);
    // A producer may retain its raw pointer across the refinement read.
    ASSERT_EQ(cudaMemsetAsync(actual_data, 0, actual.bytes(), stream_), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(expected_data, 0, expected.bytes(), stream_), cudaSuccess);
    same(actual_snapshot, expected_snapshot);
}

TEST_F(RefineOpsBytes, PositiveMedianMixedAndNoPositiveValues) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const auto& values : {std::vector<float>{nan, 0.f, -2.f, 9.f, 1.f, 4.f, 4.f, 100.f},
                               std::vector<float>{nan, -2.f, 0.f}}) {
        auto actual = Tensor::from_vector(values, {values.size()}, Device::GPU);
        auto expected = actual.clone();
        const auto before = bytes(actual);
        cuda_ops().normalize_positive_median(actual);
        kernels::launch_normalize_by_positive_median(expected.ptr<float>(), expected.numel(), stream_);
        same(actual, expected);
        EXPECT_NE(bytes(actual), before);
    }
}

TEST_F(RefineOpsBytes, ClipAndOversizeWithOptionalPartialFrozenMask) {
    auto shares = pattern({257});
    auto error = pattern({257}, -0.2f);
    for (bool frozen : {false, true}) {
        auto mask = frozen ? pattern({129}) > 0.5f : Tensor{};
        auto scales = pattern({257, 3}, -2.f);
        auto expected_scales = scales.clone();
        const auto before = bytes(scales);
        auto scores = Tensor::full({257}, -1.f, Device::GPU);
        auto expected_scores = scores.clone();
        cuda_ops().clip_scales(scales, shares, mask, 0.3f);
        kernels::launch_clip_log_scale_by_screen_share(expected_scales.ptr<float>(), shares.ptr<float>(),
                                                       frozen ? mask.ptr<bool>() : nullptr, frozen ? 129 : 0, 0.3f, 257, stream_);
        cuda_ops().oversize_scores(error, shares, mask, scores, 0.3f);
        kernels::launch_oversize_split_scores(error.ptr<float>(), shares.ptr<float>(), frozen ? mask.ptr<bool>() : nullptr,
                                              frozen ? 129 : 0, expected_scores.ptr<float>(), 0.3f, 257, stream_);
        same(scales, expected_scales);
        same(scores, expected_scores);
        EXPECT_NE(bytes(scales), before);
        EXPECT_GT(scores.sum().item<float>(), 0.f);
    }
}

TEST_F(RefineOpsBytes, DeadAndRotationMasks) {
    auto opacity = pattern({257});
    auto rotations = pattern({257, 4});
    rotations.slice(0, 0, 1).zero_();
    auto dead = Tensor::zeros({257}, Device::GPU, DataType::Bool);
    auto rotation = dead.clone();
    auto expected_dead = dead.clone();
    auto expected_rotation = dead.clone();
    cuda_ops().dead_mask(opacity, rotations, dead, 0.2f);
    pruning::launch_compute_dead_mask(opacity.ptr<float>(), rotations.ptr<float>(), expected_dead.ptr<uint8_t>(), 257, 0.2f, stream_);
    cuda_ops().rotation_mask(rotations, rotation);
    pruning::launch_compute_near_zero_rotation_mask(rotations.ptr<float>(), expected_rotation.ptr<uint8_t>(), 257, stream_);
    same(dead, expected_dead);
    same(rotation, expected_rotation);
    EXPECT_GT(dead.count_nonzero(), 1);
    EXPECT_EQ(rotation.count_nonzero(), 1);
}
