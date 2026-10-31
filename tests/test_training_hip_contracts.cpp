/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/tensor_execution.hpp"
#include "lfs/training/ops/registry.hpp"
#include "tensor_backend_contract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// These providers isolate the production registry from unrelated trainer
// implementations. They test table identity, not native CUDA/Metal/Vulkan math.
namespace lfs::training {
    const TrainingOps& cuda_training_ops_table() {
        static const TrainingOps table{.backend = core::GpuBackend::CUDA};
        return table;
    }
    const TrainingOps& vulkan_training_ops_table() {
        static const TrainingOps table{.backend = core::GpuBackend::Vulkan};
        return table;
    }
    const TrainingOps& metal_training_ops_table() {
        static const TrainingOps table{.backend = core::GpuBackend::Metal};
        return table;
    }
} // namespace lfs::training

namespace {
    using namespace lfs::core;
    using namespace lfs::training;
    namespace ops = lfs::gpu_ops;

    TEST(TrainingBackendRegistry, ExistingBackendDispatchRetainsItsTable) {
        EXPECT_EQ(&training_ops(GpuBackend::CUDA), &cuda_training_ops_table());
        EXPECT_EQ(&training_ops(GpuBackend::Vulkan), &vulkan_training_ops_table());
        EXPECT_EQ(&training_ops(GpuBackend::Metal), &metal_training_ops_table());
    }

    TEST(TrainingBackendRegistry, HipNeverUsesTheVulkanTable) {
        const auto before = alloc_counter::snapshot();
        const bool live = gpu_backend_live(GpuBackend::HIP);
        const auto& hip = training_ops(GpuBackend::HIP);
        EXPECT_EQ(hip.backend, GpuBackend::HIP);
        EXPECT_NE(&hip, &training_ops(GpuBackend::Vulkan));
        EXPECT_EQ(&hip, &training_ops(GpuBackend::HIP));
        EXPECT_EQ(gpu_backend_live(GpuBackend::HIP), live);
        EXPECT_EQ(alloc_counter::delta_since(before), 0u);
    }

    TEST(TrainingBackendRegistry, HipAdvertisesOnlyImplementedFamilies) {
        const auto& hip = training_ops(GpuBackend::HIP);
        FamilySet required;
        required.set();
        const auto missing = missing_training_families(hip, required);
        for (size_t i = 0; i < static_cast<size_t>(Family::Count); ++i) {
            const auto family = static_cast<Family>(i);
            const bool implemented = gpu_backend_compiled(GpuBackend::HIP) &&
                                     (family == Family::ExtraLoss || family == Family::SharedImage);
            EXPECT_EQ(std::ranges::find(missing, training_family_name(family)) == missing.end(), implemented)
                << training_family_name(family);
        }
        EXPECT_EQ(hip.extra_loss != nullptr, gpu_backend_compiled(GpuBackend::HIP));
        EXPECT_EQ(hip.shared_image != nullptr, gpu_backend_compiled(GpuBackend::HIP));
    }

    TEST(TrainingBackendRegistry, MissingHipTrainingIsReportedExplicitly) {
        FamilySet required;
        required.set(static_cast<size_t>(Family::Photometric));
        required.set(static_cast<size_t>(Family::Adam));
        const auto missing = missing_training_families(training_ops(GpuBackend::HIP), required);
        EXPECT_EQ(missing, (std::vector<std::string_view>{"Photometric", "Adam"}));
        EXPECT_EQ(training_unavailable_message(GpuBackend::HIP, missing),
                  "HIP training is unavailable for this configuration.\nMissing families: Photometric, Adam.");
        EXPECT_TRUE(unavailable_training_family(GpuBackend::HIP, Family::Photometric).has_value());
        EXPECT_EQ(unavailable_training_family(GpuBackend::HIP, Family::ExtraLoss).has_value(),
                  !gpu_backend_compiled(GpuBackend::HIP));
    }

#if defined(LFS_TENSOR_HIP) && LFS_TENSOR_HIP
    class HipTrainingPrimitives : public lfs::test::TensorBackendContract {
    protected:
        const ops::ExtraLossOps& regularizer() const { return *training_ops(GpuBackend::HIP).extra_loss; }

        void check(ops::Regularizer kind, size_t n, float weight, bool backward) {
            const size_t attributes = kind == ops::Regularizer::Scale ? 3 : 1;
            std::vector<float> data(n * attributes);
            for (size_t i = 0; i < data.size(); ++i)
                data[i] = static_cast<float>(static_cast<int>((i * 79 + 31) % 257) - 128) / 32.f;
            const auto raw = Tensor::from_vector(data, {n, attributes}, Device::GPU);
            auto gradient = backward ? Tensor::full(raw.shape(), 0.17f, Device::GPU) : Tensor{};
            auto loss = Tensor::full({1}, -9.f, Device::GPU);
            auto scratch = Tensor::full({5}, -3.f, Device::GPU);
            const auto* grad_pointer = backward ? gradient.data_ptr() : nullptr;
            const auto* loss_pointer = loss.data_ptr();
            // Two calls must accumulate into the existing gradient, not replace it.
            for (int repeat = 1; repeat <= 2; ++repeat) {
                regularizer().regularize(raw, gradient, loss, scratch, kind, weight);
                double sum = 0.;
                const auto actual = backward ? gradient.to_vector() : std::vector<float>{};
                for (size_t i = 0; i < data.size(); ++i) {
                    const double x = data[i];
                    const double value = kind == ops::Regularizer::Scale ? std::exp(x) : 1. / (1. + std::exp(-x));
                    sum += value;
                    if (backward) {
                        const double e = std::exp(-std::abs(x));
                        const double derivative = kind == ops::Regularizer::Scale ? value : e / ((1. + e) * (1. + e));
                        const double expected = 0.17f + repeat * weight / static_cast<double>(data.size()) * derivative;
                        EXPECT_NEAR(actual[i], expected, 2e-6 * std::max(1., std::abs(expected))) << i;
                    }
                }
                const double expected_loss = weight * sum / static_cast<double>(data.size());
                EXPECT_NEAR(loss.item<float>(), expected_loss, 2e-5 * std::max(1., std::abs(expected_loss)));
            }
            EXPECT_EQ(loss.data_ptr(), loss_pointer);
            if (backward)
                EXPECT_EQ(gradient.data_ptr(), grad_pointer);
            else
                EXPECT_FALSE(gradient.is_valid());
            EXPECT_EQ(raw.to_vector(), data);
            EXPECT_EQ(scratch.to_vector(), std::vector<float>(5, -3.f));
        }
    };

    TEST_P(HipTrainingPrimitives, RegularizersMatchIndependentLossAndAccumulatedGradient) {
        for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity})
            for (const auto n : {1u, 257u, 1027u})
                for (const float weight : {0.013f, -0.7f})
                    for (const bool backward : {false, true})
                        check(kind, n, weight, backward);
    }

    TEST_P(HipTrainingPrimitives, OpacityDerivativeRemainsNonzeroForSaturatedFiniteLogits) {
        const std::vector<float> data{-80.f, -20.f, 0.f, 20.f, 80.f};
        const auto raw = Tensor::from_vector(data, {data.size()}, Device::GPU);
        auto gradient = Tensor::zeros(raw.shape(), Device::GPU);
        auto loss = Tensor::zeros({1}, Device::GPU);
        Tensor scratch;
        regularizer().regularize(raw, gradient, loss, scratch, ops::Regularizer::Opacity, 1.f);
        const auto actual = gradient.to_vector();
        for (size_t i = 0; i < data.size(); ++i) {
            const double e = std::exp(-std::abs(static_cast<double>(data[i])));
            const double expected = e / ((1. + e) * (1. + e)) / data.size();
            EXPECT_GT(actual[i], 0.f);
            EXPECT_NEAR(actual[i], expected, expected * 5e-5) << i;
        }
    }

    TEST_P(HipTrainingPrimitives, EmptyInputsAndZeroWeightLeaveOutputsUnchanged) {
        for (const auto n : {0u, 5u}) {
            const auto raw = Tensor::zeros({n}, Device::GPU);
            auto gradient = Tensor::full({n}, 2.f, Device::GPU);
            auto loss = Tensor::full({1}, -9.f, Device::GPU);
            Tensor scratch;
            for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity}) {
                regularizer().regularize(raw, gradient, loss, scratch, kind, n == 0 ? 1.f : 0.f);
                EXPECT_EQ(loss.to_vector(), std::vector<float>{-9.f});
                EXPECT_EQ(gradient.to_vector(), std::vector<float>(n, 2.f));
            }
        }
    }

    TEST_P(HipTrainingPrimitives, RejectsForeignPlacementAndMalformedBindingsBeforeWrites) {
        const auto raw = Tensor::full({3}, 0.3f, Device::GPU);
        auto gradient = Tensor::full({3}, 2.f, Device::GPU);
        auto loss = Tensor::full({1}, -9.f, Device::GPU);
        Tensor scratch;
        const auto cpu_raw = raw.cpu();
        auto cpu_gradient = gradient.cpu();
        auto cpu_loss = loss.cpu();
        auto wrong_shape = Tensor::zeros({1, 3}, Device::GPU);
        auto wrong_loss = Tensor::zeros({2}, Device::GPU);
        const auto wrong_dtype = Tensor::zeros({3}, Device::GPU, DataType::Int32);
        const auto strided = Tensor::zeros({2, 3}, Device::GPU).transpose(0, 1);
        EXPECT_THROW(regularizer().regularize(cpu_raw, gradient, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_THROW(regularizer().regularize(raw, cpu_gradient, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_THROW(regularizer().regularize(raw, gradient, cpu_loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_THROW(regularizer().regularize(raw, wrong_shape, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_THROW(regularizer().regularize(raw, gradient, wrong_loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_THROW(regularizer().regularize(wrong_dtype, gradient, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_THROW(regularizer().regularize(strided, gradient, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_THROW(regularizer().regularize(raw, gradient, loss, scratch, static_cast<ops::Regularizer>(255), 1.f), std::invalid_argument);
        EXPECT_EQ(loss.to_vector(), std::vector<float>{-9.f});
        EXPECT_EQ(gradient.to_vector(), std::vector<float>(3, 2.f));
        EXPECT_EQ(cpu_loss.to_vector(), std::vector<float>{-9.f});
        EXPECT_EQ(cpu_gradient.to_vector(), std::vector<float>(3, 2.f));
    }

    TEST_P(HipTrainingPrimitives, RejectsOverlappingOutputsButAcceptsDisjointViewsOfOneOwner) {
        auto arena = Tensor::full({7}, 0.3f, Device::GPU);
        auto raw = arena.slice(0, 0, 3);
        auto gradient = arena.slice(0, 3, 6);
        auto loss = arena.slice(0, 6, 7);
        Tensor scratch;
        auto alias = raw;
        EXPECT_THROW(regularizer().regularize(raw, alias, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        auto overlap = arena.slice(0, 2, 5);
        EXPECT_THROW(regularizer().regularize(raw, overlap, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        auto overlap_loss = arena.slice(0, 3, 4);
        EXPECT_THROW(regularizer().regularize(raw, gradient, overlap_loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        auto input_loss = arena.slice(0, 0, 1);
        EXPECT_THROW(regularizer().regularize(raw, gradient, input_loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        EXPECT_EQ(arena.to_vector(), std::vector<float>(7, 0.3f));
        regularizer().regularize(raw, gradient, loss, scratch, ops::Regularizer::Scale, 1.f);
        EXPECT_EQ(raw.to_vector(), std::vector<float>(3, 0.3f));
        EXPECT_NEAR(loss.item<float>(), std::exp(0.3f), 1e-5f);
        for (const auto value : gradient.to_vector())
            EXPECT_NEAR(value, 0.3f + std::exp(0.3f) / 3.f, 1e-5f);
    }

    TEST_P(HipTrainingPrimitives, IntermediateAllocationRestoresTheCallersBackendScope) {
        if (!gpu_backend_compiled(GpuBackend::Vulkan))
            GTEST_SKIP() << "A second compiled backend is required for scope restoration";
        const auto raw = Tensor::zeros({3}, Device::GPU);
        auto gradient = Tensor::zeros({3}, Device::GPU);
        auto loss = Tensor::zeros({1}, Device::GPU);
        Tensor scratch;
        const GpuBackendScope other(GpuBackend::Vulkan);
        regularizer().regularize(raw, gradient, loss, scratch, ops::Regularizer::Scale, 1.f);
        EXPECT_EQ(TensorExecutionTarget::current().backend(), GpuBackend::Vulkan);
        EXPECT_EQ(gpu_backend_of(loss), GpuBackend::HIP);
        EXPECT_EQ(gpu_backend_of(gradient), GpuBackend::HIP);
        EXPECT_NEAR(loss.item<float>(), 1.f, 1e-6f);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, HipTrainingPrimitives,
                             testing::Values(lfs::test::TensorBackendTarget{"HIP", GpuBackend::HIP}),
                             lfs::test::tensor_backend_parameter_name);
#endif
} // namespace
