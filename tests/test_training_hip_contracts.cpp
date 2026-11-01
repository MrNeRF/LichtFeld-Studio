/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_upload.hpp"
#include "lfs/training/ops/registry.hpp"
#include "tensor_backend_contract.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
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

        template <typename Call>
        void expect_invalid(Call&& call, std::initializer_list<std::string_view> context) {
            try {
                call();
                FAIL() << "Expected std::invalid_argument";
            } catch (const std::invalid_argument& error) {
                const std::string_view message(error.what());
                for (const auto observed : context)
                    EXPECT_NE(message.find(observed), std::string_view::npos) << message << "; missing " << observed;
            }
        }

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
        Tensor invalid;
        const auto reject = [&](const Tensor& input, Tensor& grad, Tensor& value,
                                std::initializer_list<std::string_view> context, ops::Regularizer kind = ops::Regularizer::Scale) {
            const auto gradient_before = grad.to_vector();
            const auto loss_before = value.is_valid() ? value.to_vector() : std::vector<float>{};
            expect_invalid([&] { regularizer().regularize(input, grad, value, scratch, kind, 1.f); }, context);
            EXPECT_EQ(grad.to_vector(), gradient_before);
            if (value.is_valid())
                EXPECT_EQ(value.to_vector(), loss_before);
        };
        reject(cpu_raw, gradient, loss, {"HIP parameter", "valid=true", "device=cpu", "backend=none", "contiguous=true"});
        reject(raw, cpu_gradient, loss, {"HIP gradient", "device=cpu", "backend=none"});
        reject(raw, gradient, cpu_loss, {"HIP loss", "device=cpu", "backend=none"});
        reject(raw, wrong_shape, loss, {"gradient_shape=" + wrong_shape.shape().str(), "parameter_shape=" + raw.shape().str()});
        reject(raw, gradient, wrong_loss, {"loss_elements=2", "loss_shape=" + wrong_loss.shape().str()});
        reject(wrong_dtype, gradient, loss, {"HIP parameter", "dtype=int32", "backend=HIP", "shape=" + wrong_dtype.shape().str()});
        reject(raw, gradient, loss, {"regularizer=255", "supported=0,1"}, static_cast<ops::Regularizer>(255));
        reject(invalid, gradient, loss, {"HIP parameter", "valid=false", "dtype=invalid", "backend=none"});
        reject(raw, gradient, invalid, {"HIP loss", "valid=false", "dtype=invalid", "backend=none"});
        EXPECT_EQ(loss.to_vector(), std::vector<float>{-9.f});
        EXPECT_EQ(gradient.to_vector(), std::vector<float>(3, 2.f));
        EXPECT_EQ(cpu_loss.to_vector(), std::vector<float>{-9.f});
        EXPECT_EQ(cpu_gradient.to_vector(), std::vector<float>(3, 2.f));
    }

    TEST_P(HipTrainingPrimitives, RejectsNoncontiguousOperandsWithMatchingShapesBeforeWrites) {
        const auto strided = Tensor::full({2, 3}, 0.3f, Device::GPU).transpose(0, 1);
        const auto raw = Tensor::full(strided.shape(), 0.3f, Device::GPU);
        auto gradient = Tensor::full(strided.shape(), 2.f, Device::GPU);
        auto strided_gradient = Tensor::full({2, 3}, 2.f, Device::GPU).transpose(0, 1);
        auto loss = Tensor::full({1}, -9.f, Device::GPU);
        Tensor scratch;
        ASSERT_FALSE(strided.is_contiguous());
        ASSERT_FALSE(strided_gradient.is_contiguous());
        ASSERT_TRUE(raw.is_contiguous());
        ASSERT_TRUE(gradient.is_contiguous());
        ASSERT_EQ(strided.shape(), gradient.shape());
        ASSERT_EQ(raw.shape(), strided_gradient.shape());
        expect_invalid([&] { regularizer().regularize(strided, gradient, loss, scratch, ops::Regularizer::Scale, 1.f); },
                       {"HIP parameter", "dtype=float32", "backend=HIP", "contiguous=false", "shape=" + strided.shape().str()});
        expect_invalid([&] { regularizer().regularize(raw, strided_gradient, loss, scratch, ops::Regularizer::Scale, 1.f); },
                       {"HIP gradient", "dtype=float32", "backend=HIP", "contiguous=false", "shape=" + strided_gradient.shape().str()});
        EXPECT_EQ(loss.to_vector(), std::vector<float>{-9.f});
        EXPECT_EQ(gradient.to_vector(), std::vector<float>(6, 2.f));
        EXPECT_EQ(strided_gradient.to_vector(), std::vector<float>(6, 2.f));
        EXPECT_EQ(raw.to_vector(), std::vector<float>(6, 0.3f));
        EXPECT_EQ(strided.to_vector(), std::vector<float>(6, 0.3f));
    }

    TEST_P(HipTrainingPrimitives, RejectsOverlappingOutputsButAcceptsDisjointViewsOfOneOwner) {
        auto arena = Tensor::full({7}, 0.3f, Device::GPU);
        auto raw = arena.slice(0, 0, 3);
        auto gradient = arena.slice(0, 3, 6);
        auto loss = arena.slice(0, 6, 7);
        Tensor scratch;
        auto alias = raw;
        expect_invalid([&] { regularizer().regularize(raw, alias, loss, scratch, ops::Regularizer::Scale, 1.f); },
                       {"must not overlap", "parameter=0x", "parameter_bytes=12", "gradient=0x", "gradient_bytes=12", "loss=0x", "loss_bytes=4"});
        auto overlap = arena.slice(0, 2, 5);
        EXPECT_THROW(regularizer().regularize(raw, overlap, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        auto overlap_loss = arena.slice(0, 3, 4);
        EXPECT_THROW(regularizer().regularize(raw, gradient, overlap_loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        auto input_loss = arena.slice(0, 0, 1);
        EXPECT_THROW(regularizer().regularize(raw, gradient, input_loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
        Tensor no_gradient;
        expect_invalid([&] { regularizer().regularize(raw, no_gradient, input_loss, scratch, ops::Regularizer::Scale, 1.f); },
                       {"must not overlap", "gradient=0x0", "gradient_bytes=0", "loss_bytes=4"});
        EXPECT_FALSE(no_gradient.is_valid());
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

    TEST_P(HipTrainingPrimitives, DeferredInputsAndQueuedUpdatesRetainValuesAndOwnership) {
        constexpr size_t n = 8193;
        const auto raw = Tensor::full({n}, 0.125f, Device::GPU).add(0.25f);
        auto gradient = Tensor::full({n}, 0.25f, Device::GPU).sub(0.25f);
        auto loss = Tensor::full({1, 1}, -9.f, Device::GPU);
        Tensor scratch;
        ASSERT_TRUE(raw.is_deferred());
        ASSERT_TRUE(gradient.is_deferred());
        regularizer().regularize(raw, gradient, loss, scratch, ops::Regularizer::Scale, 0.5f);
        regularizer().regularize(raw, gradient, loss, scratch, ops::Regularizer::Opacity, 0.5f);
        const double e = std::exp(-0.375);
        const double expected = 0.5 / n * (std::exp(0.375) + e / ((1. + e) * (1. + e)));
        for (const float value : gradient.to_vector())
            EXPECT_NEAR(value, expected, expected * 2e-5);
        EXPECT_EQ(loss.shape(), (TensorShape{1, 1}));
        EXPECT_NEAR(loss.item<float>(), 0.5 / (1. + e), 1e-6);
        EXPECT_EQ(raw.to_vector(), std::vector<float>(n, 0.375f));
    }

    TEST_P(HipTrainingPrimitives, RepeatedCallsReleaseTemporaryAllocationsAfterWarmup) {
        constexpr size_t n = 4099;
        const auto raw = Tensor::full({n}, 0.125f, Device::GPU);
        auto gradient = Tensor::zeros({n}, Device::GPU);
        auto loss = Tensor::zeros({1}, Device::GPU);
        Tensor scratch;
        const auto cycle = [&] {
            for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity})
                regularizer().regularize(raw, gradient, loss, scratch, kind, 0.05f);
        };
        for (int i = 0; i < 4; ++i)
            cycle();
        const auto target = TensorExecutionTarget::default_queue(GpuBackend::HIP);
        target.wait();
        const auto warm = gpu_backend_memory_info(GpuBackend::HIP);
        ASSERT_GT(warm.allocated_bytes, 0u);
        for (int i = 0; i < 32; ++i)
            cycle();
        target.wait();
        const auto after = gpu_backend_memory_info(GpuBackend::HIP);
        RecordProperty("live_bytes_after_warmup", std::to_string(warm.allocated_bytes));
        RecordProperty("live_bytes_after_repeats", std::to_string(after.allocated_bytes));
        // Reserved cache may retain reusable blocks. Live allocations and native
        // workspace must not grow with repeated calls of the same shape.
        EXPECT_LE(after.allocated_bytes, warm.allocated_bytes);
        const double e = std::exp(-0.125);
        const double expected = 36. * 0.05 / n * (std::exp(0.125) + e / ((1. + e) * (1. + e)));
        for (const float value : gradient.to_vector())
            EXPECT_NEAR(value, expected, expected * 5e-5);
    }

    TEST_P(HipTrainingPrimitives, IndependentWorkerCallsRetainTheirPlacementAndScope) {
        constexpr size_t n = 1027;
        struct Buffers {
            Tensor raw, gradient, loss;
        };
        std::array<Buffers, 2> buffers;
        for (size_t i = 0; i < buffers.size(); ++i)
            buffers[i] = {Tensor::full({n}, i == 0 ? 0.125f : -0.25f, Device::GPU),
                          Tensor::zeros({n}, Device::GPU), Tensor::zeros({1}, Device::GPU)};
        std::array<std::future<void>, 2> workers;
        for (size_t i = 0; i < workers.size(); ++i) {
            workers[i] = std::async(std::launch::async, [&, i] {
                const auto before = TensorExecutionTarget::current();
                Tensor scratch;
                for (int repeat = 0; repeat < 8; ++repeat)
                    for (const auto kind : {ops::Regularizer::Scale, ops::Regularizer::Opacity})
                        regularizer().regularize(buffers[i].raw, buffers[i].gradient, buffers[i].loss, scratch, kind, 0.05f);
                EXPECT_EQ(TensorExecutionTarget::current(), before);
            });
        }
        for (auto& worker : workers)
            worker.get();
        for (size_t i = 0; i < buffers.size(); ++i) {
            const double x = i == 0 ? 0.125 : -0.25;
            const double e = std::exp(-std::abs(x));
            const double expected = 8. * 0.05 / n * (std::exp(x) + e / ((1. + e) * (1. + e)));
            EXPECT_EQ(gpu_backend_of(buffers[i].gradient), GpuBackend::HIP);
            EXPECT_EQ(gpu_backend_of(buffers[i].loss), GpuBackend::HIP);
            for (const float value : buffers[i].gradient.to_vector())
                EXPECT_NEAR(value, expected, expected * 5e-5);
            EXPECT_NEAR(buffers[i].loss.item<float>(), 0.05 / (1. + std::exp(-x)), 1e-6);
        }
    }

#if defined(LFS_HAS_CUDA) && LFS_HAS_CUDA
    TEST_P(HipTrainingPrimitives, ForeignCudaQueueIsIsolatedAndRestored) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP() << "A CUDA device is required for a foreign native queue";
        const auto raw = Tensor::zeros({257}, Device::GPU);
        auto gradient = Tensor::zeros(raw.shape(), Device::GPU);
        auto loss = Tensor::zeros({1}, Device::GPU);
        const auto invalid = Tensor::zeros({257}, Device::CPU);
        const auto deferred_raw = Tensor::full({8193}, 0.125f, Device::GPU).add(0.25f);
        auto deferred_gradient = Tensor::full({8193}, 0.25f, Device::GPU).sub(0.25f);
        auto deferred_loss = Tensor::zeros({1}, Device::GPU);
        ASSERT_TRUE(deferred_raw.is_deferred());
        ASSERT_TRUE(deferred_gradient.is_deferred());
        Tensor scratch;
        TensorWorkQueue foreign(GpuBackend::CUDA);
        const TensorExecutionTarget target(foreign);
        ASSERT_FALSE(target.is_default_queue());
        {
            const TensorExecutionTarget::Scope other(target);
            ASSERT_NO_THROW(regularizer().regularize(raw, gradient, loss, scratch, ops::Regularizer::Scale, 1.f));
            EXPECT_EQ(TensorExecutionTarget::current(), target);
            ASSERT_NO_THROW(regularizer().regularize(deferred_raw, deferred_gradient, deferred_loss, scratch, ops::Regularizer::Scale, 1.f));
            EXPECT_EQ(TensorExecutionTarget::current(), target);
            EXPECT_THROW(regularizer().regularize(invalid, gradient, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
            EXPECT_EQ(TensorExecutionTarget::current(), target);
            auto alias = raw;
            EXPECT_THROW(regularizer().regularize(raw, alias, loss, scratch, ops::Regularizer::Scale, 1.f), std::invalid_argument);
            EXPECT_EQ(TensorExecutionTarget::current(), target);
            const auto cuda_control = Tensor::ones({1}, Device::GPU);
            EXPECT_EQ(gpu_backend_of(cuda_control), GpuBackend::CUDA);
            EXPECT_FLOAT_EQ(cuda_control.item<float>(), 1.f);
        }
        foreign.wait();
        EXPECT_NEAR(loss.item<float>(), 1.f, 1e-6f);
        for (const auto value : gradient.to_vector())
            EXPECT_NEAR(value, 1.f / 257.f, 1e-7f);
        EXPECT_NEAR(deferred_loss.item<float>(), std::exp(0.375), 1e-5);
        for (const auto value : deferred_gradient.to_vector())
            EXPECT_NEAR(value, std::exp(0.375) / 8193., 1e-7);
    }
#endif

    INSTANTIATE_TEST_SUITE_P(Backends, HipTrainingPrimitives,
                             testing::Values(lfs::test::TensorBackendTarget{"HIP", GpuBackend::HIP}),
                             lfs::test::tensor_backend_parameter_name);
#endif
} // namespace
