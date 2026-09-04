/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/pointwise_lowering.hpp"
#include "core/tensor/internal/lazy_executor.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

    using namespace lfs::core;
    using namespace lfs::core::internal;

    bool has_cuda_device() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    std::string exception_message(const auto& operation) {
        try {
            operation();
        } catch (const std::exception& error) {
            return error.what();
        }
        return {};
    }

    void expect_values(const Tensor& tensor, const std::vector<float>& expected,
                       const float tolerance = 1e-5f) {
        const std::vector<float> actual = tensor.to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
        }
    }

    void expect_cuda_backend(const Tensor& tensor) {
        EXPECT_EQ(gpu_backend_of(tensor), GpuBackend::CUDA);
    }

    class LazyOverrideGuard {
    public:
        LazyOverrideGuard() {
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(0);
        }

        ~LazyOverrideGuard() {
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        }
    };

    TEST(TensorBackendFacade, PointwiseTableNamesAndLowersEveryFunctor) {
        // Catches table rows that omit the diagnostic name or lowering specialization.
#define LFS_POINTWISE_OP(Id, FunctorType, Name)                       \
    do {                                                              \
        constexpr auto lowered = pointwise_op_of<FunctorType>::value; \
        EXPECT_EQ(lowered, PointwiseOp::Id);                          \
        EXPECT_STREQ(pointwise_op_name(PointwiseOp::Id), Name);       \
        EXPECT_NE(*pointwise_op_name(PointwiseOp::Id), '\0');         \
    } while (false);
#include "core/tensor/backend/pointwise_ops.def"
#undef LFS_POINTWISE_OP
    }

    TEST(TensorBackendFacade, RegistryRejectsUnavailableAndCpuStorage) {
        // Catches Vulkan silently falling through to the CUDA singleton.
        const std::string vulkan_error = exception_message([] {
            (void)internal::backend_ops(GpuBackend::Vulkan);
        });
        EXPECT_NE(vulkan_error.find("Vulkan"), std::string::npos);
        EXPECT_NE(vulkan_error.find("unavailable"), std::string::npos);

        // Catches CPU tensors reaching a GPU facade through a default-backend fallback.
        const Tensor cpu = Tensor::ones({2}, Device::CPU);
        const std::string cpu_error = exception_message([&] {
            (void)internal::backend_ops_for(cpu);
        });
        EXPECT_NE(cpu_error.find("GPU storage"), std::string::npos);
    }

    TEST(TensorBackendFacade, PointwiseEntriesPreserveCudaResultsAndBackend) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        const Tensor input = Tensor::from_vector(
            std::vector<float>{-1.0f, 0.0f, 1.0f, 2.0f}, {2, 2}, Device::CUDA);

        // Catches unary dispatch selecting the wrong op or dtype specialization.
        const Tensor exponential = input.exp();
        expect_values(exponential,
                      {std::exp(-1.0f), 1.0f, std::exp(1.0f), std::exp(2.0f)});
        expect_cuda_backend(exponential);

        // Catches same-shape binary dispatch swapping or offsetting operands.
        const Tensor added = input.add(Tensor::full({2, 2}, 3.0f, Device::CUDA));
        expect_values(added, {2.0f, 3.0f, 4.0f, 5.0f});
        expect_cuda_backend(added);

        // Catches the dedicated scalar entry bypassing the CUDA scalar launcher.
        Tensor scaled = input.clone();
        scaled.mul_(2.0f);
        expect_values(scaled, {-2.0f, 0.0f, 2.0f, 4.0f});
        expect_cuda_backend(scaled);

        // Catches broadcast descriptors losing rank, dimensions, or strides.
        const Tensor lhs = Tensor::from_vector(
            std::vector<float>{1.0f, 2.0f}, {2, 1}, Device::CUDA);
        const Tensor rhs = Tensor::from_vector(
            std::vector<float>{10.0f, 20.0f, 30.0f}, {1, 3}, Device::CUDA);
        const Tensor broadcast = lhs.add(rhs);
        expect_values(broadcast, {11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f});
        expect_cuda_backend(broadcast);

        // Catches the contiguous conversion entry using source instead of destination dtype.
        const Tensor converted = Tensor::from_vector(
                                     std::vector<float>{-2.9f, 0.0f, 3.8f, 7.0f},
                                     {4}, Device::CUDA)
                                     .to(DataType::Int32);
        EXPECT_EQ(converted.to_vector_int(), (std::vector<int>{-2, 0, 3, 7}));
        expect_cuda_backend(converted);

        // Catches fill_strided ignoring a view's storage offset or physical strides.
        Tensor base = Tensor::zeros({3, 4}, Device::CUDA);
        Tensor view = base.slice(1, 1, 3);
        view.fill_(9.0f);
        expect_values(base, {0.0f, 9.0f, 9.0f, 0.0f,
                             0.0f, 9.0f, 9.0f, 0.0f,
                             0.0f, 9.0f, 9.0f, 0.0f});
        expect_cuda_backend(view);

        // Catches nonzero full() leaving the load-fill launcher outside the facade.
        const Tensor filled = Tensor::full({2, 3}, 2.25f, Device::CUDA);
        expect_values(filled, std::vector<float>(6, 2.25f));
        expect_cuda_backend(filled);
    }

    TEST(TensorBackendFacade, ClampEntriesPreserveFloatAndIntPolicies) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        // Catches out-of-place clamp failing to preserve IEEE NaN behavior.
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const Tensor source = Tensor::from_vector(
            std::vector<float>{-3.0f, -0.5f, 2.0f, nan}, {4}, Device::CUDA);
        const Tensor clamped = source.clamp(-1.0f, 1.0f);
        const auto clamped_values = clamped.to_vector();
        ASSERT_EQ(clamped_values.size(), 4u);
        EXPECT_FLOAT_EQ(clamped_values[0], -1.0f);
        EXPECT_FLOAT_EQ(clamped_values[1], -0.5f);
        EXPECT_FLOAT_EQ(clamped_values[2], 1.0f);
        EXPECT_TRUE(std::isnan(clamped_values[3]));
        expect_cuda_backend(clamped);

        // Catches the in-place Float32 clamp entry writing through the wrong descriptor.
        Tensor inplace = source.clone();
        inplace.clamp_(-2.0f, 0.5f);
        const auto inplace_values = inplace.to_vector();
        ASSERT_EQ(inplace_values.size(), 4u);
        EXPECT_FLOAT_EQ(inplace_values[0], -2.0f);
        EXPECT_FLOAT_EQ(inplace_values[1], -0.5f);
        EXPECT_FLOAT_EQ(inplace_values[2], 0.5f);
        EXPECT_TRUE(std::isnan(inplace_values[3]));

        // Catches Int32 clamp scalar tags being interpreted as floating-point values.
        Tensor integers = Tensor::from_vector(
            std::vector<int>{-5, -1, 3, 9}, {4}, Device::CUDA);
        integers.clamp_(-2.0f, 4.0f);
        EXPECT_EQ(integers.to_vector_int(), (std::vector<int>{-2, -1, 3, 4}));
        expect_cuda_backend(integers);
    }

    TEST(TensorBackendFacade, LazyFusedChainUsesUnifiedPointwiseIds) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        LazyOverrideGuard guard;
        const Tensor input = Tensor::full({4096}, 2.0f, Device::CUDA);

        // Catches add/mul/sub recipe ids diverging from the CUDA fused-chain ABI.
        const Tensor result = input.add(3.0f).mul(4.0f).sub(1.0f);
        expect_values(result, std::vector<float>(4096, 19.0f));
        expect_cuda_backend(result);
        EXPECT_GT(internal::lazy_executor_diagnostics_snapshot_for_testing().fused_launches, 0u);
    }

} // namespace
