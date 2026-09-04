/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/pointwise_lowering.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/tensor_ops.hpp"
#include "core/tensor_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
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

    TEST(TensorBackendFacade, RegistryReturnsVulkanAndRejectsCpuStorage) {
        // Catches the Vulkan registry entry silently returning the CUDA singleton.
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor vulkan = Tensor::empty({2}, Device::CUDA);
        EXPECT_EQ(internal::backend_ops(GpuBackend::Vulkan)
                      .classify_pointer(vulkan.data_ptr()),
                  PointerClass::Device);

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

    TEST(TensorBackendFacadeB, ReductionsPreserveCudaResultsBackendAndSyncBoundaries) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        const Tensor values = Tensor::from_vector(
            std::vector<float>{1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -6.0f},
            {2, 3}, Device::CUDA);

        // Catches the four synchronous direct scalar adapters selecting the wrong reduction.
        EXPECT_FLOAT_EQ(values.sum_scalar(), 0.0f);
        EXPECT_FLOAT_EQ(values.mean_scalar(), 0.0f);
        EXPECT_FLOAT_EQ(values.max_scalar(), 4.0f);
        EXPECT_FLOAT_EQ(values.min_scalar(), -6.0f);

        // Catches ReduceProgram axis/count/result-dtype loss in the general reduction entry.
        const Tensor row_sum = values.sum({1});
        expect_values(row_sum, {2.0f, -2.0f});
        expect_cuda_backend(row_sum);

        // Catches the specialized column adapter swapping rows and columns.
        const Tensor column_sum = values.sum({0});
        expect_values(column_sum, {5.0f, -2.0f, -3.0f});
        expect_cuda_backend(column_sum);

        // Catches the strided reduction adapter dropping outer/reduce/inner sizes.
        tensor_ops::set_reduce_path_override_for_testing(
            tensor_ops::ReducePathForTesting::StridedFast);
        const Tensor wide = Tensor::ones({2, 3, 256}, Device::CUDA);
        const Tensor strided_sum = wide.sum({1});
        tensor_ops::set_reduce_path_override_for_testing(
            tensor_ops::ReducePathForTesting::None);
        expect_values(strided_sum, std::vector<float>(512, 3.0f));
        EXPECT_EQ(tensor_ops::reduce_last_path_for_testing(),
                  tensor_ops::ReducePathForTesting::StridedFast);
        expect_cuda_backend(strided_sum);

        // Catches Float32 and Bool count adapters returning before their downloads complete.
        EXPECT_EQ(values.count_nonzero(), 5u);
        EXPECT_EQ(values.gt(0.0f).count_nonzero(), 3u);

        // Catches the synchronous NaN and Inf flag adapters aliasing or losing their inputs.
        const Tensor special = Tensor::from_vector(
            std::vector<float>{1.0f, std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()},
            {3}, Device::CUDA);
        EXPECT_TRUE(special.has_nan());
        EXPECT_TRUE(special.has_inf());
        EXPECT_FALSE(values.has_nan());
        EXPECT_FALSE(values.has_inf());
    }

    TEST(TensorBackendFacadeB, FusedTransformReductionsPreserveChainsAndSegments) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        LazyOverrideGuard guard;

        // Catches the full fused transform-reduce adapter dropping the pointwise chain.
        const Tensor full_source = Tensor::ones({4096}, Device::CUDA);
        const Tensor full_result = full_source.add(2.0f).mul(3.0f).sum();
        EXPECT_NEAR(full_result.item(), 4096.0f * 9.0f, 1e-2f);
        expect_cuda_backend(full_result);

        // Catches the segmented adapter confusing segment count with segment size.
        const Tensor segmented_source = Tensor::ones({2, 2048}, Device::CUDA);
        const Tensor segmented_result = segmented_source.add(1.0f).sum({1});
        expect_values(segmented_result, {4096.0f, 4096.0f}, 1e-2f);
        expect_cuda_backend(segmented_result);
        EXPECT_GE(internal::lazy_executor_diagnostics_snapshot_for_testing().fused_reduce_launches,
                  2u);
    }

    TEST(TensorBackendFacadeB, ScanAndSortPreserveLayoutsIndicesAndBackend) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        // Catches cumsum losing rank, dimension, or dtype from its layout descriptor.
        const Tensor matrix = Tensor::from_vector(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {2, 3}, Device::CUDA);
        const Tensor cumulative = matrix.cumsum(1);
        expect_values(cumulative, {1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f});
        expect_cuda_backend(cumulative);

        // Catches the 1D sort adapter reversing order or writing indices through values.
        const Tensor vector = Tensor::from_vector(
            std::vector<float>{3.0f, 1.0f, 2.0f}, {3}, Device::CUDA);
        auto [sorted_1d, indices_1d] = vector.sort(0, false);
        expect_values(sorted_1d, {1.0f, 2.0f, 3.0f});
        EXPECT_EQ(indices_1d.to_vector_int64(), (std::vector<int64_t>{1, 2, 0}));
        expect_cuda_backend(sorted_1d);
        expect_cuda_backend(indices_1d);

        // Catches the 2D sort adapter dropping outer/inner slice metadata.
        auto [sorted_2d, indices_2d] = matrix.sort(1, true);
        expect_values(sorted_2d, {3.0f, 2.0f, 1.0f, 6.0f, 5.0f, 4.0f});
        EXPECT_EQ(indices_2d.to_vector_int64(),
                  (std::vector<int64_t>{2, 1, 0, 2, 1, 0}));
        expect_cuda_backend(sorted_2d);
        expect_cuda_backend(indices_2d);
    }

    TEST(TensorBackendFacadeB, MatrixEntriesPreserveDimensionsEpiloguesAndBackend) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        const Tensor lhs = Tensor::from_vector(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
            {2, 3}, Device::CUDA);
        const Tensor rhs = Tensor::from_vector(
            std::vector<float>{1.0f, 2.0f, 0.0f, 1.0f, 1.0f, 0.0f},
            {3, 2}, Device::CUDA);

        // Catches sgemm swapping m/n/k or operand descriptors.
        const Tensor mm = lhs.mm(rhs);
        expect_values(mm, {4.0f, 4.0f, 10.0f, 13.0f});
        expect_cuda_backend(mm);

        // Catches sgemm_tn and bias_add using incompatible matrix orientation/channel data.
        const Tensor weight = Tensor::from_vector(
            std::vector<float>{1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f},
            {2, 3}, Device::CUDA);
        const Tensor bias = Tensor::from_vector(
            std::vector<float>{1.0f, -1.0f}, {2}, Device::CUDA);
        const Tensor linear = lhs.linear(weight, bias);
        expect_values(linear, {5.0f, 4.0f, 11.0f, 10.0f});
        expect_cuda_backend(linear);

        // Catches batched sgemm omitting the batch stride.
        const Tensor batch_lhs = Tensor::ones({2, 2, 3}, Device::CUDA);
        const Tensor batch_rhs = Tensor::ones({2, 3, 2}, Device::CUDA);
        const Tensor bmm = batch_lhs.bmm(batch_rhs);
        expect_values(bmm, std::vector<float>(8, 3.0f));
        expect_cuda_backend(bmm);

        // Catches dot's output descriptor or reduction length being lost.
        const Tensor dot = Tensor::from_vector(
                               std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CUDA)
                               .dot(Tensor::from_vector(
                                   std::vector<float>{4.0f, 5.0f, 6.0f}, {3}, Device::CUDA));
        EXPECT_FLOAT_EQ(dot.item(), 32.0f);
        expect_cuda_backend(dot);

        // Catches diag and eye swapping square/rectangular dimensions.
        const Tensor diagonal = Tensor::diag(Tensor::from_vector(
            std::vector<float>{2.0f, 3.0f}, {2}, Device::CUDA));
        expect_values(diagonal, {2.0f, 0.0f, 0.0f, 3.0f});
        expect_cuda_backend(diagonal);
        const Tensor identity = Tensor::eye(2, 3, Device::CUDA);
        expect_values(identity, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
        expect_cuda_backend(identity);

        // Catches cdist dropping p or the two independent row counts.
        const Tensor distances = Tensor::from_vector(
                                     std::vector<float>{0.0f, 0.0f, 3.0f, 4.0f},
                                     {2, 2}, Device::CUDA)
                                     .cdist(Tensor::from_vector(
                                                std::vector<float>{0.0f, 4.0f}, {1, 2}, Device::CUDA),
                                            2.0f);
        expect_values(distances, {4.0f, 3.0f});
        expect_cuda_backend(distances);

        // Catches the fused sgemm bias-ReLU epilogue bypassing its facade entry.
        const Tensor image = Tensor::ones({1, 1, 1, 500000}, Device::CUDA);
        const Tensor conv_weight = Tensor::full({1, 1}, 2.0f, Device::CUDA);
        const Tensor conv_bias = Tensor::full({1}, 1.0f, Device::CUDA);
        Tensor conv_output = Tensor::zeros({1, 1, 1, 500000}, Device::CUDA);
        image.conv1x1_bias_relu_out(conv_weight, conv_bias, conv_output);
        EXPECT_FLOAT_EQ(conv_output.to_vector().front(), 3.0f);
        EXPECT_FLOAT_EQ(conv_output.to_vector().back(), 3.0f);
        expect_cuda_backend(conv_output);
    }

    TEST(TensorBackendFacadeB, NnEntriesPreservePoolingBiasReluAndBackend) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        const Tensor image = Tensor::from_vector(
            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
            {1, 1, 2, 2}, Device::CUDA);

        // Catches max_pool2d losing its kernel/stride/padding descriptor fields.
        const Tensor maximum = image.max_pool2d(2, 2);
        expect_values(maximum, {4.0f});
        expect_cuda_backend(maximum);

        // Catches adaptive_avg_pool2d confusing input and output extents.
        const Tensor average = image.adaptive_avg_pool2d(1, 1);
        expect_values(average, {2.5f});
        expect_cuda_backend(average);

        const Tensor weight = Tensor::from_vector(
            std::vector<float>{1.0f, -1.0f, -1.0f, 1.0f},
            {2, 2}, Device::CUDA);
        const Tensor bias = Tensor::from_vector(
            std::vector<float>{1.0f, -1.0f}, {2}, Device::CUDA);
        const Tensor input = Tensor::from_vector(
            std::vector<float>{2.0f, 3.0f}, {1, 2}, Device::CUDA);

        // Catches linear's bias_add adapter indexing the wrong channel.
        const Tensor biased = input.linear(weight, bias);
        expect_values(biased, {0.0f, 0.0f});
        expect_cuda_backend(biased);

        // Catches the separate bias_relu adapter failing in the out API.
        Tensor fused_output = Tensor::zeros({1, 2}, Device::CUDA);
        input.linear_bias_relu_out(weight, bias, fused_output);
        expect_values(fused_output, {0.0f, 0.0f});
        expect_cuda_backend(fused_output);

        // Catches relu_out writing through its input descriptor.
        const Tensor relu_input = Tensor::from_vector(
            std::vector<float>{-2.0f, 3.0f}, {2}, Device::CUDA);
        Tensor relu_output = Tensor::zeros({2}, Device::CUDA);
        relu_input.relu_out(relu_output);
        expect_values(relu_output, {0.0f, 3.0f});
        expect_cuda_backend(relu_output);
    }

    TEST(TensorBackendFacadeB, RandomEntriesPreserveContractsAndBackend) {
        if (!has_cuda_device()) {
            GTEST_SKIP() << "CUDA device required";
        }

        // Catches uniform dropping its bounds or output descriptor.
        Tensor uniform = Tensor::zeros({2048}, Device::CUDA);
        uniform.uniform_(2.0f, 4.0f);
        const auto uniform_values = uniform.to_vector();
        EXPECT_TRUE(std::all_of(uniform_values.begin(), uniform_values.end(),
                                [](const float value) { return value >= 2.0f && value <= 4.0f; }));
        expect_cuda_backend(uniform);

        // Catches bernoulli losing p or selecting the wrong random entry.
        const Tensor bernoulli = Tensor::bernoulli({128}, 1.0f, Device::CUDA);
        expect_values(bernoulli, std::vector<float>(128, 1.0f));
        expect_cuda_backend(bernoulli);

        // Catches randint losing integer bounds or dtype metadata.
        const Tensor integers = Tensor::randint(
            {512}, -3, 5, Device::CUDA, DataType::Int32);
        const auto integer_values = integers.to_vector_int();
        EXPECT_TRUE(std::all_of(integer_values.begin(), integer_values.end(),
                                [](const int value) { return value >= -3 && value < 5; }));
        expect_cuda_backend(integers);

        // Catches multinomial swapping weight count and requested sample count.
        const Tensor samples = Tensor::multinomial(
            Tensor::from_vector(std::vector<float>{0.0f, 0.0f, 1.0f},
                                {3}, Device::CUDA),
            64, true);
        EXPECT_EQ(samples.to_vector_int64(), std::vector<int64_t>(64, 2));
        expect_cuda_backend(samples);

        // Catches normal's odd-count scratch path overflowing or omitting its synchronous copy.
        Tensor normal = Tensor::zeros({257}, Device::CUDA);
        normal.normal_(3.0f, 0.25f);
        const auto normal_values = normal.to_vector();
        const float mean = std::accumulate(normal_values.begin(), normal_values.end(), 0.0f) /
                           static_cast<float>(normal_values.size());
        EXPECT_NEAR(mean, 3.0f, 0.08f);
        expect_cuda_backend(normal);
    }

} // namespace
