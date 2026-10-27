/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/gpu_device_info.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/gpu_elapsed.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor_completion.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_protocol_export.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include "core/tensor_vulkan_interop.hpp"
#include "tensor_backend_contract.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest-spi.h>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    class TensorOperationContracts : public lfs::test::TensorBackendContract {
    protected:
        Tensor values(const std::vector<float>& data, const TensorShape& shape) const {
            return Tensor::from_vector(data, shape, device());
        }

        void expect_values(const Tensor& result, const TensorShape& shape, const std::vector<float>& expected, const float tolerance = 0.f) const {
            expect_placement(result);
            ASSERT_EQ(result.shape(), shape);
            ASSERT_EQ(result.dtype(), DataType::Float32);
            const auto actual = result.to_vector();
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i) {
                ASSERT_TRUE(std::isfinite(actual[i])) << i;
                EXPECT_NEAR(actual[i], expected[i], tolerance) << i;
            }
        }
    };

    static_assert(static_cast<unsigned>(GpuBackend::CUDA) == 0);
    static_assert(static_cast<unsigned>(GpuBackend::Vulkan) == 1);
    static_assert(static_cast<unsigned>(GpuBackend::Metal) == 2);
    static_assert(static_cast<unsigned>(GpuBackend::HIP) == 3);
    static_assert(kGpuBackendCount == 4);
    static_assert(kGpuBackends[3] == GpuBackend::HIP);

    TEST_P(TensorOperationContracts, BackendIdentityIsStableWithoutAdvertisingUncompiledDevices) {
        EXPECT_STREQ(gpu_backend_name(GpuBackend::HIP), "HIP");
        for (const auto backend : kGpuBackends) {
            const bool advertised = std::find(kCompiledGpuBackends.begin(), kCompiledGpuBackends.end(), backend) != kCompiledGpuBackends.end();
            EXPECT_EQ(gpu_backend_compiled(backend), advertised);
            if (!advertised) {
                EXPECT_FALSE(gpu_backend_available(backend));
                EXPECT_FALSE(gpu_backend_live(backend));
                EXPECT_EQ(gpu_device_count(backend), 0);
                EXPECT_FALSE(gpu_backend_device_info(backend).has_value());
                EXPECT_FALSE(gpu_backend_device_info(backend, 0).has_value());
            }
        }
        if constexpr (!gpu_backend_compiled(GpuBackend::HIP)) {
            EXPECT_THROW((void)selected_hip_device(), std::runtime_error);
            EXPECT_FALSE(gpu_backend_live(GpuBackend::HIP));
        }
    }

    TEST_P(TensorOperationContracts, ElapsedTimerUsesTheSameOptionalContractOnEveryBackend) {
        if (!GetParam().backend) {
            for (const auto backend : kGpuBackends) {
                GpuElapsed empty(backend, 0);
                EXPECT_FALSE(empty.ready());
                EXPECT_FALSE(empty.mark(0, nullptr));
                EXPECT_FALSE(empty.wait_event(0));
                EXPECT_FALSE(empty.wait_queue(nullptr));
                EXPECT_FALSE(empty.milliseconds(0, 1).has_value());
            }
            return;
        }
        const auto backend = *GetParam().backend;
        const auto queue = TensorExecutionTarget::default_queue(backend);
        GpuElapsed timer(backend, 2);
        if (!timer.ready()) {
            EXPECT_FALSE(timer.mark(0, queue));
            EXPECT_FALSE(timer.wait_event(0));
            EXPECT_FALSE(timer.milliseconds(0, 1).has_value());
            return; // Explicitly unavailable optional instrumentation.
        }
        EXPECT_FALSE(timer.mark(2, queue));
        ASSERT_TRUE(timer.mark(0, queue));
        const auto result = values({1, 2, 3}, {3}).mul(2.f);
        ASSERT_TRUE(timer.mark(1, queue));
        ASSERT_TRUE(timer.wait_event(1));
        ASSERT_TRUE(timer.wait_queue(queue));
        const auto elapsed = timer.milliseconds(0, 1);
        ASSERT_TRUE(elapsed.has_value());
        EXPECT_GE(*elapsed, 0.f);
        EXPECT_TRUE(std::isfinite(*elapsed));
        expect_values(result, {3}, {2, 4, 6});
        const auto other = backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA;
        EXPECT_THROW((void)timer.mark(0, TensorExecutionTarget::default_queue(other)), std::invalid_argument);
    }

    TEST_P(TensorOperationContracts, Int32BooleanConversionKeepsNonzeroSemantics) {
        const auto input = Tensor::from_vector(std::vector<int>{0, 1, -1, 256, -256, INT32_MAX, INT32_MIN}, {7}, device());
        const auto converted = input.to(DataType::Bool).to(DataType::Int32);
        expect_placement(converted);
        EXPECT_EQ(converted.dtype(), DataType::Int32);
        EXPECT_EQ(converted.to_vector_int(), (std::vector<int>{0, 1, 1, 1, 1, 1, 1}));
    }

    TEST_P(TensorOperationContracts, OddNormalDrawsKeepTheirShapeAndFiniteValues) {
        for (const size_t count : {size_t{1}, size_t{3}, size_t{257}}) {
            auto result = values(std::vector<float>(count, 0.f), {count});
            result.normal_(1.f, 0.25f);
            expect_placement(result);
            EXPECT_EQ(result.shape(), TensorShape({count}));
            for (const auto value : result.to_vector())
                EXPECT_TRUE(std::isfinite(value));
        }
    }

    TEST_P(TensorOperationContracts, RejectsMalformedShapeDtypeAndTruncatedResults) {
        const auto expect_rejected = [&](const Tensor& result, const TensorShape& shape,
                                         const std::vector<float>& expected, const char* diagnostic) {
            testing::TestPartResultArray failures;
            {
                testing::ScopedFakeTestPartResultReporter reporter(
                    testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &failures);
                expect_values(result, shape, expected);
            }
            ASSERT_EQ(failures.size(), 1);
            EXPECT_TRUE(failures.GetTestPartResult(0).fatally_failed());
            EXPECT_NE(std::string(failures.GetTestPartResult(0).message()).find(diagnostic), std::string::npos);
        };
        // These have valid values and placement but violate the operation contract.
        ASSERT_NO_FATAL_FAILURE(expect_rejected(values({1, 2, 3, 4}, {2, 2}), {4}, {1, 2, 3, 4}, "result.shape()"));
        ASSERT_NO_FATAL_FAILURE(expect_rejected(Tensor::from_vector(std::vector<int>{1, 2}, {2}, device()),
                                                {2}, {1, 2}, "result.dtype()"));
        ASSERT_NO_FATAL_FAILURE(expect_rejected(values({-1, -1, 0}, {3}), {6}, {-1, -1, 0, 2, 3, 3}, "result.shape()"));
    }

    TEST_P(TensorOperationContracts, PointwiseBroadcastUsesAnalyticValuesAtOddSizes) {
        constexpr size_t rows = 33, columns = 127;
        std::vector<float> a(rows * columns), b(columns), expected(a.size());
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<float>(int(i % 31) - 15) / 8.f;
        for (size_t i = 0; i < b.size(); ++i)
            b[i] = static_cast<float>(i % 7) / 4.f;
        for (size_t i = 0; i < a.size(); ++i)
            expected[i] = a[i] * b[i % columns] - 0.25f;
        // Dyadic inputs and exact products avoid a backend-specific math oracle.
        ASSERT_NO_FATAL_FAILURE(expect_values(values(a, {rows, columns}).mul(values(b, {1, columns})).add(0.25f).sub(0.5f), {rows, columns}, expected));
    }

    TEST_P(TensorOperationContracts, MovementPreservesLogicalStridedValues) {
        const auto source = values({1, 2, 3, 4, 5, 6}, {2, 3});
        const auto transposed = source.transpose(0, 1);
        const auto result = Tensor::cat({transposed.contiguous(), transposed.contiguous()}, 1);
        ASSERT_NO_FATAL_FAILURE(expect_values(result, {3, 4}, {1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6}));
        auto detached = transposed.clone();
        detached.fill_(9.f);
        ASSERT_NO_FATAL_FAILURE(expect_values(source, {2, 3}, {1, 2, 3, 4, 5, 6}));
        ASSERT_NO_FATAL_FAILURE(expect_values(detached, {3, 2}, std::vector<float>(6, 9.f)));
    }

    TEST_P(TensorOperationContracts, ReductionsAndScansMatchAnalyticIntegers) {
        const auto input = values({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
        ASSERT_NO_FATAL_FAILURE(expect_values(input.sum(0), {4}, {15, 18, 21, 24}));
        ASSERT_NO_FATAL_FAILURE(expect_values(input.sum(1), {3}, {10, 26, 42}));
        ASSERT_NO_FATAL_FAILURE(expect_values(input.cumsum(1), {3, 4}, {1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42}));
        EXPECT_FLOAT_EQ(input.sum_scalar(), 78.f);
    }

    TEST_P(TensorOperationContracts, IndexSelectReadsStridedIndices) {
        const auto input = values({1, 2, 3, 4, 5, 6}, {3, 2});
        const auto indices = Tensor::from_vector(std::vector<int>{2, 99, 0, 99, 2, 99}, {3, 2}, device())
                                 .slice(1, 0, 1)
                                 .squeeze();
        ASSERT_NO_FATAL_FAILURE(expect_values(input.index_select(0, indices), {3, 2}, {5, 6, 1, 2, 5, 6}));
    }

    TEST_P(TensorOperationContracts, NonzeroUsesSelectedCountRatherThanCapacity) {
        const auto check_indices = [&](const Tensor& input, const std::vector<int64_t>& expected) {
            const auto indices = input.nonzero();
            expect_placement(indices);
            ASSERT_EQ(indices.shape(), TensorShape({expected.size(), 1}));
            ASSERT_EQ(indices.dtype(), DataType::Int64);
            EXPECT_EQ(indices.to_vector_int64(), expected);
        };
        const auto input = values({0, -2, 0, 3, 0, 0, 4}, {7});
        ASSERT_NO_FATAL_FAILURE(check_indices(input, {1, 3, 6}));
        ASSERT_NO_FATAL_FAILURE(check_indices(input > 0.f, {3, 6}));
        ASSERT_NO_FATAL_FAILURE(check_indices(values({0, 0, 0}, {3}), {}));
        ASSERT_NO_FATAL_FAILURE(check_indices(values({1, -1, 2}, {3}), {0, 1, 2}));
    }

    TEST_P(TensorOperationContracts, MaskedSelectionReadsStridedValuesAndExactCardinality) {
        const auto input = values({1, 2, 3, 4, 5, 6}, {2, 3}).transpose(0, 1);
        const auto mask = input > 3.f;
        ASSERT_NO_FATAL_FAILURE(expect_values(input.masked_select(mask), {3}, {4, 5, 6}));
        ASSERT_NO_FATAL_FAILURE(expect_values(input.masked_select(input > 10.f), {0}, {}));
        auto filled = input.clone();
        filled.masked_fill_(mask, -2.f);
        ASSERT_NO_FATAL_FAILURE(expect_values(filled, {3, 2}, {1, -2, 2, -2, 3, -2}));
    }

    TEST_P(TensorOperationContracts, IndexedWritesPreserveOtherRowsAndAccumulateDuplicates) {
        const auto indices = Tensor::from_vector(std::vector<int>{2, 0}, {2}, device());
        auto output = values({1, 2, 3, 4, 5, 6}, {3, 2});
        output.index_copy_(0, indices, values({7, 8, 9, 10}, {2, 2}));
        ASSERT_NO_FATAL_FAILURE(expect_values(output, {3, 2}, {9, 10, 3, 4, 7, 8}));
        output.index_fill_(0, indices, -1.f);
        ASSERT_NO_FATAL_FAILURE(expect_values(output, {3, 2}, {-1, -1, 3, 4, -1, -1}));
        const auto duplicates = Tensor::from_vector(std::vector<int>{1, 1}, {2}, device());
        output.index_add_(0, duplicates, values({1, 2, 3, 4}, {2, 2}));
        ASSERT_NO_FATAL_FAILURE(expect_values(output, {3, 2}, {-1, -1, 7, 10, -1, -1}));
    }

    TEST_P(TensorOperationContracts, SortReturnsAValidPermutationForDuplicateKeys) {
        const std::vector<float> data{3, -1, 3, 0, -1, 2};
        const auto [sorted, indices] = values(data, {data.size()}).sort();
        ASSERT_NO_FATAL_FAILURE(expect_values(sorted, {data.size()}, {-1, -1, 0, 2, 3, 3}));
        ASSERT_EQ(indices.shape(), TensorShape({data.size()}));
        ASSERT_EQ(indices.dtype(), DataType::Int64);
        expect_placement(indices);
        const auto permutation = indices.to_vector_int64();
        ASSERT_EQ(permutation.size(), data.size());
        std::vector<bool> seen(data.size(), false);
        const auto ordered = sorted.to_vector();
        ASSERT_EQ(ordered.size(), data.size());
        for (size_t i = 0; i < permutation.size(); ++i) {
            ASSERT_GE(permutation[i], 0);
            ASSERT_LT(static_cast<size_t>(permutation[i]), data.size());
            EXPECT_FALSE(seen[permutation[i]]);
            seen[permutation[i]] = true;
            EXPECT_EQ(ordered[i], data[permutation[i]]);
        }
    }

    TEST_P(TensorOperationContracts, MatrixMultiplyMatchesAnalyticValuesForStridedInputs) {
        const auto a = values({1, 2, 3, 4, 5, 6}, {3, 2}).transpose(0, 1);
        const auto b = values({7, 8, 9, 10, 11, 12}, {2, 3}).transpose(0, 1);
        // [1 3 5; 2 4 6] times [7 10; 8 11; 9 12].
        ASSERT_NO_FATAL_FAILURE(expect_values(a.matmul(b), {2, 2}, {76, 103, 100, 136}));
    }

    TEST_P(TensorOperationContracts, SoftmaxMatchesAnIndependentStableReference) {
        const std::vector<float> input{1000, 1001, 999, -1000, -999, -1001};
        std::vector<float> expected(input.size());
        for (size_t row = 0; row < 2; ++row) {
            const double maximum = *std::max_element(input.begin() + row * 3, input.begin() + (row + 1) * 3);
            double denominator = 0.;
            for (size_t j = 0; j < 3; ++j)
                denominator += std::exp(double(input[row * 3 + j]) - maximum);
            for (size_t j = 0; j < 3; ++j)
                expected[row * 3 + j] = static_cast<float>(std::exp(double(input[row * 3 + j]) - maximum) / denominator);
        }
        // FP32 reductions and exp may round differently; no tolerance depends on the backend.
        ASSERT_NO_FATAL_FAILURE(expect_values(nn::softmax(values(input, {2, 3})), {2, 3}, expected, 2e-6f));
    }

    TEST_P(TensorOperationContracts, DegenerateSamplingHasBackendIndependentOutcome) {
        const auto weights = values({0, 1, 0}, {3});
        const auto result = Tensor::multinomial(weights, 17, true, 42);
        ASSERT_EQ(result.shape(), TensorShape({17}));
        ASSERT_EQ(result.dtype(), DataType::Int64);
        expect_placement(result);
        EXPECT_EQ(result.to_vector_int64(), std::vector<int64_t>(17, 1));
    }

    TEST_P(TensorOperationContracts, EmptyReadbackUsesTheSameCompletionContract) {
        TensorReadback readback;
        auto empty = Tensor::empty({0}, device());
        readback.enqueue(empty);
        empty = {};
        EXPECT_TRUE(readback.pending());
        EXPECT_NO_THROW(readback.wait(std::span<std::byte>{}));
        EXPECT_FALSE(readback.pending());
    }

    TEST_P(TensorOperationContracts, ReadbackRetainsStridedSourceAndCanBeReused) {
        TensorReadback download;
        for (const float offset : {0.f, 8.f}) {
            auto source = values({1 + offset, 2 + offset, 3 + offset, 4 + offset, 5 + offset, 6 + offset}, {2, 3}).transpose(0, 1);
            download.enqueue(source);
            source = {};
            std::array<float, 6> actual{};
            download.wait(std::as_writable_bytes(std::span(actual)));
            EXPECT_FALSE(download.pending());
            EXPECT_EQ(actual, (std::array<float, 6>{1 + offset, 4 + offset, 2 + offset, 5 + offset, 3 + offset, 6 + offset}));
        }
    }

    TEST_P(TensorOperationContracts, ProtocolExportReportsUnavailableWithoutActivatingOtherBackends) {
        const auto input = values({1, 2, 3}, {3});
        if (!GetParam().backend) {
            const auto device = tensor_dlpack_device(input);
            EXPECT_EQ(device.type, 1);
            EXPECT_EQ(device.ordinal, 0);
            EXPECT_EQ(prepare_tensor_dlpack_export(input, std::nullopt).data_ptr(), input.data_ptr());
        } else if (!gpu_backend_export_capabilities(*GetParam().backend).dlpack) {
            EXPECT_THROW((void)tensor_dlpack_device(input), std::runtime_error);
            EXPECT_THROW((void)prepare_tensor_dlpack_export(input, -1), std::runtime_error);
        }
        for (const auto backend : kGpuBackends) {
            const bool live = gpu_backend_live(backend);
            (void)gpu_backend_export_capabilities(backend);
            EXPECT_EQ(gpu_backend_live(backend), live);
        }
    }

    TEST_P(TensorOperationContracts, PackedReadbackRetainsBorrowedStagingUntilCompletion) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Packed slots target GPU queues";
        const auto backend = *GetParam().backend;
        TensorWorkQueue queue(backend, TensorWorkQueue::Mode::Independent);
        auto staging = Tensor::empty({4 * sizeof(float)}, device(), DataType::UInt8);
        TensorReadbackRing ring(backend, 1, 4 * sizeof(float), queue, &staging);
        auto source = values({9, 1, 2, 3, 4, 8}, {6});
        ring.enqueue(source, sizeof(float), 4 * sizeof(float), 0, 0, true);
        source = {};
        staging = {};
        ring.seal(0).wait();
        ASSERT_TRUE(ring.poll(0));
        std::array<float, 4> actual{};
        const auto bytes = ring.slot_bytes(0);
        std::memcpy(actual.data(), bytes.data(), sizeof(actual));
        EXPECT_EQ(actual, (std::array<float, 4>{1, 2, 3, 4}));
        ring.release(0);
    }

    TEST_P(TensorOperationContracts, ScalarSelectionPreservesExactAndOverlappingAliases) {
        const auto condition = Tensor::from_vector(std::vector<int>{1, 0, 1, 0}, {4}, device()).to(DataType::Bool);
        for (const auto dtype : {DataType::Float32, DataType::Float16}) {
            auto input = values({1, 2, 3, 4, 5}, {5}).to(dtype);
            auto source = input.slice(0, 0, 4);
            auto output = input.slice(0, 1, 5);
            where_into(output, condition, 9.f, source);
            ASSERT_NO_FATAL_FAILURE(expect_values(output.to(DataType::Float32), {4}, {9, 2, 9, 4}));
            where_into(output, condition, 7.f, output);
            ASSERT_NO_FATAL_FAILURE(expect_values(output.to(DataType::Float32), {4}, {7, 2, 7, 4}));
        }
    }

    TEST_P(TensorOperationContracts, PackedReadbackReusesSlotsAndPreservesStagedRanges) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Packed pinned slots target GPU queues";
        const auto backend = *GetParam().backend;
        TensorWorkQueue queue(backend, static_cast<void*>(nullptr));
        TensorReadbackRing ring(backend, 1, 6 * sizeof(float), queue);
        for (const float offset : {0.f, 8.f}) {
            auto source = values({1 + offset, 2 + offset, 3 + offset, 4 + offset, 5 + offset, 6 + offset}, {6});
            ring.enqueue(source, 0, 3 * sizeof(float), 0, 0);
            ring.enqueue(source, 3 * sizeof(float), 3 * sizeof(float), 0, 3 * sizeof(float), true);
            source = {};
            ring.seal(0).wait();
            ASSERT_TRUE(ring.poll(0));
            std::array<float, 6> actual{};
            const auto bytes = ring.slot_bytes(0);
            ASSERT_EQ(bytes.size(), sizeof(actual));
            std::memcpy(actual.data(), bytes.data(), sizeof(actual));
            EXPECT_EQ(actual, (std::array<float, 6>{1 + offset, 2 + offset, 3 + offset, 4 + offset, 5 + offset, 6 + offset}));
            ring.release(0);
        }
    }

    TEST_P(TensorOperationContracts, InvalidIndicesNeverPublishStagedReadbackBytes) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Staged GPU readback requires GPU storage";
        const auto input = values({1, 2, 3}, {3});
        const auto indices = Tensor::from_vector(std::vector<int>{-1, 3}, {2}, device());
        std::array<float, 2> destination{-7, -8};
        EXPECT_ANY_THROW({
            const auto invalid = input.index_select(0, indices);
            TensorReadback download;
            download.enqueue(invalid);
            download.wait(std::as_writable_bytes(std::span(destination)));
        });
        EXPECT_EQ(destination, (std::array<float, 2>{-7, -8}));
    }

#ifdef LFS_TENSOR_HIP
    class HipReadbackSnapshots : public lfs::test::TensorBackendContract {};

    TEST_P(HipReadbackSnapshots, QueueDestructionReleasesUnconsumedExecutionFaults) {
        for (const int mode : {0, 1, 2}) {
            SCOPED_TRACE(mode);
            {
                const auto input = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device());
                const auto indices = Tensor::from_vector(std::vector<int>{-1, 3}, {2}, device());
                Tensor invalid;
                TensorWorkQueue queue(*GetParam().backend, static_cast<void*>(nullptr));
                if (mode)
                    (void)queue.execute([&] { invalid = input.index_select(0, indices); });
                else
                    invalid = input.index_select(0, indices);
                // Mode 2 consumes the context fault, leaving only last.wait() to fail.
                if (mode == 2)
                    EXPECT_ANY_THROW(queue.wait());
            }
            EXPECT_TRUE(tensor_backend_selftest(*GetParam().backend));
        }
    }

    TEST_P(HipReadbackSnapshots, EmptyBufferedReadbackRetainsExecutionFault) {
        const auto input = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device());
        const auto indices = Tensor::from_vector(std::vector<int>{-1, 3}, {2}, device());
        TensorWorkQueue queue(*GetParam().backend, static_cast<void*>(nullptr));
        const auto invalid = input.index_select(0, indices);
        TensorReadback download;
        download.enqueue_range(invalid, 0, 0);
        EXPECT_ANY_THROW(download.wait(std::span<std::byte>{}));
        EXPECT_TRUE(download.pending());
        EXPECT_ANY_THROW(queue.wait());
        EXPECT_ANY_THROW(download.wait(std::span<std::byte>{}));
        TensorReadback valid;
        valid.enqueue_range(input, 0, 0);
        EXPECT_NO_THROW(valid.wait(std::span<std::byte>{}));
        EXPECT_FALSE(valid.pending());
    }

    TEST_P(HipReadbackSnapshots, RepeatedReadinessQueriesDoNotDiscardFaults) {
        const auto input = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device());
        const auto indices = Tensor::from_vector(std::vector<int>{-1, 3}, {2}, device());
        TensorWorkQueue queue(*GetParam().backend, static_cast<void*>(nullptr));
        TensorFence fence(*GetParam().backend);
        const auto invalid = input.index_select(0, indices);
        const std::array<const Tensor*, 1> inputs{&invalid};
        TensorCompletion completion(inputs);
        fence.record(static_cast<void*>(nullptr));
        EXPECT_ANY_THROW(queue.wait());
        for (int attempt = 0; attempt < 2; ++attempt) {
            EXPECT_ANY_THROW((void)fence.ready());
            EXPECT_ANY_THROW((void)completion.ready());
        }
        EXPECT_ANY_THROW(fence.wait());
        EXPECT_ANY_THROW(completion.wait());
    }

    TEST_P(HipReadbackSnapshots, FailedUploadsRetainTheirErrorAndRetireOwnedStorage) {
        for (const bool poll_first : {false, true}) {
            SCOPED_TRACE(poll_first);
            {
                const auto input = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device());
                const auto indices = Tensor::from_vector(std::vector<int>{-1, 3}, {2}, device());
                auto destination = Tensor::zeros({3}, device());
                const auto source = Tensor::from_vector(std::vector<float>{4, 5, 6}, {3}, Device::CPU);
                TensorWorkQueue queue(*GetParam().backend, static_cast<void*>(nullptr));
                const auto invalid = input.index_select(0, indices);
                TensorUpload upload;
                upload.enqueue(destination, source);
                EXPECT_ANY_THROW(queue.wait());
                if (poll_first) {
                    EXPECT_ANY_THROW((void)upload.poll());
                    EXPECT_ANY_THROW((void)upload.poll());
                }
                EXPECT_ANY_THROW(upload.wait());
            }
            // No transfer, Tensor or queue lease may be stranded by a failed
            // upload destructor, even when the runtime defers native cleanup.
            EXPECT_TRUE(tensor_backend_selftest(*GetParam().backend));
        }
    }

    TEST_P(HipReadbackSnapshots, PackedFaultSurvivesContextConsumption) {
        const auto input = Tensor::from_vector(std::vector<float>{1, 2, 3}, {3}, device());
        const auto indices = Tensor::from_vector(std::vector<int>{-1, 3}, {2}, device());
        const auto invalid = input.index_select(0, indices);
        TensorWorkQueue queue(*GetParam().backend, static_cast<void*>(nullptr));
        TensorReadbackRing ring(*GetParam().backend, 1, 2 * sizeof(float), queue);
        ring.enqueue(invalid, 0, 2 * sizeof(float), 0, 0);
        EXPECT_ANY_THROW(queue.wait());
        // The marker is recorded after the context fault has been consumed;
        // only the packed slot still owns the failed snapshot.
        EXPECT_NO_THROW(ring.seal(0).wait());
        EXPECT_ANY_THROW((void)ring.poll(0));
        EXPECT_ANY_THROW(ring.wait(0));
        EXPECT_ANY_THROW((void)ring.slot_bytes(0));
        EXPECT_ANY_THROW(ring.release(0));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, HipReadbackSnapshots,
                             testing::Values(lfs::test::TensorBackendTarget{"HIP", GpuBackend::HIP}),
                             lfs::test::tensor_backend_parameter_name);
#endif

    TEST_P(TensorOperationContracts, CompiledExpressionsUseHostParametersAndTerminalFolds) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Compiled expression kernels target GPU backends";
        fused::Builder builder(2);
        const auto input = builder.input(DataType::Float32, 2);
        const auto gain = builder.input(DataType::Float32, 1);
        builder.output(input.load() * gain.at({0}) + builder.iota(1).cast(DataType::Float32), DataType::Float32);
        const fused::Kernel kernel(builder);
        const auto host_gain = Tensor::from_vector(std::vector<float>{2.f}, {1}, Device::CPU);
        const auto image = values({1, 2, 3, 4, 5, 6}, {2, 3});
        const auto output = Tensor::empty({2, 3}, Device::GPU);
        kernel.prepare({2, 3}, {image, host_gain}, {output});
        for (int repeat = 0; repeat < 2; ++repeat) {
            kernel({2, 3}, {image, host_gain}, {output});
            ASSERT_NO_FATAL_FAILURE(expect_values(output, {2, 3}, {2, 5, 8, 8, 11, 14}));
        }
        fused::Builder reduction(2);
        const auto term = reduction.input(DataType::Float32, 2).load();
        reduction.output(reduction.fold(fused::Fold::Sum, term, 1), DataType::Float32);
        const fused::Kernel sum(reduction);
        ASSERT_NO_FATAL_FAILURE(expect_values(sum({2, 3}, {image})[0], {2}, {6, 15}));
    }

    TEST_P(TensorOperationContracts, BackendSelftestCanRecreateItsExecutionContext) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Backend lifecycle self-tests target GPU contexts";
        const auto status = tensor_backend_selftest(*GetParam().backend);
        ASSERT_TRUE(status) << "Tensor backend self-test failed across shutdown/recreation";
        ASSERT_NO_FATAL_FAILURE(expect_values(values({1, 2, 3}, {3}).mul(2.f), {3}, {2, 4, 6}));
    }

    TEST_P(TensorOperationContracts, NeuralOperatorsMatchIndependentSmallReferences) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Native NN operators require GPU storage";
        const auto input = values({1, 2, 3, 4, 5, 6}, {2, 3});
        const auto weight = values({2, 0, 1, -1, 1, 0}, {2, 3});
        const auto bias = values({0.5f, -0.5f}, {2});
        ASSERT_NO_FATAL_FAILURE(expect_values(nn::linear(input, weight, &bias), {2, 2}, {5.5f, 0.5f, 14.5f, 0.5f}));
        const auto image = values({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
        const auto filter = values({1, 2, 3, 4}, {1, 1, 2, 2});
        const auto conv_bias = values({0.5f}, {1});
        ASSERT_NO_FATAL_FAILURE(expect_values(nn::conv2d(image, filter, &conv_bias, {}), {1, 1, 2, 2}, {37.5f, 47.5f, 67.5f, 77.5f}));
        const auto norm_input = values({1, 3, -1, 1}, {2, 2});
        const auto norm_weight = values({2, 3}, {2});
        ASSERT_NO_FATAL_FAILURE(expect_values(nn::layer_norm(norm_input, norm_weight, bias, 0.f), {2, 2}, {-1.5f, 2.5f, -1.5f, 2.5f}, 1e-6f));
        const auto query = values(std::vector<float>(4, 0.f), {1, 1, 2, 2});
        const auto key = values(std::vector<float>(6, 0.f), {1, 1, 3, 2});
        const auto value = values({1, 2, 3, 4, 5, 6}, {1, 1, 3, 2});
        ASSERT_NO_FATAL_FAILURE(expect_values(nn::attention(query, key, value), {1, 1, 2, 2}, {3, 4, 3, 4}, 1e-6f));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, TensorOperationContracts,
                             testing::ValuesIn(lfs::test::tensor_backend_targets()),
                             lfs::test::tensor_backend_parameter_name);
} // namespace
