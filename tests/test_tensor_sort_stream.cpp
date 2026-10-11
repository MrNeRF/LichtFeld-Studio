/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_hardening_test_utils.hpp"

#include "core/tensor/internal/cuda_stream_context.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <tuple>

using namespace lfs::core;
using namespace tensor_hardening;

namespace {

    class TensorStreamDataTest : public CudaTest {
    protected:
        void SetUp() override {
            CudaTest::SetUp();
            if (IsSkipped())
                return;
            std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "data/bicycle/sparse/0/points3D.bin",
                               std::ios::binary);
            if (!file)
                GTEST_SKIP() << "data/bicycle sparse points3D not available";

            uint64_t count = 0;
            file.read(reinterpret_cast<char*>(&count), sizeof(count));
            ASSERT_GE(count, sample_count);
            for (size_t i = 0; i < sample_count; ++i) {
                file.ignore(sizeof(uint64_t));
                double xyz[3];
                file.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
                file.ignore(3 + sizeof(double));
                uint64_t track_length = 0;
                file.read(reinterpret_cast<char*>(&track_length), sizeof(track_length));
                ASSERT_TRUE(file.good());
                file.ignore(static_cast<std::streamsize>(track_length * 2 * sizeof(uint32_t)));
                ASSERT_TRUE(file.good());
                samples.push_back(static_cast<float>(xyz[0]));
            }
        }

        static constexpr size_t sample_count = 256;
        std::vector<float> samples;
    };

    class TensorSortStreamTest : public TensorStreamDataTest,
                                 public ::testing::WithParamInterface<std::tuple<int, bool, bool>> {};

    TEST_P(TensorSortStreamTest, OrdersKernelOutputWithoutHostSynchronization) {
        const auto [axis, descending, cross_stream] = GetParam();
        const TensorShape shape = axis < 0 ? TensorShape{sample_count} : TensorShape{16, 16};
        const int dim = axis < 0 ? 0 : axis;
        const size_t dim_size = shape[dim];
        const size_t inner_size = axis == 0 ? 16 : 1;
        const size_t outer_size = sample_count / (dim_size * inner_size);
        std::vector<float> expected(sample_count);
        std::vector<int64_t> expected_indices(sample_count);
        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t inner = 0; inner < inner_size; ++inner) {
                const auto offset = [&](size_t d) { return outer * dim_size * inner_size + d * inner_size + inner; };
                std::vector<int64_t> order(dim_size);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
                    return descending ? samples[offset(a)] > samples[offset(b)] : samples[offset(a)] < samples[offset(b)];
                });
                for (size_t d = 0; d < dim_size; ++d) {
                    expected[offset(d)] = samples[offset(order[d])] * 2.0f;
                    expected_indices[offset(d)] = order[d];
                }
            }
        }

        GateStream producer;
        GateStream consumer;
        const auto execution_stream = cross_stream ? consumer.get() : producer.get();
        Tensor source;
        {
            CUDAStreamGuard guard(producer.get());
            source = Tensor::from_vector(samples, shape, Device::CUDA);
        }
        {
            CUDAStreamGuard producer_guard(producer.get());
            const auto warm = source.mul(-2.0f).contiguous();
            CUDAStreamGuard consumer_guard(execution_stream);
            const auto sorted = warm.sort(dim, descending);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }

        producer.close();
        Tensor produced;
        {
            CUDAStreamGuard guard(producer.get());
            produced = source.mul(2.0f).contiguous();
        }
        {
            CUDAStreamGuard guard(execution_stream);
            const auto [values, indices] = produced.sort(dim, descending);
            EXPECT_EQ(values.stream(), execution_stream);
            EXPECT_EQ(indices.stream(), execution_stream);
            EXPECT_EQ(values.cpu().to_vector(), expected);
            EXPECT_EQ(indices.cpu().to_vector_int64(), expected_indices);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    INSTANTIATE_TEST_SUITE_P(NonBlocking, TensorSortStreamTest,
                             ::testing::Combine(::testing::Values(-1, 0, 1),
                                                ::testing::Bool(), ::testing::Bool()));

    class TensorBoolReadbackStreamTest : public TensorStreamDataTest,
                                         public ::testing::WithParamInterface<DataType> {};

    TEST_P(TensorBoolReadbackStreamTest, WaitsForKernelOutput) {
        std::vector<bool> expected;
        for (const float value : samples)
            expected.push_back(value < 0.0f);

        GateStream producer;
        CUDAStreamGuard guard(producer.get());
        const auto source = Tensor::from_vector(samples, {sample_count}, Device::CUDA);
        {
            auto warm = source.lt(0.0f).to(GetParam());
            warm.copy_from(source.gt(0.0f).to(GetParam()));
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
        producer.close();
        const auto result = source.lt(0.0f).to(GetParam());
        EXPECT_EQ(result.stream(), producer.get());
        EXPECT_EQ(result.to_vector_bool(), expected);
        ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    }

    INSTANTIATE_TEST_SUITE_P(NonBlocking, TensorBoolReadbackStreamTest,
                             ::testing::Values(DataType::Bool, DataType::UInt8));

} // namespace
