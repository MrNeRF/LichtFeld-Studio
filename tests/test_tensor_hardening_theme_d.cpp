/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_hardening_test_utils.hpp"

#include "core/tensor/internal/cuda_stream_context.hpp"

#include <algorithm>
#include <optional>

using namespace lfs::core;
using namespace tensor_hardening;
namespace {

    cudaStream_t make_consumer_stream() {
        cudaStream_t stream = nullptr;
        EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        return stream;
    }

    template <typename Callback>
    void invoke_while_producer_is_gated(GateStream&, const cudaStream_t consumer,
                                        Callback&& callback) {
        CUDAStreamGuard guard(consumer);
        callback();
    }

} // namespace

TEST_F(CudaTest, D1_CloneWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();

    Tensor source;
    {
        CUDAStreamGuard guard(producer.get());
        source = Tensor::zeros({1 << 20}, Device::CUDA);
    }
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    producer.close();
    source.fill_(3.0f, producer.get());

    Tensor result;
    invoke_while_producer_is_gated(producer, consumer, [&] { result = source.clone(); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);

    expect_float_values_match(
        result, torch::full({1 << 20}, 3.0f, torch::TensorOptions().device(torch::kCUDA)),
        "D1 clone producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D1_ContiguousWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();

    Tensor base;
    {
        CUDAStreamGuard guard(producer.get());
        base = Tensor::zeros({1024, 1024}, Device::CUDA);
    }
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    const auto view = base.transpose(0, 1);
    producer.close();
    base.fill_(4.0f, producer.get());

    Tensor result;
    invoke_while_producer_is_gated(producer, consumer, [&] { result = view.contiguous(); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);

    expect_float_values_match(
        result, torch::full({1024, 1024}, 4.0f, torch::TensorOptions().device(torch::kCUDA)),
        "D1 contiguous producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D1_DtypeConversionWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();

    Tensor source;
    {
        CUDAStreamGuard guard(producer.get());
        source = Tensor::zeros({1 << 20}, Device::CUDA);
    }
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    producer.close();
    source.fill_(5.0f, producer.get());

    Tensor result;
    invoke_while_producer_is_gated(
        producer, consumer, [&] { result = source.to(DataType::Int32); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);

    expect_int_values_match(
        result, torch::full({1 << 20}, 5, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)),
        "D1 dtype conversion producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D1_CopyFromWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();

    Tensor source;
    Tensor destination;
    {
        CUDAStreamGuard guard(producer.get());
        source = Tensor::zeros({1 << 20}, Device::CUDA);
    }
    {
        CUDAStreamGuard guard(consumer);
        destination = Tensor::zeros({1 << 20}, Device::CUDA);
    }
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    producer.close();
    source.fill_(6.0f, producer.get());

    invoke_while_producer_is_gated(
        producer, consumer, [&] { destination.copy_from(source); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);

    expect_float_values_match(
        destination, torch::full({1 << 20}, 6.0f, torch::TensorOptions().device(torch::kCUDA)),
        "D1 copy_from producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D1_BinaryInPlaceWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();

    Tensor source;
    Tensor destination;
    {
        CUDAStreamGuard guard(producer.get());
        source = Tensor::zeros({1 << 20}, Device::CUDA);
    }
    {
        CUDAStreamGuard guard(consumer);
        destination = Tensor::ones({1 << 20}, Device::CUDA);
    }
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    producer.close();
    source.fill_(2.0f, producer.get());

    invoke_while_producer_is_gated(
        producer, consumer, [&] { destination.add_(source); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);

    expect_float_values_match(
        destination, torch::full({1 << 20}, 3.0f, torch::TensorOptions().device(torch::kCUDA)),
        "D1 in-place producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D2_MMWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();
    constexpr int size = 256;

    Tensor input;
    {
        CUDAStreamGuard guard(producer.get());
        input = Tensor::zeros({size, size}, Device::CUDA);
    }
    const auto identity = Tensor::eye(size, Device::CUDA);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    producer.close();
    input.fill_(2.0f, producer.get());

    Tensor result;
    invoke_while_producer_is_gated(producer, consumer, [&] { result = input.mm(identity); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    expect_float_values_match(
        result, torch::full({size, size}, 2.0f, torch::TensorOptions().device(torch::kCUDA)),
        "D2 mm producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D2_DotWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();
    constexpr int count = 1 << 20;

    Tensor input;
    {
        CUDAStreamGuard guard(producer.get());
        input = Tensor::zeros({count}, Device::CUDA);
    }
    const auto ones = Tensor::ones({count}, Device::CUDA);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    producer.close();
    input.fill_(2.0f, producer.get());

    Tensor result;
    invoke_while_producer_is_gated(producer, consumer, [&] { result = input.dot(ones); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    expect_float_values_match(
        result, torch::tensor(static_cast<float>(count * 2), torch::TensorOptions().device(torch::kCUDA)),
        "D2 dot producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D2_DiagWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();
    constexpr int count = 1024;

    Tensor input;
    {
        CUDAStreamGuard guard(producer.get());
        input = Tensor::zeros({count}, Device::CUDA);
    }
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    producer.close();
    input.fill_(7.0f, producer.get());

    Tensor result;
    invoke_while_producer_is_gated(producer, consumer, [&] { result = Tensor::diag(input); });
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    expect_float_values_match(
        result, torch::diag(torch::full({count}, 7.0f, torch::TensorOptions().device(torch::kCUDA))),
        "D2 diag producer ordering");
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D2_MultinomialWaitsForGatedProducer) {
    GateStream producer;
    const cudaStream_t consumer = make_consumer_stream();

    Tensor weights;
    {
        CUDAStreamGuard guard(producer.get());
        weights = Tensor::zeros({2}, Device::CUDA);
    }
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    producer.close();
    weights.fill_(1.0f, producer.get());

    std::optional<Tensor> result;
    invoke_while_producer_is_gated(
        producer, consumer, [&] { result = Tensor::multinomial(weights, 4096, true); });
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);

    const auto values = result->cpu().to_vector_int64();
    ASSERT_EQ(values.size(), 4096u);
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](const int64_t value) {
        return value == 0 || value == 1;
    }));
    EXPECT_NE(std::count(values.begin(), values.end(), 0), 0);
    EXPECT_NE(std::count(values.begin(), values.end(), 1), 0);
    destroy_stream_safely(consumer);
}

TEST_F(CudaTest, D3_WhereMetadataIsNotReusedAcrossConcurrentStreams) {
    cudaStream_t first_stream = make_consumer_stream();
    cudaStream_t second_stream = make_consumer_stream();

    const auto first_condition = Tensor::ones({2048, 1}, Device::CUDA, DataType::Bool);
    const auto first_x = Tensor::ones({1, 2048}, Device::CUDA);
    const auto first_y = Tensor::zeros({2048, 2048}, Device::CUDA);
    const auto second_condition = Tensor::zeros({1, 8, 1}, Device::CUDA, DataType::Bool);
    const auto second_x = Tensor::ones({4, 1, 16}, Device::CUDA);
    const auto second_y = Tensor::full({4, 8, 16}, 2.0f, Device::CUDA);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    for (int iteration = 0; iteration < 20; ++iteration) {
        Tensor first_result;
        Tensor second_result;
        {
            CUDAStreamGuard guard(first_stream);
            first_result = Tensor::where(first_condition, first_x, first_y);
        }
        {
            CUDAStreamGuard guard(second_stream);
            second_result = Tensor::where(second_condition, second_x, second_y);
        }
        ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);

        const auto first_values = first_result.cpu().to_vector();
        ASSERT_EQ(first_values.size(), static_cast<size_t>(2048 * 2048));
        EXPECT_FLOAT_EQ(first_values.front(), 1.0f) << "iteration " << iteration;
        EXPECT_FLOAT_EQ(first_values[first_values.size() / 2], 1.0f) << "iteration " << iteration;
        EXPECT_FLOAT_EQ(first_values.back(), 1.0f) << "iteration " << iteration;
        const auto second_values = second_result.cpu().to_vector();
        EXPECT_TRUE(std::all_of(second_values.begin(), second_values.end(), [](const float value) {
            return value == 2.0f;
        })) << "iteration "
            << iteration;
    }

    destroy_stream_safely(second_stream);
    destroy_stream_safely(first_stream);
}

TEST_F(CudaTest, D4_OverlappingTransposeCopyUsesSnapshotSemantics) {
    const std::vector<float> values = {1, 2, 3, 4,
                                       5, 6, 7, 8,
                                       9, 10, 11, 12,
                                       13, 14, 15, 16};
    for (int iteration = 0; iteration < 100; ++iteration) {
        auto ours = lfs_float_tensor(values, {4, 4}, Device::CUDA);
        auto destination = ours.transpose(0, 1);
        destination.copy_from(ours);

        const auto theirs = torch::tensor(values, torch::TensorOptions().device(torch::kCUDA))
                                .reshape({4, 4})
                                .transpose(0, 1)
                                .contiguous();
        expect_float_values_match(ours, theirs,
                                  "D4 transpose copy iteration " + std::to_string(iteration));
    }
}

TEST_F(CudaTest, D4_OverlappingIndexSelectIntoUsesSnapshotSemantics) {
    const auto index = lfs_int_tensor({1, 0}, {2}, Device::CUDA);
    for (int iteration = 0; iteration < 100; ++iteration) {
        auto ours = lfs_float_tensor({1, 2, 3, 4}, {2, 2}, Device::CUDA);
        ours.index_select_into(ours, 0, index, BoundaryMode::Assert);
        const auto theirs = torch::tensor({3.0f, 4.0f, 1.0f, 2.0f},
                                          torch::TensorOptions().device(torch::kCUDA))
                                .reshape({2, 2});
        expect_float_values_match(ours, theirs,
                                  "D4 index_select_into iteration " + std::to_string(iteration));
    }
}

TEST_F(CudaTest, D5_AliasedScatterMatchesTorchOverlapContract) {
    const auto index = lfs_int_tensor({1, 2, 0}, {3}, Device::CUDA);
    auto ours = lfs_float_tensor({1, 2, 3}, {3}, Device::CUDA);
    bool ours_threw = false;
    try {
        ours.scatter_(0, index, ours);
    } catch (const std::exception&) {
        ours_threw = true;
    }

    auto theirs = torch::tensor({1.0f, 2.0f, 3.0f},
                                torch::TensorOptions().device(torch::kCUDA));
    const auto torch_index = torch::tensor({1, 2, 0},
                                           torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    bool torch_threw = false;
    try {
        theirs.scatter_(0, torch_index, theirs);
    } catch (const c10::Error&) {
        torch_threw = true;
    }

    EXPECT_EQ(ours_threw, torch_threw);
    if (!torch_threw) {
        expect_float_values_match(ours, theirs, "D5 aliased scatter");
    }
}
