/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_event_pool.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "tensor_hardening_test_utils.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

namespace {

    constexpr size_t kValueCount = 4096;
    constexpr size_t kValueBytes = kValueCount * sizeof(float);

    void requireCudaDevice() {
        ASSERT_TRUE(tensor_hardening::has_cuda_device());
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    class DeviceBuffer {
    public:
        explicit DeviceBuffer(const size_t bytes)
            : bytes_(bytes) {
            EXPECT_EQ(cudaMalloc(&data_, bytes_), cudaSuccess);
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        ~DeviceBuffer() {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
        }

        [[nodiscard]] void* get() const { return data_; }
        [[nodiscard]] size_t bytes() const { return bytes_; }

    private:
        void* data_ = nullptr;
        size_t bytes_ = 0;
    };

    class PinnedHostBuffer {
    public:
        explicit PinnedHostBuffer(const size_t count)
            : count_(count) {
            EXPECT_EQ(cudaMallocHost(reinterpret_cast<void**>(&data_),
                                     count_ * sizeof(float)),
                      cudaSuccess);
        }

        PinnedHostBuffer(const PinnedHostBuffer&) = delete;
        PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

        ~PinnedHostBuffer() {
            if (data_ != nullptr) {
                cudaFreeHost(data_);
            }
        }

        [[nodiscard]] float* data() const { return data_; }
        [[nodiscard]] size_t size() const { return count_; }

    private:
        float* data_ = nullptr;
        size_t count_ = 0;
    };

    class TestStream {
    public:
        TestStream() {
            EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
        }

        TestStream(const TestStream&) = delete;
        TestStream& operator=(const TestStream&) = delete;

        ~TestStream() {
            tensor_hardening::destroy_stream_safely(stream_);
        }

        [[nodiscard]] cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
    };

    class LazyStateReset {
    public:
        LazyStateReset() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        }

        ~LazyStateReset() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        }
    };

    void primeAndEnqueueZero(tensor_hardening::GateStream& producer,
                             const DeviceBuffer& buffer) {
        ASSERT_EQ(cudaMemset(buffer.get(), 0x7f, buffer.bytes()), cudaSuccess);
        producer.close();
        ASSERT_EQ(cudaMemsetAsync(buffer.get(), 0, buffer.bytes(), producer.get()),
                  cudaSuccess);
    }

    void expectAllZero(const PinnedHostBuffer& values) {
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_FLOAT_EQ(values.data()[i], 0.0f) << "index " << i;
        }
    }

} // namespace

class TensorD4FromBlobHomeStreamTest : public ::testing::Test {};

TEST_F(TensorD4FromBlobHomeStreamTest, FromBlobCuda_DefaultHomeIsNull) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    ASSERT_NE(buffer.get(), nullptr);

    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32);

    EXPECT_EQ(blob.stream(), nullptr);
    EXPECT_FALSE(blob.owns_memory());
}

TEST_F(TensorD4FromBlobHomeStreamTest, FromBlobCuda_WithHome_StreamMetadata) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    TestStream home;
    TestStream retagged_home;
    ASSERT_NE(buffer.get(), nullptr);
    ASSERT_NE(home.get(), nullptr);
    ASSERT_NE(retagged_home.get(), nullptr);

    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32, home.get());
    ASSERT_EQ(blob.stream(), home.get());

    auto copy = blob;
    EXPECT_EQ(copy.stream(), home.get());
    EXPECT_FALSE(copy.owns_memory());

    copy.set_stream(retagged_home.get());
    EXPECT_EQ(copy.stream(), retagged_home.get());
    EXPECT_EQ(blob.stream(), home.get());
}

TEST_F(TensorD4FromBlobHomeStreamTest, FromBlobCuda_WithHome_ViewInheritsStream) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    TestStream home;
    ASSERT_NE(buffer.get(), nullptr);
    ASSERT_NE(home.get(), nullptr);

    auto blob = Tensor::from_blob(
        buffer.get(), {64, 64}, Device::CUDA, DataType::Float32, home.get());
    auto sliced = blob.slice(0, 0, 32);
    auto reshaped = blob.reshape({16, 256});
    auto unsqueezed = blob.unsqueeze(0);

    EXPECT_EQ(sliced.stream(), home.get());
    EXPECT_EQ(reshaped.stream(), home.get());
    EXPECT_EQ(unsqueezed.stream(), home.get());
}

TEST_F(TensorD4FromBlobHomeStreamTest, FromBlobCuda_WithHome_SyncToStreamOrdersReader) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    PinnedHostBuffer values(kValueCount);
    tensor_hardening::GateStream producer;
    TestStream consumer;
    ASSERT_NE(buffer.get(), nullptr);
    ASSERT_NE(values.data(), nullptr);

    primeAndEnqueueZero(producer, buffer);
    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32, producer.get());

    blob.sync_to_stream(consumer.get());
    ASSERT_EQ(cudaMemcpyAsync(values.data(), buffer.get(), kValueBytes,
                              cudaMemcpyDeviceToHost, consumer.get()),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer.get()), cudaSuccess);
    expectAllZero(values);
}

TEST_F(TensorD4FromBlobHomeStreamTest, FromBlobCuda_NullHome_RecordStreamIsStillNoop) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    TestStream reader;
    ASSERT_NE(buffer.get(), nullptr);

    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32);

    EXPECT_NO_THROW(blob.record_stream(reader.get()));
    EXPECT_EQ(blob.stream(), nullptr);
    EXPECT_FALSE(blob.owns_memory());
}

TEST_F(TensorD4FromBlobHomeStreamTest,
       FromBlobCuda_WithHome_RecordStreamRemainsNoopForFreeTracking) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    TestStream home;
    TestStream reader;
    ASSERT_NE(buffer.get(), nullptr);

    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32, home.get());

    EXPECT_NO_THROW(blob.record_stream(reader.get()));
    EXPECT_EQ(blob.stream(), home.get());
    EXPECT_FALSE(blob.owns_memory());
}

TEST_F(TensorD4FromBlobHomeStreamTest, FromBlobHost_HomeDefaultIgnored) {
    std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
    auto blob = Tensor::from_blob(
        values.data(), {values.size()}, Device::CPU, DataType::Float32);

    EXPECT_EQ(blob.stream(), nullptr);
    EXPECT_FALSE(blob.owns_memory());
    EXPECT_EQ(blob.to_vector(), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST_F(TensorD4FromBlobHomeStreamTest, GsplatContract_ArenaViewStamp_Unit) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    PinnedHostBuffer values(kValueCount);
    tensor_hardening::GateStream frame_stream;
    TestStream consumer;
    ASSERT_NE(buffer.get(), nullptr);
    ASSERT_NE(values.data(), nullptr);

    primeAndEnqueueZero(frame_stream, buffer);
    auto arena_view = Tensor::from_blob(
        buffer.get(), {1, 64, 64, 1}, Device::CUDA, DataType::Float32,
        frame_stream.get());

    const cudaStream_t execution =
        prepare_inputs_for_stream({&arena_view}, consumer.get());
    ASSERT_EQ(execution, consumer.get());
    ASSERT_EQ(cudaMemcpyAsync(values.data(), arena_view.data_ptr(), kValueBytes,
                              cudaMemcpyDeviceToHost, consumer.get()),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer.get()), cudaSuccess);
    expectAllZero(values);
}

TEST_F(TensorD4FromBlobHomeStreamTest, CpuOfStampedBlobOrdersAfterProducer) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    tensor_hardening::GateStream producer;
    ASSERT_NE(buffer.get(), nullptr);

    primeAndEnqueueZero(producer, buffer);
    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32, producer.get());

    const auto host = blob.cpu();
    EXPECT_EQ(host.to_vector(), std::vector<float>(kValueCount, 0.0f));
}

TEST_F(TensorD4FromBlobHomeStreamTest, ItemTypedDrainsStampedHome) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(sizeof(int));
    tensor_hardening::GateStream producer;
    ASSERT_NE(buffer.get(), nullptr);

    ASSERT_EQ(cudaMemset(buffer.get(), 0x7f, buffer.bytes()), cudaSuccess);
    producer.close();
    ASSERT_EQ(cudaMemsetAsync(buffer.get(), 0, buffer.bytes(), producer.get()),
              cudaSuccess);
    auto blob = Tensor::from_blob(
        buffer.get(), {1}, Device::CUDA, DataType::Int32, producer.get());

    EXPECT_EQ(blob.item<int>(), 0);
}

TEST_F(TensorD4FromBlobHomeStreamTest, FloatItemSynchronizesRegardlessOfHome) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(sizeof(float));
    tensor_hardening::GateStream producer;
    ASSERT_NE(buffer.get(), nullptr);

    ASSERT_EQ(cudaMemset(buffer.get(), 0x7f, buffer.bytes()), cudaSuccess);
    producer.close();
    ASSERT_EQ(cudaMemsetAsync(buffer.get(), 0, buffer.bytes(), producer.get()),
              cudaSuccess);
    auto blob = Tensor::from_blob(
        buffer.get(), {1}, Device::CUDA, DataType::Float32);

    EXPECT_FLOAT_EQ(blob.item(), 0.0f);
}

TEST_F(TensorD4FromBlobHomeStreamTest, PinOperandsOnNonOwningIsNoop) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    TestStream home;
    ASSERT_NE(buffer.get(), nullptr);

    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32, home.get());
    void* const original_data = blob.data_ptr();

    pin_operands({&blob});

    EXPECT_EQ(blob.data_ptr(), original_data);
    EXPECT_EQ(blob.stream(), home.get());
    EXPECT_FALSE(blob.owns_memory());
    EXPECT_FALSE(blob.is_deferred());
}

TEST_F(TensorD4FromBlobHomeStreamTest, StampedBlobThroughPinnedOpCrossStream) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    DeviceBuffer buffer(kValueBytes);
    tensor_hardening::GateStream producer;
    TestStream consumer;
    ASSERT_NE(buffer.get(), nullptr);

    primeAndEnqueueZero(producer, buffer);
    auto blob = Tensor::from_blob(
        buffer.get(), {kValueCount}, Device::CUDA, DataType::Float32, producer.get());

    Tensor result;
    {
        CUDAStreamGuard guard(consumer.get());
        const auto bias =
            Tensor::ones({kValueCount}, Device::CUDA, DataType::Float32);
        result = blob.add(bias);
    }

    ASSERT_EQ(result.stream(), consumer.get());
    EXPECT_EQ(result.cpu().to_vector(), std::vector<float>(kValueCount, 1.0f));
}

TEST_F(TensorD4FromBlobHomeStreamTest,
       DeferredSetStreamKeepsMetadataAndStaysDeferred) {
    if (!tensor_hardening::has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }
    requireCudaDevice();

    LazyStateReset lazy_state;
    TestStream target;
    auto deferred =
        Tensor::ones({64}, Device::CUDA, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.is_deferred());

    deferred.set_stream(target.get());

    EXPECT_TRUE(deferred.is_deferred());
    EXPECT_EQ(deferred.stream(), target.get());
}
