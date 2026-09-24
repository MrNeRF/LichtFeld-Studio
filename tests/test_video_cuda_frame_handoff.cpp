/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_backend_test.hpp"

#include "io/video/cuda_frame_handoff.hpp"

#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <future>
#include <semaphore>
#include <stdexcept>
#include <vector>

namespace {

    using namespace std::chrono_literals;

    struct CallbackGate {
        std::binary_semaphore entered{0};
        std::binary_semaphore release{0};
    };

    void CUDART_CB waitAtGate(void* const data) {
        auto& gate = *static_cast<CallbackGate*>(data);
        gate.entered.release();
        gate.release.acquire();
    }

    struct CudaStream {
        CudaStream() {
            status = cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking);
        }

        ~CudaStream() {
            if (value)
                cudaStreamDestroy(value);
        }

        cudaStream_t value = nullptr;
        cudaError_t status = cudaSuccess;
    };

    struct CudaBuffer {
        explicit CudaBuffer(const std::size_t bytes) {
            status = cudaMalloc(&value, bytes);
        }

        ~CudaBuffer() {
            if (value)
                cudaFree(value);
        }

        void* value = nullptr;
        cudaError_t status = cudaSuccess;
    };

} // namespace

class VideoCudaFrameHandoff : public lfs::test::CudaBackendTest {};

TEST_F(VideoCudaFrameHandoff, WaitsForProducerAndConsumerStreams) {

    constexpr std::size_t bytes = 1U << 20;
    CudaStream producer;
    CudaStream consumer;
    CudaBuffer source(bytes);
    CudaBuffer destination(bytes);
    ASSERT_EQ(producer.status, cudaSuccess);
    ASSERT_EQ(consumer.status, cudaSuccess);
    ASSERT_EQ(source.status, cudaSuccess);
    ASSERT_EQ(destination.status, cudaSuccess);

    CUcontext current_context = nullptr;
    ASSERT_EQ(cuCtxGetCurrent(&current_context), CUDA_SUCCESS);
    ASSERT_NE(current_context, nullptr);

    CallbackGate producer_gate;
    CallbackGate consumer_gate;
    ASSERT_EQ(cudaLaunchHostFunc(producer.value, waitAtGate, &producer_gate), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(source.value, 0x5a, bytes, producer.value), cudaSuccess);
    const bool producer_started = producer_gate.entered.try_acquire_for(5s);
    if (!producer_started)
        producer_gate.release.release();
    ASSERT_TRUE(producer_started);

    AVCUDADeviceContext decoder_context{};
    decoder_context.cuda_ctx = current_context;
    decoder_context.stream = reinterpret_cast<CUstream>(producer.value);

    auto handoff = std::async(std::launch::async, [&]() {
        lfs::io::video::CudaFrameHandoff frame_handoff(
            decoder_context, consumer.value);
        if (const cudaError_t result = cudaMemcpyAsync(
                destination.value, source.value, bytes,
                cudaMemcpyDeviceToDevice, consumer.value);
            result != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(result));
        }
        if (const cudaError_t result = cudaLaunchHostFunc(
                consumer.value, waitAtGate, &consumer_gate);
            result != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(result));
        }
        frame_handoff.finish();
    });

    EXPECT_EQ(handoff.wait_for(20ms), std::future_status::timeout)
        << "handoff returned while producer work was blocked";
    producer_gate.release.release();

    const bool consumer_started = consumer_gate.entered.try_acquire_for(5s);
    EXPECT_TRUE(consumer_started);
    if (consumer_started) {
        EXPECT_EQ(handoff.wait_for(20ms), std::future_status::timeout)
            << "handoff returned while consumer work was blocked";
    }
    consumer_gate.release.release();

    ASSERT_EQ(handoff.wait_for(5s), std::future_status::ready);
    EXPECT_NO_THROW(handoff.get());

    std::vector<unsigned char> result(bytes);
    ASSERT_EQ(cudaMemcpy(result.data(), destination.value, bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(result.front(), 0x5a);
    EXPECT_EQ(result.back(), 0x5a);
}

TEST_F(VideoCudaFrameHandoff, UnwindWaitsForConsumerStream) {

    CudaStream producer;
    CudaStream consumer;
    ASSERT_EQ(producer.status, cudaSuccess);
    ASSERT_EQ(consumer.status, cudaSuccess);

    CUcontext current_context = nullptr;
    ASSERT_EQ(cuCtxGetCurrent(&current_context), CUDA_SUCCESS);
    ASSERT_NE(current_context, nullptr);

    AVCUDADeviceContext decoder_context{};
    decoder_context.cuda_ctx = current_context;
    decoder_context.stream = reinterpret_cast<CUstream>(producer.value);

    CallbackGate consumer_gate;
    auto unwind = std::async(std::launch::async, [&]() -> std::string {
        try {
            lfs::io::video::CudaFrameHandoff frame_handoff(
                decoder_context, consumer.value);
            if (const cudaError_t result = cudaLaunchHostFunc(
                    consumer.value, waitAtGate, &consumer_gate);
                result != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(result));
            }
            throw std::runtime_error("sentinel failure");
        } catch (const std::exception& error) {
            return error.what();
        }
    });

    const bool consumer_started = consumer_gate.entered.try_acquire_for(5s);
    if (!consumer_started)
        consumer_gate.release.release();
    ASSERT_TRUE(consumer_started);
    EXPECT_EQ(unwind.wait_for(20ms), std::future_status::timeout)
        << "AVFrame lifetime guard returned during pending consumer work";

    consumer_gate.release.release();
    ASSERT_EQ(unwind.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(unwind.get(), "sentinel failure");
}
