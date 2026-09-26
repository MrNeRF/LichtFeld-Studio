/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_backend_test.hpp"

#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include "core/tensor/backend/cuda/runtime/memory_pool.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include "cuda_stream_gate.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <span>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace std::chrono_literals;

    class TensorCompletionBackends : public testing::TestWithParam<GpuBackend> {};

    TEST_P(TensorCompletionBackends, CapturesJoinedProducersOnTheirStorageBackend) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP() << "Backend unavailable";
        std::array<Tensor, 2> outputs;
        for (size_t index = 0; index < outputs.size(); ++index) {
            std::jthread producer([&, index] {
                const GpuBackendScope scope(GetParam());
                outputs[index] = Tensor::full({65539}, static_cast<float>(index + 1), Device::GPU).mul(3.0f);
            });
        }
        const GpuBackendScope other(GetParam() == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        const Tensor host = Tensor::ones({4}, Device::CPU);
        const Tensor empty;
        const Tensor* tensors[] = {&outputs[0], &outputs[1], &outputs[0], &host, &empty, nullptr};
        auto completion = std::make_shared<TensorCompletion>(tensors);
        completion->wait();
        EXPECT_TRUE(completion->ready());
        completion->wait();
        for (size_t index = 0; index < outputs.size(); ++index) {
            EXPECT_EQ(gpu_backend_of(outputs[index]), GetParam());
            EXPECT_EQ(outputs[index].to_vector(), std::vector<float>(65539, static_cast<float>((index + 1) * 3)));
        }
    }

    TEST_P(TensorCompletionBackends, DestructionSettlesCapturedTensor) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP() << "Backend unavailable";
        const GpuBackendScope scope(GetParam());
        const Tensor output = Tensor::full({4096}, 6.0f, Device::GPU).square();
        {
            const Tensor* inputs[] = {&output};
            TensorCompletion completion(inputs);
        }
        EXPECT_EQ(output.to_vector(), std::vector<float>(4096, 36.0f));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, TensorCompletionBackends,
                             testing::ValuesIn(kGpuBackends),
                             [](const testing::TestParamInfo<GpuBackend>& info) {
                                 return info.param == GpuBackend::CUDA ? "Cuda" : "Vulkan";
                             });

    TEST(TensorCompletionHost, EmptyAndCpuInputsNeedNoGpu) {
        const Tensor host = Tensor::full({3}, 7.0f, Device::CPU);
        const Tensor invalid;
        const Tensor* inputs[] = {&host, &invalid, nullptr};
        TensorCompletion completion(inputs);
        EXPECT_TRUE(completion.ready());
        completion.wait();
        std::make_shared<TensorCompletion>()->wait();
        EXPECT_EQ(host.to_vector(), (std::vector<float>{7.0f, 7.0f, 7.0f}));
    }

    class TensorCompletionCuda : public lfs::test::CudaBackendTest {};

    TEST_F(TensorCompletionCuda, WaitAndDestructionSettleCapturedStreamWork) {
        for (bool explicit_wait : {true, false}) {
            SCOPED_TRACE(explicit_wait);
            cudaStream_t stream = nullptr;
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            {
                const CUDAStreamGuard stream_scope(stream);
                Tensor output = Tensor::zeros({1024}, Device::GPU);
                ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                const Tensor* warmup[] = {&output};
                std::make_shared<TensorCompletion>(warmup)->wait();
                lfs::test::CudaStreamGate gate;
                ASSERT_EQ(gate.block(stream), cudaSuccess);
                output.fill_(7.0f);
                const Tensor* inputs[] = {&output, &output};
                auto completion = std::make_shared<TensorCompletion>(inputs);
                EXPECT_FALSE(completion->ready());
                auto finished = std::async(std::launch::async, [completion = std::move(completion), explicit_wait]() mutable {
                    if (explicit_wait)
                        completion->wait();
                    completion.reset();
                });
                EXPECT_TRUE(gate.entered());
                EXPECT_EQ(finished.wait_for(20ms), std::future_status::timeout);
                gate.release();
                finished.get();
                EXPECT_EQ(cudaStreamQuery(stream), cudaSuccess);
                EXPECT_EQ(output.to_vector(), std::vector<float>(1024, 7.0f));
            }
            CudaMemoryPool::instance().release_stream(stream);
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
    }

    TEST_F(TensorCompletionCuda, CapturedWorkCompletesWhileAnotherStreamIsBlocked) {
        cudaStream_t captured = nullptr;
        cudaStream_t unrelated = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&captured, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&unrelated, cudaStreamNonBlocking), cudaSuccess);
        {
            const CUDAStreamGuard stream_scope(captured);
            Tensor output = Tensor::zeros({1024}, Device::GPU);
            ASSERT_EQ(cudaStreamSynchronize(captured), cudaSuccess);
            {
                lfs::test::CudaStreamGate gate;
                ASSERT_EQ(gate.block(unrelated), cudaSuccess);
                output.fill_(11.0f);
                TensorCompletion completion;
                completion.include(output);
                ASSERT_TRUE(gate.entered());
                auto waited = std::async(std::launch::async, [&] { completion.wait(); });
                EXPECT_EQ(waited.wait_for(1s), std::future_status::ready);
                EXPECT_TRUE(completion.ready());
                EXPECT_FALSE(gate.released());
                gate.release();
                waited.get();
            }
            EXPECT_EQ(output.to_vector(), std::vector<float>(1024, 11.0f));
        }
        CudaMemoryPool::instance().release_stream(captured);
        CudaMemoryPool::instance().release_stream(unrelated);
        EXPECT_EQ(cudaStreamDestroy(captured), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(unrelated), cudaSuccess);
    }

    TEST_F(TensorCompletionCuda, BackendScopeIncludesLaterWork) {
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        {
            lfs::test::CudaStreamGate gate;
            TensorCompletion completion;
            completion.include(GpuBackend::CUDA);
            ASSERT_EQ(gate.block(stream), cudaSuccess);
            ASSERT_TRUE(gate.entered());
            EXPECT_FALSE(completion.ready());
            auto finished = std::async(std::launch::async, [&] { completion.wait(); });
            EXPECT_EQ(finished.wait_for(20ms), std::future_status::timeout);
            gate.release();
            finished.get();
            EXPECT_TRUE(completion.ready());
        }
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST_F(TensorCompletionCuda, CoversDistinctStreams) {
        {
            std::array<cudaStream_t, 2> streams{};
            std::array<Tensor, 2> tensors;
            for (size_t i = 0; i < streams.size(); ++i) {
                ASSERT_EQ(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking), cudaSuccess);
                const CUDAStreamGuard guard(streams[i]);
                tensors[i] = Tensor::full({1024 * 1024}, static_cast<float>(i + 2), Device::GPU).square();
            }
            const Tensor* inputs[] = {&tensors[0], &tensors[1], &tensors[0]};
            auto completion = std::make_shared<TensorCompletion>(inputs);
            completion->wait();
            for (size_t i = 0; i < streams.size(); ++i) {
                EXPECT_EQ(cudaStreamQuery(streams[i]), cudaSuccess);
                EXPECT_EQ(tensors[i].to_vector(), std::vector<float>(1024 * 1024, static_cast<float>((i + 2) * (i + 2))));
                tensors[i] = {};
                CudaMemoryPool::instance().release_stream(streams[i]);
                EXPECT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
            }
        }
    }
    TEST_P(TensorCompletionBackends, QueueCopiesRetainCompletionAcrossReuse) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP() << "Backend unavailable";
        const GpuBackendScope scope(GetParam());
        TensorWorkQueue queue(GetParam(), nullptr, nullptr);
        TensorUpload upload;
        TensorReadback readback;
        Tensor output = Tensor::empty({4099}, Device::GPU, DataType::Int32);
        for (int iteration = 0; iteration < 3; ++iteration) {
            Tensor source = Tensor::full({4099}, float(iteration + 5), Device::CPU, DataType::Int32);
            auto completion = queue.execute([&] { upload.enqueue(output, source); });
            auto retained = completion;
            completion = {};
            retained.wait();
            EXPECT_TRUE(retained.ready());
            EXPECT_TRUE(upload.poll());
            readback.enqueue(output);
            std::vector<int> values(4099);
            readback.wait(std::as_writable_bytes(std::span(values)));
            EXPECT_EQ(values, std::vector<int>(4099, iteration + 5));
        }
    }

    TEST_P(TensorCompletionBackends, TensorUploadRetainsByteSpanOnExplicitQueue) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP() << "Backend unavailable";
        const GpuBackendScope scope(GetParam());
        Tensor destination = Tensor::empty({4}, Device::GPU, DataType::Int64);
        TensorUpload upload;
        std::array<int64_t, 4> source{};
        constexpr std::array<std::size_t, 5> sizes{4, 2, 3, 1, 4};
        cudaStream_t stream = nullptr;
        void* queue = nullptr;
        if (GetParam() == GpuBackend::CUDA) {
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            queue = stream;
        }
        if (GetParam() == GpuBackend::Vulkan) {
            EXPECT_THROW(upload.enqueue(destination, std::as_bytes(std::span(source)),
                                        reinterpret_cast<void*>(1)),
                         std::invalid_argument);
        }
        for (std::size_t iteration = 0; iteration < sizes.size(); ++iteration) {
            const auto count = sizes[iteration];
            const auto value = static_cast<int64_t>(iteration);
            source = {3 + value, 5 + value, 7 + value, 11 + value};
            auto output = destination.slice(0, 0, count);
            upload.enqueue(output,
                           std::as_bytes(std::span(source).first(count)),
                           queue);
            const std::vector<int64_t> expected(source.begin(), source.begin() + count);
            source.fill(0);
            upload.wait();
            EXPECT_EQ(output.cpu().to_vector_int64(), expected);
        }
        if (stream != nullptr) {
            CudaMemoryPool::instance().release_stream(stream);
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }
    }

    TEST_P(TensorCompletionBackends, CompletionOutlivesItsQueue) {
        if (!gpu_backend_available(GetParam()))
            GTEST_SKIP() << "Backend unavailable";
        const GpuBackendScope scope(GetParam());
        TensorCompletion retained;
        Tensor output;
        {
            TensorWorkQueue queue(GetParam(), nullptr, nullptr);
            retained = queue.execute([&] { output = Tensor::full({1024}, 3.0f, Device::GPU); });
            queue.execute([] {});
        }
        retained.wait();
        EXPECT_TRUE(retained.ready());
        EXPECT_EQ(output.to_vector(), std::vector<float>(1024, 3.0f));
    }

    TEST_F(TensorCompletionCuda, IndexSelectDoesNotWaitForAnUnfinishedProducer) {
#ifndef NDEBUG
        GTEST_SKIP() << "Debug index validation waits for the producer";
#endif
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP() << "CUDA unavailable";
        const GpuBackendScope scope(GpuBackend::CUDA);
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        {
            const CUDAStreamGuard stream_scope(stream);
            const Tensor input = Tensor::full({1024}, 7.0f, Device::GPU);
            const Tensor indices = Tensor::from_vector(std::vector<int>{0, 7, 23}, {3}, Device::GPU);
            (void)input.index_select(0, indices).cpu();
            lfs::test::CudaStreamGate gate;
            ASSERT_EQ(gate.block(stream), cudaSuccess);
            ASSERT_TRUE(gate.entered());
            auto pending = std::async(std::launch::async, [&] {
                const GpuBackendScope backend(GpuBackend::CUDA);
                const CUDAStreamGuard stream_scope(stream);
                return input.index_select(0, indices);
            });
            const auto state = pending.wait_for(1s);
            gate.release();
            EXPECT_EQ(state, std::future_status::ready);
            EXPECT_EQ(pending.get().to_vector(), std::vector<float>(3, 7.0f));
        }
        CudaMemoryPool::instance().release_stream(stream);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST(TensorQueue, ModesAndGpuFenceDoNotWaitOnHost) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        const GpuBackendScope backend(GpuBackend::CUDA);
        TensorWorkQueue producer(GpuBackend::CUDA, TensorWorkQueue::Mode::LegacyOrdered);
        TensorWorkQueue consumer(GpuBackend::CUDA);
        unsigned flags = 0;
        ASSERT_EQ(cudaStreamGetFlags(static_cast<cudaStream_t>(producer.native_handle()), &flags), cudaSuccess);
        EXPECT_EQ(flags, cudaStreamDefault);
        ASSERT_EQ(cudaStreamGetFlags(static_cast<cudaStream_t>(consumer.native_handle()), &flags), cudaSuccess);
        EXPECT_EQ(flags, cudaStreamNonBlocking);
        TensorFence fence(GpuBackend::CUDA);
        lfs::test::CudaStreamGate gate;
        ASSERT_EQ(gate.block(static_cast<cudaStream_t>(producer.native_handle())), cudaSuccess);
        ASSERT_TRUE(gate.entered());
        std::atomic<bool> callback_ran{false};
        auto submitted = std::async(std::launch::async, [&] {
            producer.record(fence);
            consumer.wait_for(fence);
            consumer.enqueue_host_callback([](void* value) {
                static_cast<std::atomic<bool>*>(value)->store(true);
            },
                                           &callback_ran);
        });
        const auto status = submitted.wait_for(1s);
        EXPECT_FALSE(callback_ran.load());
        EXPECT_FALSE(fence.ready());
        EXPECT_FALSE(consumer.ready());
        gate.release();
        EXPECT_EQ(status, std::future_status::ready);
        submitted.get();
        consumer.wait();
        EXPECT_TRUE(callback_ran.load());
        EXPECT_TRUE(fence.ready());
        // Re-record the same event, then consume that generation on the GPU.
        producer.record(fence);
        consumer.wait_for(fence);
        consumer.wait();
    }

    TEST(TensorQueue, UnsupportedVulkanOperationsAreExplicit) {
        EXPECT_THROW(TensorWorkQueue(GpuBackend::Vulkan), std::runtime_error);
        EXPECT_THROW(TensorFence(GpuBackend::Vulkan), std::runtime_error);
    }

    TEST(TensorQueue, PackedReadbackPreservesSourceAndDestinationOffsets) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP();
        const GpuBackendScope backend(GpuBackend::CUDA);
        TensorWorkQueue queue(GpuBackend::CUDA);
        TensorWorkQueue::Scope scope(queue);
        TensorReadbackRing ring(GpuBackend::CUDA, 2, 24, queue);
        auto source = Tensor::from_vector(std::vector<float>{1, 3, 5, 7, 9, 11}, {6}, Device::GPU);
        // Queue a blocked producer after upload; enqueue/seal/poll must return
        // while that producer is still blocked, without exposing partial bytes.
        queue.wait();
        lfs::test::CudaStreamGate gate;
        ASSERT_EQ(gate.block(static_cast<cudaStream_t>(queue.native_handle())), cudaSuccess);
        ASSERT_TRUE(gate.entered());
        ring.enqueue(source, sizeof(float), 2 * sizeof(float), 0, 4, true);
        ring.enqueue(source, 4 * sizeof(float), sizeof(float), 0, 16);
        ring.enqueue(source, 2 * sizeof(float), 2 * sizeof(float), 1, 0);
        ring.seal(0);
        ring.seal(1);
        EXPECT_FALSE(gate.released());
        EXPECT_FALSE(ring.poll(0));
        EXPECT_THROW(ring.release(0), std::logic_error);
        gate.release();
        source = {};
        EXPECT_THROW(ring.enqueue(Tensor{}, 0, 0, 0, 0), std::logic_error);
        ring.wait(0);
        ring.wait(1);
        EXPECT_TRUE(ring.poll(0));
        std::array<float, 6> values{};
        std::memcpy(values.data(), ring.slot_bytes(0).data(), 24);
        EXPECT_EQ(values, (std::array<float, 6>{0, 3, 5, 0, 9, 0}));
        std::memcpy(values.data(), ring.slot_bytes(1).data(), 24);
        EXPECT_EQ(values[0], 5);
        EXPECT_EQ(values[1], 7);
        ring.release(0);
        ring.release(1);
        auto next = Tensor::full({3}, 13.f, Device::GPU);
        EXPECT_THROW(ring.enqueue(next, 12, 1, 0, 0), std::out_of_range);
        EXPECT_THROW(ring.enqueue(next, 0, 4, 0, 22), std::out_of_range);
        ring.enqueue(next, 0, 12, 0, 0, true);
        ring.seal(0);
        ring.wait(0);
        std::memcpy(values.data(), ring.slot_bytes(0).data(), 12);
        EXPECT_EQ(values[0], 13);
        EXPECT_EQ(values[2], 13);
    }

} // namespace
