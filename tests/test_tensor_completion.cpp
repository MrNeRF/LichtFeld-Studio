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
#include <chrono>
#include <future>
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

} // namespace
