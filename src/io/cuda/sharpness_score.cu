/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error_typed.hpp"
#include "core/error_reporter.hpp"
#include "sharpness_edge.hpp"
#include "sharpness_score.cuh"
#include <algorithm>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

namespace lfs::io {
    namespace {
        constexpr int BLOCK_SIZE = 256;
        __global__ void countEdges(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm,
                                   unsigned long long* total) {
            const auto pixels = static_cast<unsigned long long>(w - 2) * (h - 2);
            const auto stride = static_cast<std::ptrdiff_t>(w) * 3;
            unsigned long long count = 0;
            for (auto index = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
                 index < pixels; index += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
                const auto y = index / (w - 2) + 1;
                const auto x = index % (w - 2) + 1;
                count += detail::sharpEdge(rgb + y * stride + x * 3 + 1, stride, algorithm);
            }
            using Reduce = cub::BlockReduce<unsigned long long, BLOCK_SIZE>;
            __shared__ typename Reduce::TempStorage storage;
            const auto sum = Reduce(storage).Sum(count);
            if (threadIdx.x == 0)
                atomicAdd(total, sum);
        }
        void launchSharpnessCount(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm,
                                  unsigned long long* count, cudaStream_t stream) {
            const auto pixels = static_cast<unsigned long long>(w - 2) * (h - 2);
            const int blocks = static_cast<int>(std::min<unsigned long long>((pixels + BLOCK_SIZE - 1) / BLOCK_SIZE, 4096));
            countEdges<<<blocks, BLOCK_SIZE, 0, stream>>>(rgb, w, h, algorithm, count);
            LFS_CUDA_LAUNCH_CHECK(stream, "io.video.sharpness_count");
        }
    } // namespace

    struct SharpnessScorer::State {
        cudaStream_t stream = nullptr;
        cudaEvent_t ready = nullptr;
        unsigned long long* count = nullptr;
        bool usable = true;

        ~State() {
            if (stream)
                LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "sharpness stream cleanup");
            if (count)
                LFS_CUDA_LOG_TEARDOWN(cudaFree(count), stream, "sharpness counter cleanup");
            if (ready)
                LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(ready), stream, "sharpness event cleanup");
            if (stream)
                LFS_CUDA_LOG_STREAM_DESTROY(stream, "sharpness stream destroy");
        }

        bool prepare() {
            if (!usable)
                return false;
            if (stream)
                return true;
            try {
                LFS_CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), nullptr, "sharpness stream initialization");
                LFS_CUDA_TRY(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), stream, "sharpness event initialization");
                LFS_CUDA_TRY(cudaMalloc(&count, sizeof(*count)), stream, "sharpness counter initialization");
                return true;
            } catch (const lfs::Exception& error) {
                core::ErrorReporter::get().report(error.error(), core::ReportChannel::OwnerLog);
                usable = false;
                // Initialization failure must not poison the extractor's subsequent CUDA checks.
                static_cast<void>(cudaGetLastError());
                return false;
            }
        }

        double evaluate(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm) {
            LFS_CUDA_TRY(cudaMemsetAsync(count, 0, sizeof(*count), stream), stream, "sharpness counter reset");
            launchSharpnessCount(rgb, w, h, algorithm, count, stream);
            unsigned long long edges = 0;
            LFS_CUDA_TRY(cudaMemcpyAsync(&edges, count, sizeof(edges), cudaMemcpyDeviceToHost, stream), stream, "sharpness count readback");
            LFS_CUDA_TRY(cudaStreamSynchronize(stream), stream, "sharpness completion");
            return detail::edgeRatio(edges, w, h);
        }
    };

    SharpnessScorer::SharpnessScorer() = default;
    SharpnessScorer::~SharpnessScorer() = default;

    std::optional<double> SharpnessScorer::scoreDevice(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm) {
        if (w < 3 || h < 3)
            return 0.0;
        if (!state_)
            state_ = std::make_unique<State>();
        if (!state_->prepare())
            return std::nullopt;
        LFS_CUDA_TRY(cudaEventRecord(state_->ready, nullptr), nullptr, "sharpness producer ready");
        LFS_CUDA_TRY(cudaStreamWaitEvent(state_->stream, state_->ready, 0), state_->stream, "sharpness producer wait");
        return state_->evaluate(rgb, w, h, algorithm);
    }
} // namespace lfs::io
