/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "sharpness_score.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/error_reporter.hpp"
#include "sharpness_edge.hpp"
#include <cuda_runtime.h>
#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace lfs::io {
    namespace {
        double edgeRatio(unsigned long long edges, int w, int h) {
            return static_cast<double>(edges) * 100.0 / static_cast<double>(static_cast<long long>(w) * h);
        }

        template <SharpnessAlgorithm Algorithm>
        long long countSharpEdges(const uint8_t* rgb, int w, int h) {
            const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(w) * 3;
            const auto count_rows = [&](const tbb::blocked_range<int>& rows, long long edges) {
                for (int y = rows.begin(); y < rows.end(); ++y) {
                    for (int x = 1; x < w - 1; ++x)
                        edges += detail::sharpEdge(rgb + static_cast<std::ptrdiff_t>(y) * stride + static_cast<std::ptrdiff_t>(x) * 3 + 1, stride, Algorithm);
                }
                return edges;
            };
            const tbb::blocked_range<int> rows(1, h - 1, 16);
            if (static_cast<long long>(w) * h < 256 * 256)
                return count_rows(rows, 0);
            return tbb::parallel_reduce(rows, 0LL, count_rows, std::plus<long long>{});
        }
    } // namespace

    [[nodiscard]] double computeSharpnessScore(const uint8_t* rgb, const int w, const int h, const SharpnessAlgorithm algo) {
        if (w < 3 || h < 3)
            return 0.0;
        long long edges;
        switch (algo) {
        case SharpnessAlgorithm::LAPLACIAN:
            edges = countSharpEdges<SharpnessAlgorithm::LAPLACIAN>(rgb, w, h);
            break;
        case SharpnessAlgorithm::COMBINED:
            edges = countSharpEdges<SharpnessAlgorithm::COMBINED>(rgb, w, h);
            break;
        default:
            edges = countSharpEdges<SharpnessAlgorithm::TENENGRAD>(rgb, w, h);
            break;
        }
        return edgeRatio(edges, w, h);
    }

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
            detail::launchSharpnessCount(rgb, w, h, algorithm, count, stream);
            unsigned long long edges = 0;
            LFS_CUDA_TRY(cudaMemcpyAsync(&edges, count, sizeof(edges), cudaMemcpyDeviceToHost, stream), stream, "sharpness count readback");
            LFS_CUDA_TRY(cudaStreamSynchronize(stream), stream, "sharpness completion");
            return edgeRatio(edges, w, h);
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
