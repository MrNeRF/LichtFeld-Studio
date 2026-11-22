/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../sharpness_score.hpp"
#include <cuda_runtime_api.h>
#include <memory>
#include <optional>

namespace lfs::io {
    namespace detail {
        void launchSharpnessCount(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm,
                                  unsigned long long* count, cudaStream_t stream);
    }

    // One instance per extraction worker. CUDA resources are lazy and reused; failure permits CPU fallback.
    class SharpnessScorer {
    public:
        SharpnessScorer();
        ~SharpnessScorer();
        SharpnessScorer(const SharpnessScorer&) = delete;
        SharpnessScorer& operator=(const SharpnessScorer&) = delete;
        // Packed RGB8 on the current CUDA device, produced on the calling thread's default stream.
        // Synchronizes before returning. Unavailable resources return no value for CPU fallback;
        // execution errors propagate through the existing CUDA error boundary.
        [[nodiscard]] std::optional<double> scoreDevice(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm);

    private:
        struct State;
        std::unique_ptr<State> state_;
    };
} // namespace lfs::io
