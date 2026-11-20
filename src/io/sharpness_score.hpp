/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>
#include <memory>
#include <optional>

namespace lfs::io {
    enum class SharpnessAlgorithm {
        LAPLACIAN, // Ratio of pixels with a strong Laplacian response
        TENENGRAD, // Sobel energy — directional blur detection
        COMBINED   // Tenengrad + Laplacian — best overall
    };

    // Sharp-edge pixel ratio (0-100), evaluated on the green channel of packed RGB8.
    // rgb must contain w*h*3 bytes for positive dimensions; images smaller than 3x3 score zero.
    [[nodiscard]] double computeSharpnessScore(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm);

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
