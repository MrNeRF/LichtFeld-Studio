/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace lfs::rendering {
    struct TensorSceneMotionParameters {
        std::array<float, 16> inverse_current_view_projection{};
        std::array<float, 16> previous_view_projection{};
        std::array<std::uint32_t, 4> render_info{}; // width, height, flip_y, depth encoding
        std::array<float, 4> depth_info{};          // near, far, source flip_y, depth row stride (0 = render width)
    };

    struct TensorSceneResolveParameters {
        std::array<std::uint32_t, 4> extents{};       // render width/height, output width/height
        std::array<std::uint32_t, 4> current_layout{}; // allocation width/height, channels, float
        std::array<float, 4> control{};               // history valid, weight, motion rejection, unused
        std::array<float, 4> current_uv{};            // scale xy, clamp max zw
        std::array<float, 4> depth_control{};          // enabled, relative, absolute, far
        std::array<float, 4> jitter_pixels{};          // current xy, previous zw
        std::array<float, 4> reconstruction{};         // sharpness, motion confidence span
    };

    // Packed oracle records used by the same resolve core as the image pass.
    struct TensorSceneResolveSample {
        std::array<float, 4> current{}, history{};
        std::array<float, 4> neighborhood_min{}, neighborhood_max{}, neighborhood_cross_sum{};
        std::array<float, 4> pixel_motion{}; // current pixel center, motion
        std::array<float, 4> jitter{};       // current, previous
        std::array<std::uint32_t, 4> extents{};
        std::array<float, 4> depth{}; // current, history, far, history valid
        std::array<float, 4> settings0{}; // weight, relative depth, absolute depth, motion rejection
        std::array<float, 4> settings1{}; // confidence span, sharpness, depth available, unused
    };

    struct TensorSceneResolveResult {
        std::array<float, 4> color{};
        std::array<float, 4> uv{}; // current render uv, previous render uv
        std::array<float, 4> status{}; // previous uv, effective history weight, rejection
    };
    static_assert(sizeof(TensorSceneResolveSample) == 176);
    static_assert(sizeof(TensorSceneResolveResult) == 48);

    class TensorSceneTemporalKernels {
    public:
        explicit TensorSceneTemporalKernels(core::GpuBackend backend);
        ~TensorSceneTemporalKernels();
        TensorSceneTemporalKernels(const TensorSceneTemporalKernels&) = delete;
        TensorSceneTemporalKernels& operator=(const TensorSceneTemporalKernels&) = delete;

        [[nodiscard]] lfs::Result<void> motion(
            const core::Tensor& depth, const TensorSceneMotionParameters& parameters,
            core::Tensor& output);
        [[nodiscard]] lfs::Result<void> spatial(
            const core::Tensor& current, const TensorSceneResolveParameters& parameters,
            core::Tensor& output);
        [[nodiscard]] lfs::Result<void> resolve(
            const core::Tensor& current, const core::Tensor* history,
            const core::Tensor& motion, const core::Tensor* current_depth,
            const core::Tensor* history_depth,
            const TensorSceneResolveParameters& parameters, core::Tensor& output);
        [[nodiscard]] lfs::Result<void> resolveSamples(
            std::span<const TensorSceneResolveSample> samples, core::Tensor& output);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::rendering
