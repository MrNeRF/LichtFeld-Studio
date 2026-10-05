/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "rendering/scene_temporal_coordinator.hpp"
#include "rendering/scene_temporal_resolve.hpp"

#include <memory>

namespace lfs::core { class Tensor; }
namespace lfs::rendering { class TensorSceneTemporalKernels; }

namespace lfs::vis {
    struct TensorSceneTemporalRequest {
        TemporalViewId view = TemporalViewId::Main;
        std::shared_ptr<lfs::core::Tensor> color;
        std::shared_ptr<lfs::core::Tensor> depth;
        TemporalFrameInput frame;
        glm::ivec2 render_extent{0};
        glm::ivec2 output_extent{0};
        SceneTemporalResolveSettings settings;
        bool flip_y = false;
    };

    struct TensorSceneTemporalResult {
        std::shared_ptr<lfs::core::Tensor> color;
        std::uint64_t sequence = 0;
    };

    class TensorSceneTemporalPipeline {
    public:
        explicit TensorSceneTemporalPipeline(lfs::core::GpuBackend backend);
        ~TensorSceneTemporalPipeline();
        [[nodiscard]] lfs::Result<TensorSceneTemporalResult> resolve(
            const TensorSceneTemporalRequest& request);
        [[nodiscard]] lfs::Result<std::shared_ptr<lfs::core::Tensor>> spatial(
            const std::shared_ptr<lfs::core::Tensor>& color,
            glm::ivec2 render_extent, glm::ivec2 output_extent);
        void reset(TemporalViewId view,
                   TemporalResetReason reason = TemporalResetReason::HistoryDisabled);
        void resetAll(TemporalResetReason reason = TemporalResetReason::HistoryDisabled);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
