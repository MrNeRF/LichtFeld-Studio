/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/tensor_scene_temporal_pipeline.hpp"

namespace lfs::vis {
    // One owner per viewport/offline session; three independent histories for
    // main and PLY comparison panels. No Vulkan objects cross this boundary.
    class LFS_VIS_API MetalSceneUpscaler {
    public:
        MetalSceneUpscaler();
        ~MetalSceneUpscaler();
        MetalSceneUpscaler(const MetalSceneUpscaler&) = delete;
        MetalSceneUpscaler& operator=(const MetalSceneUpscaler&) = delete;
        [[nodiscard]] lfs::Result<TensorSceneTemporalResult> resolve(
            SceneUpscalerBackend backend, const TensorSceneTemporalRequest& request);
        void reset(TemporalViewId view);
        void resetAll();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
