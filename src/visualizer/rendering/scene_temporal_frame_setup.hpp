/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "dirty_flags.hpp"
#include "rendering/temporal_frame_tracker.hpp"

namespace lfs::vis {
    struct SceneTemporalFrameSetupInput {
        bool backend_requested = false;
        bool runtime_ready = false;
        bool pipeline_eligible = true;
        bool projection_supported = true;
        bool equirectangular = false;
        bool appearance_correction = false;
        bool split_supported = true;
        bool raster_supported = true;
        bool interactive_scale = false;
        bool memory_pressure = false;
        bool lod_results_ready = false;
        bool lod_transition_active = false;
        DirtyMask frame_dirty = 0; // before LOD redraws are added
        DirtyMask training_refresh_dirty = 0;
        std::uint32_t settle_sample_count = TemporalConvergenceController::SAMPLE_COUNT;
        std::uint32_t jitter_phase_count = 0;
    };

    struct SceneTemporalFrameSetup {
        bool eligible = false;
        bool mode_unsupported = false;
        bool training_refresh_only = false;
        bool defer_convergence_until_idle = false;
        glm::vec2 jitter_pixels{0.0f};
    };

    [[nodiscard]] LFS_VIS_API SceneTemporalFrameSetup prepareSceneTemporalFrame(
        TemporalConvergenceController& convergence,
        const SceneTemporalFrameSetupInput& input);
} // namespace lfs::vis
