/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "scene_temporal_frame_setup.hpp"

namespace lfs::vis {
    SceneTemporalFrameSetup prepareSceneTemporalFrame(
        TemporalConvergenceController& convergence,
        const SceneTemporalFrameSetupInput& input) {
        constexpr DirtyMask TEMPORAL_SOURCE_DIRTY =
            DirtyFlag::CAMERA | DirtyFlag::SPLATS | DirtyFlag::MESH |
            DirtyFlag::VIEWPORT | DirtyFlag::BACKGROUND | DirtyFlag::SPLIT_VIEW;
        SceneTemporalFrameSetup result;
        result.eligible = input.backend_requested && input.pipeline_eligible &&
                          input.projection_supported && !input.equirectangular &&
                          !input.appearance_correction && input.split_supported &&
                          input.raster_supported;
        result.mode_unsupported =
            input.backend_requested && !input.interactive_scale && !input.memory_pressure &&
            (!input.projection_supported || input.equirectangular ||
             input.appearance_correction || !input.split_supported);
        const DirtyMask independent_sources =
            input.frame_dirty & ~input.training_refresh_dirty & TEMPORAL_SOURCE_DIRTY;
        result.training_refresh_only =
            input.training_refresh_dirty != 0 && independent_sources == 0;
        const bool allow_settle = !result.training_refresh_only &&
                                  !input.lod_results_ready && !input.lod_transition_active;
        // Ready LOD results and transitions redraw the camera view.
        const DirtyMask dirty = input.frame_dirty |
                                (input.lod_results_ready || input.lod_transition_active
                                     ? DirtyMask{DirtyFlag::CAMERA}
                                     : DirtyMask{0});
        convergence.prepare(result.eligible,
                            (dirty & TEMPORAL_SOURCE_DIRTY) != 0,
                            allow_settle,
                            input.settle_sample_count,
                            input.jitter_phase_count);
        result.defer_convergence_until_idle =
            result.eligible && allow_settle && (input.frame_dirty & DirtyFlag::CAMERA) != 0;
        result.jitter_pixels = convergence.jitter();
        if (input.backend_requested && !input.runtime_ready)
            result.jitter_pixels = glm::vec2(0.0f);
        return result;
    }
} // namespace lfs::vis
