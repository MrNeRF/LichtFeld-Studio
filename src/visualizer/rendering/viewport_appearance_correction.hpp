/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include "rendering_types.hpp"
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace lfs::vis {

    class SceneManager;

    [[nodiscard]] LFS_VIS_API std::shared_ptr<lfs::core::Tensor> applyViewportAppearanceCorrection(
        std::shared_ptr<lfs::core::Tensor> image,
        SceneManager* scene_manager,
        const RenderSettings& settings,
        int camera_uid);

    enum class ExportPostProcessMode {
        Opaque,              // [H,W,3] in -> [H,W,3] out, PPISP only
        Transparent,         // [H,W,4] in -> [H,W,4] out, PPISP on RGB, alpha preserved
        EnvironmentComposite // [H,W,4] in -> [H,W,3] out, PPISP then environment under alpha
    };

    struct ExportPostProcessView {
        glm::mat3 rotation{1.0f};
        float focal_length_mm = 0.0f;
        bool equirectangular_view = false;
        // Live viewport size; the AUTO-mode controller predicts from a thumbnail of
        // this size so exports match the on-screen appearance at every resolution.
        glm::ivec2 controller_predict_size{0, 0};
    };

    // Streamed GPU post-process for high-resolution exports: the full CPU u8 HWC
    // image is corrected (and composited) in bounded row bands so peak GPU memory
    // stays O(band) regardless of export size. Errors propagate; nothing is
    // silently exported uncorrected.
    [[nodiscard]] LFS_VIS_API std::expected<lfs::core::Tensor, std::string> applyExportPostProcess(
        lfs::core::Tensor image_u8_hwc_cpu,
        SceneManager* scene_manager,
        const RenderSettings& settings,
        int camera_uid,
        ExportPostProcessMode mode,
        const ExportPostProcessView& view);

} // namespace lfs::vis
