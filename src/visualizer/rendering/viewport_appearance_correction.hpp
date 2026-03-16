/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "rendering_types.hpp"
#include <memory>

namespace lfs::vis {

    class SceneManager;

    [[nodiscard]] std::shared_ptr<lfs::core::Tensor> applyViewportAppearanceCorrection(
        std::shared_ptr<lfs::core::Tensor> image,
        SceneManager* scene_manager,
        const RenderSettings& settings,
        int camera_uid);

} // namespace lfs::vis
