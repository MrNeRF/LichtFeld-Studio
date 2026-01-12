/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "scene.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis {

    // Snapshot of scene state for rendering
    struct SceneRenderState {
        const lfs::core::SplatData* combined_model = nullptr;
        const lfs::core::PointCloud* point_cloud = nullptr; // For pre-training point cloud rendering
        std::vector<glm::mat4> model_transforms;
        std::shared_ptr<lfs::core::Tensor> transform_indices; // Per-Gaussian index into model_transforms
        std::shared_ptr<lfs::core::Tensor> selection_mask;    // Per-Gaussian selection group ID
        std::vector<bool> selected_node_mask;                 // Per-node: true = selected, false = desaturate
        std::vector<bool> node_visibility_mask;               // Per-node: true = visible, false = culled (for consolidated models)
        std::string selected_node_name;
        std::vector<Scene::RenderableCropBox> cropboxes;
        int selected_cropbox_index = -1;
        bool has_selection = false;
        size_t visible_splat_count = 0;
    };

} // namespace lfs::vis
