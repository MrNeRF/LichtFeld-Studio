/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/mesh_data.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/video/video_export_options.hpp"
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <vector>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::gui {

    struct VideoExportMeshSnapshot {
        std::shared_ptr<lfs::core::MeshData> mesh;
        glm::mat4 transform{1.0f};
        bool is_selected = false;
    };

    struct VideoExportCropBoxSnapshot {
        bool has_data = false;
        lfs::core::NodeId node_id = lfs::core::NULL_NODE;
        int parent_node_index = -1;
        lfs::core::CropBoxData data;
        glm::mat4 world_transform{1.0f};
    };

    struct VideoExportEllipsoidSnapshot {
        lfs::core::NodeId node_id = lfs::core::NULL_NODE;
        int parent_node_index = -1;
        lfs::core::EllipsoidData data;
        glm::mat4 world_transform{1.0f};
    };

    struct VideoExportSceneSnapshot {
        std::shared_ptr<lfs::core::SplatData> combined_model;
        std::shared_ptr<lfs::core::PointCloud> point_cloud;
        glm::mat4 point_cloud_transform{1.0f};
        std::vector<VideoExportMeshSnapshot> meshes;
        std::vector<glm::mat4> model_transforms;
        std::shared_ptr<lfs::core::Tensor> transform_indices;
        std::shared_ptr<lfs::core::Tensor> selection_mask;
        std::vector<bool> selected_node_mask;
        std::vector<bool> node_visibility_mask;
        std::vector<VideoExportCropBoxSnapshot> cropboxes;
        int selected_cropbox_index = -1;
        std::optional<VideoExportEllipsoidSnapshot> active_ellipsoid;

        [[nodiscard]] bool hasRenderableContent() const {
            return (combined_model && combined_model->size() > 0) ||
                   (point_cloud && point_cloud->size() > 0) ||
                   !meshes.empty();
        }
    };

    LFS_VIS_API std::expected<VideoExportSceneSnapshot, std::string> captureVideoExportSceneSnapshot(
        const lfs::vis::SceneManager& scene_manager);

    LFS_VIS_API std::expected<lfs::io::video::VideoExportOptions, std::string> validateVideoExportOptions(
        lfs::io::video::VideoExportOptions options);

} // namespace lfs::vis::gui
