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
        const lfs::core::MeshData* borrowed_mesh = nullptr;
        lfs::core::NodeId node_id = lfs::core::NULL_NODE;
        glm::mat4 transform{1.0f};
        bool is_selected = false;
        [[nodiscard]] const lfs::core::MeshData* meshData() const {
            return borrowed_mesh ? borrowed_mesh : mesh.get();
        }
    };

    struct VideoExportCropBoxSnapshot {
        bool has_data = false;
        lfs::core::NodeId node_id = lfs::core::NULL_NODE;
        lfs::core::NodeId parent_splat_id = lfs::core::NULL_NODE;
        int parent_node_index = -1;
        lfs::core::CropBoxData data;
        glm::mat4 world_transform{1.0f};
    };

    struct VideoExportEllipsoidSnapshot {
        lfs::core::NodeId node_id = lfs::core::NULL_NODE;
        lfs::core::NodeId parent_splat_id = lfs::core::NULL_NODE;
        int parent_node_index = -1;
        lfs::core::EllipsoidData data;
        glm::mat4 world_transform{1.0f};
    };

    struct VideoExportSceneSnapshot {
        std::shared_ptr<lfs::core::SplatData> combined_model;
        // Valid only during the viewer-thread render that captured it.
        const lfs::core::SplatData* borrowed_model = nullptr;
        std::shared_ptr<lfs::core::PointCloud> point_cloud;
        const lfs::core::PointCloud* borrowed_point_cloud = nullptr;
        glm::mat4 point_cloud_transform{1.0f};
        std::vector<VideoExportMeshSnapshot> meshes;
        std::vector<glm::mat4> model_transforms;
        std::vector<int> node_active_sh_degrees;
        std::shared_ptr<lfs::core::Tensor> transform_indices;
        std::shared_ptr<lfs::core::Tensor> selection_mask;
        std::vector<bool> selected_node_mask;
        std::vector<bool> node_visibility_mask;
        std::vector<VideoExportCropBoxSnapshot> cropboxes;
        int selected_cropbox_index = -1;
        std::optional<VideoExportEllipsoidSnapshot> active_ellipsoid;

        [[nodiscard]] bool hasRenderableContent() const {
            return (gaussianModel() && gaussianModel()->size() > 0) ||
                   (pointCloud() && pointCloud()->size() > 0) ||
                   !meshes.empty();
        }

        [[nodiscard]] const lfs::core::SplatData* gaussianModel() const {
            return borrowed_model ? borrowed_model : combined_model.get();
        }
        [[nodiscard]] const lfs::core::PointCloud* pointCloud() const {
            return borrowed_point_cloud ? borrowed_point_cloud : point_cloud.get();
        }
    };

    enum class VideoExportCapture { Owned,
                                    ImmediateRender };

    LFS_VIS_API std::expected<VideoExportSceneSnapshot, std::string> captureVideoExportSceneSnapshot(
        const lfs::vis::SceneManager& scene_manager,
        VideoExportCapture capture = VideoExportCapture::Owned);

    LFS_VIS_API void refreshVideoExportMeshTransforms(
        VideoExportSceneSnapshot& snapshot,
        const lfs::core::Scene& scene);

    LFS_VIS_API std::expected<lfs::io::video::VideoExportOptions, std::string> validateVideoExportOptions(
        lfs::io::video::VideoExportOptions options);

} // namespace lfs::vis::gui
