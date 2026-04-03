/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/video_export_utils.hpp"
#include "rendering/render_constants.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include <format>
#include <optional>
#include <shared_mutex>

namespace lfs::vis::gui {

    namespace {

        std::unique_ptr<lfs::core::SplatData> cloneSplatData(const lfs::core::SplatData& src) {
            auto cloned = std::make_unique<lfs::core::SplatData>(
                src.get_max_sh_degree(),
                src.means_raw().clone(),
                src.sh0_raw().clone(),
                src.shN_raw().is_valid() ? src.shN_raw().clone() : lfs::core::Tensor{},
                src.scaling_raw().clone(),
                src.rotation_raw().clone(),
                src.opacity_raw().clone(),
                src.get_scene_scale());
            cloned->set_active_sh_degree(src.get_active_sh_degree());
            cloned->set_max_sh_degree(src.get_max_sh_degree());
            if (src.has_deleted_mask()) {
                cloned->deleted() = src.deleted().clone();
            }
            if (src._densification_info.is_valid()) {
                cloned->_densification_info = src._densification_info.clone();
            }
            return cloned;
        }

        std::shared_ptr<lfs::core::PointCloud> clonePointCloud(const lfs::core::PointCloud& src) {
            auto cloned = std::make_shared<lfs::core::PointCloud>();
            cloned->means = src.means.is_valid() ? src.means.clone() : src.means;
            cloned->colors = src.colors.is_valid() ? src.colors.clone() : src.colors;
            cloned->normals = src.normals.is_valid() ? src.normals.clone() : src.normals;
            cloned->sh0 = src.sh0.is_valid() ? src.sh0.clone() : src.sh0;
            cloned->shN = src.shN.is_valid() ? src.shN.clone() : src.shN;
            cloned->opacity = src.opacity.is_valid() ? src.opacity.clone() : src.opacity;
            cloned->scaling = src.scaling.is_valid() ? src.scaling.clone() : src.scaling;
            cloned->rotation = src.rotation.is_valid() ? src.rotation.clone() : src.rotation;
            cloned->attribute_names = src.attribute_names;
            return cloned;
        }

        std::shared_ptr<lfs::core::MeshData> cloneMeshData(const lfs::core::MeshData& src) {
            auto cloned = std::make_shared<lfs::core::MeshData>();
            cloned->vertices = src.vertices.is_valid() ? src.vertices.clone() : src.vertices;
            cloned->normals = src.normals.is_valid() ? src.normals.clone() : src.normals;
            cloned->tangents = src.tangents.is_valid() ? src.tangents.clone() : src.tangents;
            cloned->texcoords = src.texcoords.is_valid() ? src.texcoords.clone() : src.texcoords;
            cloned->colors = src.colors.is_valid() ? src.colors.clone() : src.colors;
            cloned->indices = src.indices.is_valid() ? src.indices.clone() : src.indices;
            cloned->materials = src.materials;
            cloned->submeshes = src.submeshes;
            cloned->texture_images = src.texture_images;
            cloned->generation_.store(src.generation(), std::memory_order_relaxed);
            return cloned;
        }

        std::shared_ptr<lfs::core::Tensor> cloneOptionalTensor(const std::shared_ptr<lfs::core::Tensor>& tensor) {
            if (!tensor || !tensor->is_valid()) {
                return nullptr;
            }
            return std::make_shared<lfs::core::Tensor>(tensor->clone());
        }

        [[nodiscard]] std::optional<std::shared_lock<std::shared_mutex>> acquireLiveModelRenderLock(
            const lfs::vis::SceneManager& scene_manager) {
            std::optional<std::shared_lock<std::shared_mutex>> lock;
            if (const auto* tm = scene_manager.getTrainerManager()) {
                if (const auto* trainer = tm->getTrainer()) {
                    lock.emplace(trainer->getRenderMutex());
                }
            }
            return lock;
        }

    } // namespace

    std::expected<VideoExportSceneSnapshot, std::string> captureVideoExportSceneSnapshot(
        const lfs::vis::SceneManager& scene_manager) {
        VideoExportSceneSnapshot snapshot;

        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        const auto render_state = scene_manager.buildRenderState();
        const auto& scene = scene_manager.getScene();

        if (const auto* const model = scene_manager.getModelForRendering();
            model && model->size() > 0) {
            snapshot.combined_model = std::shared_ptr<lfs::core::SplatData>(cloneSplatData(*model).release());
            snapshot.model_transforms = render_state.model_transforms;
            snapshot.transform_indices = cloneOptionalTensor(render_state.transform_indices);
            snapshot.selection_mask = cloneOptionalTensor(render_state.selection_mask);
            snapshot.selected_node_mask = render_state.selected_node_mask;
            snapshot.node_visibility_mask = render_state.node_visibility_mask;
        } else if (render_state.point_cloud && render_state.point_cloud->size() > 0) {
            snapshot.point_cloud = clonePointCloud(*render_state.point_cloud);
            snapshot.point_cloud_transform = render_state.point_cloud_transform;
        }

        snapshot.meshes.reserve(render_state.meshes.size());
        for (const auto& vm : render_state.meshes) {
            if (!vm.mesh)
                continue;
            snapshot.meshes.push_back(VideoExportMeshSnapshot{
                .mesh = cloneMeshData(*vm.mesh),
                .transform = vm.transform,
                .is_selected = vm.is_selected,
            });
        }

        snapshot.cropboxes.reserve(render_state.cropboxes.size());
        for (const auto& cb : render_state.cropboxes) {
            VideoExportCropBoxSnapshot cropbox_snapshot;
            cropbox_snapshot.has_data = cb.data != nullptr;
            cropbox_snapshot.node_id = cb.node_id;
            cropbox_snapshot.parent_node_index = scene.getVisibleNodeIndex(cb.parent_splat_id);
            cropbox_snapshot.world_transform = cb.world_transform;
            if (cb.data) {
                cropbox_snapshot.data = *cb.data;
            }
            snapshot.cropboxes.push_back(std::move(cropbox_snapshot));
        }
        snapshot.selected_cropbox_index = render_state.selected_cropbox_index;

        const lfs::core::NodeId active_ellipsoid_id = scene_manager.getActiveSelectionEllipsoidId();
        for (const auto& el : render_state.ellipsoids) {
            if (!el.data)
                continue;
            if (active_ellipsoid_id != lfs::core::NULL_NODE && el.node_id != active_ellipsoid_id)
                continue;
            snapshot.active_ellipsoid = VideoExportEllipsoidSnapshot{
                .node_id = el.node_id,
                .parent_node_index = scene.getVisibleNodeIndex(el.parent_splat_id),
                .data = *el.data,
                .world_transform = el.world_transform,
            };
            break;
        }

        if (!snapshot.active_ellipsoid && active_ellipsoid_id == lfs::core::NULL_NODE) {
            for (const auto& el : render_state.ellipsoids) {
                if (!el.data)
                    continue;
                snapshot.active_ellipsoid = VideoExportEllipsoidSnapshot{
                    .node_id = el.node_id,
                    .parent_node_index = scene.getVisibleNodeIndex(el.parent_splat_id),
                    .data = *el.data,
                    .world_transform = el.world_transform,
                };
                break;
            }
        }

        if (!snapshot.hasRenderableContent()) {
            return std::unexpected("No renderable content to export");
        }

        return snapshot;
    }

    std::expected<lfs::io::video::VideoExportOptions, std::string> validateVideoExportOptions(
        lfs::io::video::VideoExportOptions options) {
        if (options.width <= 0 || options.height <= 0) {
            return std::unexpected("Video export width and height must be positive");
        }
        if (options.width > lfs::rendering::MAX_VIEWPORT_SIZE ||
            options.height > lfs::rendering::MAX_VIEWPORT_SIZE) {
            return std::unexpected(
                std::format("Video export resolution exceeds maximum viewport size of {}",
                            lfs::rendering::MAX_VIEWPORT_SIZE));
        }
        if (options.framerate <= 0) {
            return std::unexpected("Video export framerate must be positive");
        }
        if (options.crf < 0 || options.crf > 51) {
            return std::unexpected("Video export CRF must be between 0 and 51");
        }
        return options;
    }

} // namespace lfs::vis::gui
