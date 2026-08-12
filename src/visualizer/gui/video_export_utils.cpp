/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/video_export_utils.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "io/loader.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <shared_mutex>

namespace lfs::vis::gui {

    namespace {

        std::shared_ptr<lfs::core::Tensor> cloneOptionalTensor(
            const std::shared_ptr<lfs::core::Tensor>& tensor);

        std::unique_ptr<lfs::core::SplatData> cloneSplatDataFloat32(const lfs::core::SplatData& src) {
            const bool reduced_sh = src.shN_raw().is_valid() &&
                                    (src.shN_raw().dtype() != lfs::core::DataType::Float32 ||
                                     src.shN_value_quantized() || src.shN_ieee_f16());
            auto cloned = std::make_unique<lfs::core::SplatData>(
                src.get_max_sh_degree(),
                src.means_raw().clone(),
                src.sh0_raw().clone(),
                src.shN_raw().is_valid()
                    ? (reduced_sh ? src.shN_canonical() : src.shN_raw().clone())
                    : lfs::core::Tensor{},
                src.scaling_raw().clone(),
                src.rotation_raw().clone(),
                src.opacity_raw().clone(),
                src.get_scene_scale(),
                reduced_sh ? lfs::core::SplatData::ShNLayout::Canonical
                           : lfs::core::SplatData::ShNLayout::Swizzled);
            cloned->set_active_sh_degree(src.get_active_sh_degree());
            if (src.has_deleted_mask()) {
                cloned->deleted() = src.deleted().clone();
            }
            if (src._densification_info.is_valid()) {
                cloned->_densification_info = src._densification_info.clone();
            }
            return cloned;
        }

        struct SourceFloat32Snapshot {
            std::shared_ptr<lfs::core::SplatData> combined_model;
            std::vector<glm::mat4> model_transforms;
            std::shared_ptr<lfs::core::Tensor> transform_indices;
        };

        std::expected<std::unique_ptr<lfs::core::SplatData>, std::string> loadSourceFloat32Splat(
            const std::filesystem::path& path) {
            auto loader = lfs::io::Loader::create();
            auto loaded = loader->load(path);
            if (!loaded) {
                return std::unexpected("Failed to reload source '" + path.string() + "': " +
                                       loaded.error().format());
            }

            auto* const splat = std::get_if<std::shared_ptr<lfs::core::SplatData>>(&loaded->data);
            if (!splat || !*splat) {
                return std::unexpected("Video source '" + path.string() + "' is not a Gaussian splat");
            }

            auto source = std::make_unique<lfs::core::SplatData>(std::move(**splat));
            if (source->shN_raw().is_valid() &&
                (source->shN_raw().dtype() != lfs::core::DataType::Float32 ||
                 source->shN_value_quantized() || source->shN_ieee_f16())) {
                auto decoded = std::make_unique<lfs::core::SplatData>(
                    source->get_max_sh_degree(),
                    source->means_raw(),
                    source->sh0_raw(),
                    source->shN_canonical(),
                    source->scaling_raw(),
                    source->rotation_raw(),
                    source->opacity_raw(),
                    source->get_scene_scale(),
                    lfs::core::SplatData::ShNLayout::Canonical);
                decoded->set_active_sh_degree(source->get_active_sh_degree());
                source = std::move(decoded);
            }
            return source;
        }

        std::expected<SourceFloat32Snapshot, std::string> captureSourceFloat32Splats(
            const lfs::vis::SceneManager& scene_manager) {
            const auto& source_scene = scene_manager.getScene();
            const auto visible_slots = source_scene.getVisibleSplatNodeSlots();
            if (visible_slots.empty()) {
                return std::unexpected("Source FP32 video export requires at least one visible file-backed splat");
            }
            lfs::core::Scene decoded_scene;
            for (const auto& slot : visible_slots) {
                const auto* const node = slot.node;
                if (!node) {
                    return std::unexpected("Source FP32 video export found an invalid scene node");
                }
                const auto source_path = scene_manager.getPlyPath(node->id);
                if (!source_path) {
                    return std::unexpected("Source FP32 video export is unavailable for generated node '" +
                                           node->name + "'; 32-bit export requires its source asset");
                }

                auto source = loadSourceFloat32Splat(*source_path);
                if (!source)
                    return std::unexpected(source.error());

                const size_t current_size = node->model
                                                ? static_cast<size_t>(node->model->size())
                                                : node->gaussian_count.load(std::memory_order_acquire);
                if ((*source)->size() != current_size) {
                    return std::unexpected(
                        "Source FP32 video export cannot reproduce modified node '" + node->name +
                        "' because its source Gaussian count differs from the current scene");
                }

                if (node->model) {
                    (*source)->set_active_sh_degree(node->model->get_active_sh_degree());
                    if (node->model->has_deleted_mask() &&
                        node->model->deleted().numel() == (*source)->size()) {
                        (*source)->deleted() = node->model->deleted().clone();
                    }
                } else if (const auto* combined = source_scene.getCombinedModel()) {
                    (*source)->set_active_sh_degree(combined->get_active_sh_degree());
                }

                const auto decoded_id = decoded_scene.addSplat(node->name, std::move(*source));
                if (decoded_id == lfs::core::NULL_NODE) {
                    return std::unexpected("Failed to reconstruct source node '" + node->name + "'");
                }
                decoded_scene.setNodeTransform(node->name, source_scene.getWorldTransform(node->id));
            }

            const auto* const combined = decoded_scene.getCombinedModel();
            if (!combined || combined->size() == 0) {
                return std::unexpected("Source FP32 video export produced an empty splat snapshot");
            }
            SourceFloat32Snapshot result;
            result.combined_model =
                std::shared_ptr<lfs::core::SplatData>(cloneSplatDataFloat32(*combined).release());
            result.model_transforms = decoded_scene.getVisibleNodeTransforms();
            for (auto& transform : result.model_transforms) {
                transform = rendering::dataWorldTransformToVisualizerWorld(transform);
            }
            result.transform_indices = cloneOptionalTensor(decoded_scene.getTransformIndices());
            return result;
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
        const lfs::vis::SceneManager& scene_manager,
        const lfs::io::video::VideoSplatPrecision precision) {
        VideoExportSceneSnapshot snapshot;

        auto render_lock = acquireLiveModelRenderLock(scene_manager);
        const auto render_state = scene_manager.buildRenderState();
        const auto& scene = scene_manager.getScene();

        if (const auto* const model = scene_manager.getModelForRendering(); model && model->size() > 0) {
            const bool resident_is_reduced =
                model->shN_raw().is_valid() &&
                (model->shN_raw().dtype() != lfs::core::DataType::Float32 ||
                 model->shN_value_quantized() || model->shN_ieee_f16());
            if (precision == lfs::io::video::VideoSplatPrecision::Float32 &&
                resident_is_reduced) {
                auto source_snapshot = captureSourceFloat32Splats(scene_manager);
                if (!source_snapshot)
                    return std::unexpected(source_snapshot.error());
                snapshot.combined_model = std::move(source_snapshot->combined_model);
                snapshot.model_transforms = std::move(source_snapshot->model_transforms);
                snapshot.transform_indices = std::move(source_snapshot->transform_indices);
            } else {
                snapshot.combined_model = std::shared_ptr<lfs::core::SplatData>(
                    cloneSplatDataFloat32(*model).release());
                snapshot.model_transforms = render_state.model_transforms;
                snapshot.transform_indices = cloneOptionalTensor(render_state.transform_indices);
            }
            if (auto allocator = lfs::vis::makeViewerSplatTensorAllocator()) {
                if (auto migrated = lfs::io::migrateSplatTensorsToAllocator(*snapshot.combined_model, allocator);
                    !migrated) {
                    return std::unexpected(std::string(LOC(lichtfeld::Strings::Runtime::VIDEO_SPLAT_PREPARATION_FAILED)) +
                                           migrated.error().format());
                }
            }
            if (precision == lfs::io::video::VideoSplatPrecision::Float16 &&
                snapshot.combined_model->shN_raw().is_valid() &&
                snapshot.combined_model->shN_raw().numel() > 0) {
                try {
                    if (!lfs::training::sh_value::apply_shN_value_quant(
                            *snapshot.combined_model) &&
                        !snapshot.combined_model->shN_value_quantized()) {
                        return std::unexpected(
                            "Failed to prepare the requested 16-bit SH representation for video export");
                    }
                } catch (const std::exception& e) {
                    return std::unexpected(
                        std::string("Failed to prepare 16-bit SH data for video export: ") + e.what());
                }
            }
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
                .node_id = vm.node_id,
                .transform = vm.transform,
                .is_selected = vm.is_selected,
            });
        }

        snapshot.cropboxes.reserve(render_state.cropboxes.size());
        for (const auto& cb : render_state.cropboxes) {
            VideoExportCropBoxSnapshot cropbox_snapshot;
            cropbox_snapshot.has_data = cb.data != nullptr;
            cropbox_snapshot.node_id = cb.node_id;
            cropbox_snapshot.parent_splat_id = cb.parent_splat_id;
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
                .parent_splat_id = el.parent_splat_id,
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
                    .parent_splat_id = el.parent_splat_id,
                    .parent_node_index = scene.getVisibleNodeIndex(el.parent_splat_id),
                    .data = *el.data,
                    .world_transform = el.world_transform,
                };
                break;
            }
        }

        if (!snapshot.hasRenderableContent()) {
            return std::unexpected(std::string(LOC(lichtfeld::Strings::Runtime::VIDEO_NO_RENDERABLE_CONTENT)));
        }

        return snapshot;
    }

    void refreshVideoExportMeshTransforms(
        VideoExportSceneSnapshot& snapshot,
        const lfs::core::Scene& scene) {
        for (auto& mesh : snapshot.meshes) {
            if (mesh.node_id == lfs::core::NULL_NODE || !scene.getNodeById(mesh.node_id))
                continue;
            mesh.transform = rendering::dataWorldTransformToVisualizerWorld(
                scene.getWorldTransform(mesh.node_id));
        }
    }

    std::expected<lfs::io::video::VideoExportOptions, std::string> validateVideoExportOptions(
        lfs::io::video::VideoExportOptions options) {
        if (const auto validation = lfs::io::video::validateVideoEncodingOptions(options); !validation)
            return std::unexpected(validation.error());
        return options;
    }

    std::expected<VideoExportRenderPlan, std::string> makeVideoExportRenderPlan(
        const lfs::io::video::VideoExportOptions& options) {
        const auto validated = validateVideoExportOptions(options);
        if (!validated)
            return std::unexpected(validated.error());

        const bool native = options.upscaler.backend == "native";
        const float scale = native ? 1.0f : options.upscaler.input_scale;
        const auto scaled_dimension = [scale](const int value) {
            const int scaled = std::max(1, static_cast<int>(std::lround(value * scale)));
            return scaled + (scaled & 1);
        };

        return VideoExportRenderPlan{
            .output_width = options.width,
            .output_height = options.height,
            .input_width = scaled_dimension(options.width),
            .input_height = scaled_dimension(options.height),
            .requires_upscale = !native && scale < 1.0f,
            .backend = options.upscaler.backend,
        };
    }

    std::expected<VideoExportRenderPlan, std::string> resolveVideoExportRenderPlan(
        const lfs::io::video::VideoExportOptions& options,
        const bool supports_reconstruction) {
        auto plan = makeVideoExportRenderPlan(options);
        if (!plan || !plan->requires_upscale || supports_reconstruction)
            return plan;

        if (options.upscaler.fallback == lfs::io::video::VideoUpscalerFallback::Abort) {
            return std::unexpected(
                "The selected video upscaler cannot currently preserve mesh or environment compositing");
        }

        auto native_options = options;
        native_options.upscaler.backend = "native";
        native_options.upscaler.input_scale = 1.0f;
        return makeVideoExportRenderPlan(native_options);
    }

} // namespace lfs::vis::gui
