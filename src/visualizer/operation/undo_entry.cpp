/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "undo_entry.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "undo_history.hpp"
#include <algorithm>
#include <array>
#include <cuda_runtime.h>
#include <limits>
#include <set>
#include <stdexcept>

namespace lfs::vis::op {

    namespace {
        using lfs::core::events::state::NodeReparented;
        using lfs::core::events::state::PLYAdded;
        using lfs::core::events::state::PLYRemoved;
        using lfs::core::events::state::SceneCleared;
        constexpr auto PROPERTY_COALESCE_WINDOW = std::chrono::milliseconds(500);

        bool cropBoxesEqual(const lfs::core::CropBoxData& lhs, const lfs::core::CropBoxData& rhs) {
            return lhs.min == rhs.min &&
                   lhs.max == rhs.max &&
                   lhs.inverse == rhs.inverse &&
                   lhs.enabled == rhs.enabled &&
                   lhs.color == rhs.color &&
                   lhs.line_width == rhs.line_width &&
                   lhs.flash_intensity == rhs.flash_intensity;
        }

        bool ellipsoidsEqual(const lfs::core::EllipsoidData& lhs, const lfs::core::EllipsoidData& rhs) {
            return lhs.radii == rhs.radii &&
                   lhs.inverse == rhs.inverse &&
                   lhs.enabled == rhs.enabled &&
                   lhs.color == rhs.color &&
                   lhs.line_width == rhs.line_width &&
                   lhs.flash_intensity == rhs.flash_intensity;
        }

        std::string propertyUndoLabel(const std::string& property_path) {
            if (property_path.ends_with(".transform")) {
                return "Set Transform";
            }
            if (property_path.ends_with(".visible")) {
                return "Set Visibility";
            }
            if (property_path.ends_with(".locked")) {
                return "Set Lock State";
            }
            return "Set Property";
        }

        std::string propertyUndoScope(const std::string& property_path) {
            if (property_path.contains("scene_node")) {
                return "scene_graph";
            }
            if (property_path.contains("crop_box")) {
                return "cropbox";
            }
            if (property_path.contains("ellipsoid")) {
                return "ellipsoid";
            }
            return "property";
        }

        size_t estimateAnyBytes(const std::any& value) {
            if (!value.has_value()) {
                return 0;
            }
            if (value.type() == typeid(bool)) {
                return sizeof(bool);
            }
            if (value.type() == typeid(int)) {
                return sizeof(int);
            }
            if (value.type() == typeid(size_t)) {
                return sizeof(size_t);
            }
            if (value.type() == typeid(float)) {
                return sizeof(float);
            }
            if (value.type() == typeid(double)) {
                return sizeof(double);
            }
            if (value.type() == typeid(glm::vec2)) {
                return sizeof(glm::vec2);
            }
            if (value.type() == typeid(glm::vec3)) {
                return sizeof(glm::vec3);
            }
            if (value.type() == typeid(glm::vec4)) {
                return sizeof(glm::vec4);
            }
            if (value.type() == typeid(glm::mat4)) {
                return sizeof(glm::mat4);
            }
            if (value.type() == typeid(std::string)) {
                return std::any_cast<const std::string&>(value).size();
            }
            if (value.type() == typeid(std::filesystem::path)) {
                return std::any_cast<const std::filesystem::path&>(value).native().size();
            }
            return sizeof(std::any);
        }

        DirtyMask propertyDirtyFlags(const std::string& property_path) {
            if (property_path.ends_with(".transform")) {
                return DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY;
            }
            if (property_path.ends_with(".visible")) {
                return DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY;
            }
            if (property_path.ends_with(".locked")) {
                return DirtyFlag::OVERLAY;
            }
            if (property_path.contains("crop_box") || property_path.contains("ellipsoid")) {
                return DirtyFlag::SPLATS | DirtyFlag::OVERLAY;
            }
            if (property_path.contains("scene_node")) {
                return DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY;
            }
            return DirtyFlag::ALL;
        }

        std::string snapshotScope(const ModifiesFlag captured) {
            const bool selection = hasFlag(captured, ModifiesFlag::SELECTION);
            const bool transforms = hasFlag(captured, ModifiesFlag::TRANSFORMS);
            const bool topology = hasFlag(captured, ModifiesFlag::TOPOLOGY);
            if (selection && !transforms && !topology) {
                return "selection";
            }
            if (transforms && !selection && !topology) {
                return "transform";
            }
            if (topology && !selection && !transforms) {
                return "topology";
            }
            if (selection || transforms || topology) {
                return "scene";
            }
            return "general";
        }

        DirtyMask snapshotDirtyFlags(const ModifiesFlag captured) {
            DirtyMask flags = 0;
            if (hasFlag(captured, ModifiesFlag::SELECTION)) {
                flags |= DirtyFlag::SELECTION;
            }
            if (hasFlag(captured, ModifiesFlag::TRANSFORMS)) {
                flags |= DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY;
            }
            if (hasFlag(captured, ModifiesFlag::TOPOLOGY)) {
                flags |= DirtyFlag::SPLATS | DirtyFlag::SELECTION;
            }
            return flags == 0 ? DirtyFlag::ALL : flags;
        }

        DirtyMask sceneGraphMetadataDirtyFlags() {
            return DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY | DirtyFlag::SELECTION;
        }

        SceneGraphNodeMetadataSnapshot captureNodeMetadataSnapshot(const SceneManager& scene_manager,
                                                                   const lfs::core::SceneNode& node);

        void restorePlyPathForSnapshot(SceneManager& scene_manager,
                                       const std::string& node_name,
                                       const std::optional<std::filesystem::path>& source_path) {
            if (source_path) {
                scene_manager.setPlyPath(node_name, *source_path);
            } else {
                scene_manager.clearPlyPath(node_name);
            }
        }

        [[nodiscard]] std::string resolveExistingNodeName(const lfs::core::Scene& scene,
                                                          const std::vector<std::string>& candidates) {
            std::set<std::string> seen;
            for (const auto& candidate : candidates) {
                if (candidate.empty() || !seen.insert(candidate).second) {
                    continue;
                }
                if (scene.getNode(candidate)) {
                    return candidate;
                }
            }
            return {};
        }

        void applyNodeMetadataSnapshotUnchecked(SceneManager& scene_manager,
                                                const SceneGraphNodeMetadataSnapshot& target,
                                                std::string current_name,
                                                const bool emit_reparent_event) {
            auto& scene = scene_manager.getScene();

            auto* node = scene.getMutableNode(current_name);
            if (!node) {
                throw std::runtime_error("Cannot restore scene node metadata for '" + target.name + "'");
            }

            if (current_name != target.name) {
                if (!scene.renameNode(current_name, target.name)) {
                    throw std::runtime_error("Failed to rename node '" + current_name + "' to '" + target.name + "'");
                }
                scene_manager.movePlyPath(current_name, target.name);
                current_name = target.name;
                node = scene.getMutableNode(current_name);
                if (!node) {
                    throw std::runtime_error("Renamed node not found: '" + current_name + "'");
                }
            }

            lfs::core::NodeId desired_parent = lfs::core::NULL_NODE;
            if (!target.parent_name.empty()) {
                desired_parent = scene.getNodeIdByName(target.parent_name);
                if (desired_parent == lfs::core::NULL_NODE) {
                    throw std::runtime_error("Missing parent '" + target.parent_name + "' for node '" + target.name + "'");
                }
            }

            if (node->parent_id != desired_parent) {
                std::string old_parent_name;
                if (node->parent_id != lfs::core::NULL_NODE) {
                    if (const auto* old_parent = scene.getNodeById(node->parent_id)) {
                        old_parent_name = old_parent->name;
                    }
                }

                scene.reparent(node->id, desired_parent);
                node = scene.getMutableNode(current_name);
                if (!node || node->parent_id != desired_parent) {
                    throw std::runtime_error("Failed to reparent node '" + current_name + "'");
                }

                scene_manager.invalidateNodeSelectionMask();
                if (emit_reparent_event) {
                    NodeReparented{
                        .name = target.name,
                        .old_parent = old_parent_name,
                        .new_parent = target.parent_name,
                        .from_history = true}
                        .emit();
                }
            }

            scene.setNodeTransform(current_name, target.local_transform);
            node->visible.set(target.visible, false);
            node->locked.set(target.locked, false);
            node->training_enabled = target.training_enabled;
            restorePlyPathForSnapshot(scene_manager, current_name, target.source_path);
        }

        void applyNodeMetadataSnapshot(SceneManager& scene_manager,
                                       const SceneGraphNodeMetadataSnapshot& target,
                                       const std::vector<std::string>& candidates,
                                       const bool emit_reparent_event) {
            auto& scene = scene_manager.getScene();
            const std::string current_name = resolveExistingNodeName(scene, candidates);
            if (current_name.empty()) {
                throw std::runtime_error("Cannot restore scene node metadata for '" + target.name + "'");
            }

            const auto* current_node = scene.getNode(current_name);
            if (!current_node) {
                throw std::runtime_error("Cannot restore scene node metadata for '" + target.name + "'");
            }
            const auto before = captureNodeMetadataSnapshot(scene_manager, *current_node);

            try {
                applyNodeMetadataSnapshotUnchecked(scene_manager, target, current_name, emit_reparent_event);
            } catch (const HistoryCorruptionError&) {
                throw;
            } catch (...) {
                try {
                    const std::vector<std::string> rollback_candidates{
                        before.name,
                        target.name,
                        current_name,
                    };
                    const auto rollback_name = resolveExistingNodeName(
                        scene,
                        rollback_candidates);
                    if (rollback_name.empty()) {
                        throw std::runtime_error("Rollback target node missing");
                    }
                    applyNodeMetadataSnapshotUnchecked(scene_manager, before, rollback_name, false);
                } catch (const std::exception& rollback_error) {
                    throw HistoryCorruptionError(
                        "Failed to rollback scene node metadata for '" + target.name + "': " +
                        std::string(rollback_error.what()));
                } catch (...) {
                    throw HistoryCorruptionError(
                        "Failed to rollback scene node metadata for '" + target.name + "': unknown exception");
                }
                throw;
            }
        }

        size_t tensorBytes(const lfs::core::Tensor& tensor) {
            return tensor.is_valid() ? tensor.bytes() : 0;
        }

        UndoMemoryBreakdown tensorMemory(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid()) {
                return {};
            }
            if (tensor.device() == lfs::core::Device::CUDA) {
                return UndoMemoryBreakdown{
                    .cpu_bytes = 0,
                    .gpu_bytes = tensor.bytes(),
                };
            }
            return UndoMemoryBreakdown{
                .cpu_bytes = tensor.bytes(),
                .gpu_bytes = 0,
            };
        }

        void offloadTensor(lfs::core::Tensor& tensor) {
            if (tensor.is_valid() && tensor.device() != lfs::core::Device::CPU) {
                tensor = tensor.to(lfs::core::Device::CPU).contiguous();
            }
        }

        void restoreTensorToDevice(lfs::core::Tensor& tensor, const lfs::core::Device device) {
            if (tensor.is_valid() && tensor.device() != device) {
                tensor = tensor.to(device).contiguous();
            }
        }

        bool selectionGroupsEqual(const std::vector<lfs::core::SelectionGroup>& lhs,
                                  const std::vector<lfs::core::SelectionGroup>& rhs) {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i) {
                const auto& a = lhs[i];
                const auto& b = rhs[i];
                if (a.id != b.id || a.name != b.name || a.color != b.color || a.count != b.count ||
                    a.locked != b.locked) {
                    return false;
                }
            }
            return true;
        }

        bool selectionMetadataEqual(const lfs::core::Scene::SelectionStateSnapshot& lhs,
                                    const lfs::core::Scene::SelectionStateMetadata& rhs) {
            return lhs.has_selection == rhs.has_selection &&
                   lhs.active_group_id == rhs.active_group_id &&
                   lhs.next_group_id == rhs.next_group_id &&
                   selectionGroupsEqual(lhs.groups, rhs.groups);
        }

        size_t tensorNumel(const std::shared_ptr<lfs::core::Tensor>& tensor) {
            return (tensor && tensor->is_valid()) ? tensor->numel() : 0;
        }

        size_t tensorNumel(const lfs::core::Tensor* tensor) {
            return (tensor && tensor->is_valid()) ? tensor->numel() : 0;
        }

        lfs::core::Tensor materializeMaskTensor(const std::shared_ptr<lfs::core::Tensor>& tensor,
                                                const size_t total_size,
                                                const lfs::core::Device device,
                                                const lfs::core::DataType dtype) {
            if (tensor && tensor->is_valid()) {
                auto materialized = tensor->device() == device ? *tensor : tensor->to(device);
                return materialized.dtype() == dtype ? materialized : materialized.to(dtype);
            }
            return total_size > 0 ? lfs::core::Tensor::zeros({total_size}, device, dtype) : lfs::core::Tensor{};
        }

        lfs::core::Tensor materializeMaskTensor(const lfs::core::Tensor* tensor,
                                                const size_t total_size,
                                                const lfs::core::Device device,
                                                const lfs::core::DataType dtype) {
            if (tensor && tensor->is_valid()) {
                auto materialized = tensor->device() == device ? *tensor : tensor->to(device);
                return materialized.dtype() == dtype ? materialized : materialized.to(dtype);
            }
            return total_size > 0 ? lfs::core::Tensor::zeros({total_size}, device, dtype) : lfs::core::Tensor{};
        }

        TensorSwapStorage buildTensorSwapStorage(const std::shared_ptr<lfs::core::Tensor>& before,
                                                 const lfs::core::Tensor* after,
                                                 const size_t total_size,
                                                 const lfs::core::Device fallback_device,
                                                 const lfs::core::DataType dtype,
                                                 const bool before_present,
                                                 const bool after_present) {
            TensorSwapStorage storage;
            storage.total_size = total_size;
            storage.device = fallback_device;
            storage.dtype = dtype;
            storage.before_present = before_present;
            storage.after_present = after_present;

            if (total_size == 0) {
                return storage;
            }

            const lfs::core::Device device =
                (after && after->is_valid()) ? after->device()
                                             : ((before && before->is_valid()) ? before->device() : fallback_device);
            storage.device = device;

            auto before_tensor = materializeMaskTensor(before, total_size, device, dtype);
            auto after_tensor = materializeMaskTensor(after, total_size, device, dtype);
            auto changed_mask =
                ((dtype == lfs::core::DataType::UInt8) || (dtype == lfs::core::DataType::Bool))
                    ? before_tensor.to(lfs::core::DataType::Int32).ne(after_tensor.to(lfs::core::DataType::Int32))
                    : before_tensor.ne(after_tensor);
            auto changed_indices = changed_mask.nonzero();
            if (changed_indices.is_valid() && changed_indices.ndim() == 2 && changed_indices.size(1) == 1) {
                changed_indices = changed_indices.squeeze(1);
            }

            const size_t changed_count = changed_indices.is_valid() ? changed_indices.numel() : 0;
            if (changed_count == 0) {
                if (before_present != after_present) {
                    storage.mode = TensorSwapStorageMode::DENSE;
                    storage.stored_values = std::move(before_tensor);
                }
                return storage;
            }

            auto indices_int32 = changed_indices.dtype() == lfs::core::DataType::Int32
                                     ? changed_indices.contiguous()
                                     : changed_indices.to(lfs::core::DataType::Int32).contiguous();
            const size_t sparse_bytes =
                indices_int32.bytes() + (changed_count * lfs::core::dtype_size(dtype));
            const size_t dense_bytes = before_tensor.bytes();

            if (sparse_bytes < dense_bytes) {
                storage.mode = TensorSwapStorageMode::SPARSE;
                storage.indices = std::move(indices_int32);
                storage.stored_values = before_tensor.index_select(0, storage.indices).contiguous();
            } else {
                storage.mode = TensorSwapStorageMode::DENSE;
                storage.stored_values = std::move(before_tensor);
            }

            return storage;
        }

        void applyTensorSwapStorage(lfs::core::Tensor& live_tensor, TensorSwapStorage& storage) {
            switch (storage.mode) {
            case TensorSwapStorageMode::DENSE: {
                if (!live_tensor.is_valid()) {
                    live_tensor = lfs::core::Tensor::zeros({storage.total_size}, storage.device, storage.dtype);
                }
                if (!storage.stored_values.is_valid()) {
                    storage.stored_values =
                        lfs::core::Tensor::zeros({storage.total_size}, storage.device, storage.dtype);
                }
                std::swap(live_tensor, storage.stored_values);
                return;
            }
            case TensorSwapStorageMode::SPARSE: {
                if (!storage.indices.is_valid() || storage.indices.numel() == 0) {
                    return;
                }
                if (!live_tensor.is_valid()) {
                    throw std::runtime_error("Cannot replay sparse tensor history against a missing tensor");
                }
                if (live_tensor.numel() < storage.total_size) {
                    throw std::runtime_error("Cannot replay sparse tensor history after tensor resized");
                }
                if (!storage.stored_values.is_valid() || storage.stored_values.numel() != storage.indices.numel()) {
                    throw std::runtime_error("Cannot replay sparse tensor history with invalid stored values");
                }

                const auto max_index = storage.indices.max().item<int32_t>();
                if (max_index < 0 || static_cast<size_t>(max_index) >= live_tensor.numel()) {
                    throw std::runtime_error("Cannot replay sparse tensor history: indices out of bounds");
                }

                auto current_values = live_tensor.index_select(0, storage.indices).contiguous();
                live_tensor.scatter_(0, storage.indices, storage.stored_values);
                if (live_tensor.is_valid() && live_tensor.device() == lfs::core::Device::CUDA) {
                    if (const cudaError_t err = cudaStreamSynchronize(live_tensor.stream()); err != cudaSuccess) {
                        LOG_ERROR("Failed to synchronize sparse tensor swap replay: {}",
                                  cudaGetErrorString(err));
                    }
                }
                storage.stored_values = std::move(current_values);
                return;
            }
            case TensorSwapStorageMode::NONE:
            default:
                return;
            }
        }

        UndoMemoryBreakdown estimateSplatDataMemory(const lfs::core::SplatData& splat) {
            UndoMemoryBreakdown total;
            total += tensorMemory(splat.means_raw());
            total += tensorMemory(splat.sh0_raw());
            total += tensorMemory(splat.shN_raw());
            total += tensorMemory(splat.scaling_raw());
            total += tensorMemory(splat.rotation_raw());
            total += tensorMemory(splat.opacity_raw());
            total += tensorMemory(splat.deleted());
            total += tensorMemory(splat._densification_info);
            return total;
        }

        void offloadSplatData(lfs::core::SplatData& splat) {
            offloadTensor(splat.means_raw());
            offloadTensor(splat.sh0_raw());
            offloadTensor(splat.shN_raw());
            offloadTensor(splat.scaling_raw());
            offloadTensor(splat.rotation_raw());
            offloadTensor(splat.opacity_raw());
            offloadTensor(splat.deleted());
            offloadTensor(splat._densification_info);
        }

        void restoreSplatDataToDevice(lfs::core::SplatData& splat, const lfs::core::Device device) {
            restoreTensorToDevice(splat.means_raw(), device);
            restoreTensorToDevice(splat.sh0_raw(), device);
            restoreTensorToDevice(splat.shN_raw(), device);
            restoreTensorToDevice(splat.scaling_raw(), device);
            restoreTensorToDevice(splat.rotation_raw(), device);
            restoreTensorToDevice(splat.opacity_raw(), device);
            restoreTensorToDevice(splat.deleted(), device);
            restoreTensorToDevice(splat._densification_info, device);
        }

        UndoMemoryBreakdown estimatePointCloudMemory(const lfs::core::PointCloud& point_cloud) {
            UndoMemoryBreakdown total;
            total += tensorMemory(point_cloud.means);
            total += tensorMemory(point_cloud.colors);
            total += tensorMemory(point_cloud.normals);
            total += tensorMemory(point_cloud.sh0);
            total += tensorMemory(point_cloud.shN);
            total += tensorMemory(point_cloud.opacity);
            total += tensorMemory(point_cloud.scaling);
            total += tensorMemory(point_cloud.rotation);
            return total;
        }

        void offloadPointCloud(lfs::core::PointCloud& point_cloud) {
            offloadTensor(point_cloud.means);
            offloadTensor(point_cloud.colors);
            offloadTensor(point_cloud.normals);
            offloadTensor(point_cloud.sh0);
            offloadTensor(point_cloud.shN);
            offloadTensor(point_cloud.opacity);
            offloadTensor(point_cloud.scaling);
            offloadTensor(point_cloud.rotation);
        }

        void restorePointCloudToDevice(lfs::core::PointCloud& point_cloud, const lfs::core::Device device) {
            restoreTensorToDevice(point_cloud.means, device);
            restoreTensorToDevice(point_cloud.colors, device);
            restoreTensorToDevice(point_cloud.normals, device);
            restoreTensorToDevice(point_cloud.sh0, device);
            restoreTensorToDevice(point_cloud.shN, device);
            restoreTensorToDevice(point_cloud.opacity, device);
            restoreTensorToDevice(point_cloud.scaling, device);
            restoreTensorToDevice(point_cloud.rotation, device);
        }

        UndoMemoryBreakdown estimateMeshMemory(const lfs::core::MeshData& mesh) {
            UndoMemoryBreakdown total;
            total += tensorMemory(mesh.vertices);
            total += tensorMemory(mesh.normals);
            total += tensorMemory(mesh.tangents);
            total += tensorMemory(mesh.texcoords);
            total += tensorMemory(mesh.colors);
            total += tensorMemory(mesh.indices);
            return total;
        }

        void offloadMesh(lfs::core::MeshData& mesh) {
            offloadTensor(mesh.vertices);
            offloadTensor(mesh.normals);
            offloadTensor(mesh.tangents);
            offloadTensor(mesh.texcoords);
            offloadTensor(mesh.colors);
            offloadTensor(mesh.indices);
        }

        void restoreMeshToDevice(lfs::core::MeshData& mesh, const lfs::core::Device device) {
            restoreTensorToDevice(mesh.vertices, device);
            restoreTensorToDevice(mesh.normals, device);
            restoreTensorToDevice(mesh.tangents, device);
            restoreTensorToDevice(mesh.texcoords, device);
            restoreTensorToDevice(mesh.colors, device);
            restoreTensorToDevice(mesh.indices, device);
        }

        UndoMemoryBreakdown estimateCameraMemory(const SceneGraphCameraSnapshot& camera) {
            UndoMemoryBreakdown total;
            total += tensorMemory(camera.R);
            total += tensorMemory(camera.T);
            total += tensorMemory(camera.radial_distortion);
            total += tensorMemory(camera.tangential_distortion);
            return total;
        }

        void offloadCamera(SceneGraphCameraSnapshot& camera) {
            offloadTensor(camera.R);
            offloadTensor(camera.T);
            offloadTensor(camera.radial_distortion);
            offloadTensor(camera.tangential_distortion);
        }

        void restoreCameraToDevice(SceneGraphCameraSnapshot& camera) {
            restoreTensorToDevice(camera.R, camera.device);
            restoreTensorToDevice(camera.T, camera.device);
            restoreTensorToDevice(camera.radial_distortion, camera.device);
            restoreTensorToDevice(camera.tangential_distortion, camera.device);
        }

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

        SceneGraphCameraSnapshot captureCameraSnapshot(const lfs::core::Camera& camera) {
            SceneGraphCameraSnapshot snapshot;
            snapshot.R = camera.R().clone();
            snapshot.T = camera.T().clone();
            snapshot.radial_distortion =
                camera.radial_distortion().is_valid() ? camera.radial_distortion().clone() : lfs::core::Tensor{};
            snapshot.tangential_distortion =
                camera.tangential_distortion().is_valid() ? camera.tangential_distortion().clone() : lfs::core::Tensor{};
            snapshot.device = camera.R().device();
            snapshot.camera_model_type = camera.camera_model_type();
            snapshot.image_name = camera.image_name();
            snapshot.image_path = camera.image_path();
            snapshot.mask_path = camera.mask_path();
            snapshot.focal_x = camera.focal_x();
            snapshot.focal_y = camera.focal_y();
            snapshot.center_x = camera.center_x();
            snapshot.center_y = camera.center_y();
            snapshot.camera_width = camera.camera_width();
            snapshot.camera_height = camera.camera_height();
            snapshot.image_width = camera.image_width();
            snapshot.image_height = camera.image_height();
            snapshot.uid = camera.uid();
            snapshot.camera_id = camera.camera_id();
            return snapshot;
        }

        std::shared_ptr<lfs::core::Camera> restoreCamera(const SceneGraphCameraSnapshot& snapshot) {
            auto camera = std::make_shared<lfs::core::Camera>(
                snapshot.R.clone(),
                snapshot.T.clone(),
                snapshot.focal_x,
                snapshot.focal_y,
                snapshot.center_x,
                snapshot.center_y,
                snapshot.radial_distortion.is_valid() ? snapshot.radial_distortion.clone() : lfs::core::Tensor{},
                snapshot.tangential_distortion.is_valid() ? snapshot.tangential_distortion.clone() : lfs::core::Tensor{},
                snapshot.camera_model_type,
                snapshot.image_name,
                snapshot.image_path,
                snapshot.mask_path,
                snapshot.camera_width,
                snapshot.camera_height,
                snapshot.uid,
                snapshot.camera_id);
            camera->set_image_dimensions(snapshot.image_width, snapshot.image_height);
            return camera;
        }

        SceneGraphNodeMetadataSnapshot captureNodeMetadataSnapshot(const SceneManager& scene_manager,
                                                                   const lfs::core::SceneNode& node) {
            SceneGraphNodeMetadataSnapshot snapshot;
            snapshot.name = node.name;
            snapshot.parent_name.clear();
            snapshot.local_transform = node.local_transform.get();
            snapshot.visible = node.visible.get();
            snapshot.locked = node.locked.get();
            snapshot.training_enabled = node.training_enabled;

            if (node.parent_id != lfs::core::NULL_NODE) {
                if (const auto* parent = scene_manager.getScene().getNodeById(node.parent_id)) {
                    snapshot.parent_name = parent->name;
                }
            }

            if (auto path = scene_manager.getPlyPath(node.name); path) {
                snapshot.source_path = *path;
            }

            return snapshot;
        }

        SceneGraphNodeSnapshot captureNodeSnapshot(const SceneManager& scene_manager,
                                                   const lfs::core::SceneNode& node,
                                                   const SceneGraphCaptureMode mode) {
            SceneGraphNodeSnapshot snapshot;
            snapshot.name = node.name;
            snapshot.type = node.type;
            snapshot.local_transform = node.local_transform.get();
            snapshot.visible = node.visible.get();
            snapshot.locked = node.locked.get();
            snapshot.training_enabled = node.training_enabled;
            snapshot.gaussian_count = node.gaussian_count.load(std::memory_order_acquire);
            snapshot.centroid = node.centroid;

            if (node.parent_id != lfs::core::NULL_NODE) {
                if (const auto* parent = scene_manager.getScene().getNodeById(node.parent_id)) {
                    snapshot.parent_name = parent->name;
                }
            }

            if (auto path = scene_manager.getPlyPath(node.name); path) {
                snapshot.source_path = *path;
            }

            if (mode == SceneGraphCaptureMode::FULL && node.model) {
                snapshot.payload_device = node.model->means_raw().device();
                snapshot.model = cloneSplatData(*node.model);
            }
            if (mode == SceneGraphCaptureMode::FULL && node.point_cloud) {
                snapshot.payload_device = node.point_cloud->means.device();
                snapshot.point_cloud = clonePointCloud(*node.point_cloud);
            }
            if (mode == SceneGraphCaptureMode::FULL && node.mesh) {
                snapshot.payload_device = node.mesh->vertices.device();
                snapshot.mesh = cloneMeshData(*node.mesh);
            }
            if (node.cropbox) {
                snapshot.cropbox = std::make_unique<lfs::core::CropBoxData>(*node.cropbox);
            }
            if (node.ellipsoid) {
                snapshot.ellipsoid = std::make_unique<lfs::core::EllipsoidData>(*node.ellipsoid);
            }
            if (node.keyframe) {
                snapshot.keyframe = std::make_unique<lfs::core::KeyframeData>(*node.keyframe);
            }
            if (node.camera) {
                snapshot.camera = captureCameraSnapshot(*node.camera);
            }

            for (const auto child_id : node.children) {
                if (const auto* child = scene_manager.getScene().getNodeById(child_id)) {
                    snapshot.children.push_back(captureNodeSnapshot(scene_manager, *child, mode));
                }
            }

            return snapshot;
        }

        std::vector<std::string> rootNames(const SceneGraphStateSnapshot& state) {
            std::vector<std::string> names;
            names.reserve(state.roots.size());
            for (const auto& root : state.roots) {
                names.push_back(root.name);
            }
            return names;
        }

        size_t estimateSnapshotBytes(const SceneGraphNodeSnapshot& snapshot) {
            size_t total = 0;
            if (snapshot.model) {
                total += tensorBytes(snapshot.model->means_raw());
                total += tensorBytes(snapshot.model->sh0_raw());
                total += tensorBytes(snapshot.model->shN_raw());
                total += tensorBytes(snapshot.model->scaling_raw());
                total += tensorBytes(snapshot.model->rotation_raw());
                total += tensorBytes(snapshot.model->opacity_raw());
                total += tensorBytes(snapshot.model->deleted());
                total += tensorBytes(snapshot.model->_densification_info);
            }
            if (snapshot.point_cloud) {
                total += tensorBytes(snapshot.point_cloud->means);
                total += tensorBytes(snapshot.point_cloud->colors);
                total += tensorBytes(snapshot.point_cloud->normals);
                total += tensorBytes(snapshot.point_cloud->sh0);
                total += tensorBytes(snapshot.point_cloud->shN);
                total += tensorBytes(snapshot.point_cloud->opacity);
                total += tensorBytes(snapshot.point_cloud->scaling);
                total += tensorBytes(snapshot.point_cloud->rotation);
            }
            if (snapshot.mesh) {
                total += tensorBytes(snapshot.mesh->vertices);
                total += tensorBytes(snapshot.mesh->normals);
                total += tensorBytes(snapshot.mesh->tangents);
                total += tensorBytes(snapshot.mesh->texcoords);
                total += tensorBytes(snapshot.mesh->colors);
                total += tensorBytes(snapshot.mesh->indices);
            }
            if (snapshot.camera) {
                total += tensorBytes(snapshot.camera->R);
                total += tensorBytes(snapshot.camera->T);
                total += tensorBytes(snapshot.camera->radial_distortion);
                total += tensorBytes(snapshot.camera->tangential_distortion);
            }
            for (const auto& child : snapshot.children) {
                total += estimateSnapshotBytes(child);
            }
            return total;
        }

        UndoMemoryBreakdown snapshotMemoryBreakdown(const SceneGraphNodeSnapshot& snapshot) {
            UndoMemoryBreakdown total;
            total.cpu_bytes += snapshot.name.size() + snapshot.parent_name.size() + sizeof(snapshot.local_transform) +
                               sizeof(snapshot.visible) + sizeof(snapshot.locked) +
                               sizeof(snapshot.training_enabled) + sizeof(snapshot.gaussian_count) +
                               sizeof(snapshot.centroid);
            if (snapshot.source_path) {
                total.cpu_bytes += snapshot.source_path->native().size();
            }
            if (snapshot.model) {
                total += estimateSplatDataMemory(*snapshot.model);
            }
            if (snapshot.point_cloud) {
                total += estimatePointCloudMemory(*snapshot.point_cloud);
            }
            if (snapshot.mesh) {
                total += estimateMeshMemory(*snapshot.mesh);
            }
            if (snapshot.camera) {
                total += estimateCameraMemory(*snapshot.camera);
            }
            for (const auto& child : snapshot.children) {
                total += snapshotMemoryBreakdown(child);
            }
            return total;
        }

        void offloadSnapshot(SceneGraphNodeSnapshot& snapshot) {
            if (snapshot.model) {
                offloadSplatData(*snapshot.model);
            }
            if (snapshot.point_cloud) {
                offloadPointCloud(*snapshot.point_cloud);
            }
            if (snapshot.mesh) {
                offloadMesh(*snapshot.mesh);
            }
            if (snapshot.camera) {
                offloadCamera(*snapshot.camera);
            }
            for (auto& child : snapshot.children) {
                offloadSnapshot(child);
            }
        }

        void restoreSnapshot(SceneGraphNodeSnapshot& snapshot) {
            if (snapshot.model) {
                restoreSplatDataToDevice(*snapshot.model, snapshot.payload_device);
            }
            if (snapshot.point_cloud) {
                restorePointCloudToDevice(*snapshot.point_cloud, snapshot.payload_device);
            }
            if (snapshot.mesh) {
                restoreMeshToDevice(*snapshot.mesh, snapshot.payload_device);
            }
            if (snapshot.camera) {
                restoreCameraToDevice(*snapshot.camera);
            }
            for (auto& child : snapshot.children) {
                restoreSnapshot(child);
            }
        }

        size_t estimateMetadataBytes(const SceneGraphNodeMetadataSnapshot& snapshot) {
            size_t total = snapshot.name.size() + snapshot.parent_name.size();
            if (snapshot.source_path) {
                total += snapshot.source_path->native().size();
            }
            total += sizeof(snapshot.local_transform) + sizeof(bool) * 3;
            return total;
        }

        void emitRemovedEvents(const SceneGraphNodeSnapshot& snapshot) {
            for (const auto& child : snapshot.children) {
                emitRemovedEvents(child);
            }
            PLYRemoved{
                .name = snapshot.name,
                .children_kept = false,
                .parent_of_removed = {},
                .from_history = true,
            }
                .emit();
        }

        void emitAddedEvents(const SceneManager& scene_manager, const SceneGraphNodeSnapshot& snapshot) {
            if (const auto* node = scene_manager.getScene().getNode(snapshot.name)) {
                PLYAdded{
                    .name = node->name,
                    .node_gaussians = node->gaussian_count.load(std::memory_order_acquire),
                    .total_gaussians = scene_manager.getScene().getTotalGaussianCount(),
                    .is_visible = node->visible,
                    .parent_name = snapshot.parent_name,
                    .is_group = node->type == lfs::core::NodeType::GROUP,
                    .node_type = static_cast<int>(node->type),
                    .from_history = true}
                    .emit();
            }
            for (const auto& child : snapshot.children) {
                emitAddedEvents(scene_manager, child);
            }
        }

        bool restoreNodeSnapshot(SceneManager& scene_manager, const SceneGraphNodeSnapshot& snapshot) {
            auto& scene = scene_manager.getScene();
            lfs::core::NodeId parent_id = lfs::core::NULL_NODE;
            if (!snapshot.parent_name.empty()) {
                parent_id = scene.getNodeIdByName(snapshot.parent_name);
                if (parent_id == lfs::core::NULL_NODE) {
                    LOG_ERROR("Cannot restore node '{}': missing parent '{}'",
                              snapshot.name, snapshot.parent_name);
                    return false;
                }
            }

            lfs::core::NodeId node_id = lfs::core::NULL_NODE;
            switch (snapshot.type) {
            case lfs::core::NodeType::GROUP:
                node_id = scene.addGroup(snapshot.name, parent_id);
                break;
            case lfs::core::NodeType::SPLAT:
                if (!snapshot.model)
                    return false;
                node_id = scene.addSplat(snapshot.name, cloneSplatData(*snapshot.model), parent_id);
                break;
            case lfs::core::NodeType::POINTCLOUD:
                if (!snapshot.point_cloud)
                    return false;
                node_id = scene.addPointCloud(snapshot.name, clonePointCloud(*snapshot.point_cloud), parent_id);
                break;
            case lfs::core::NodeType::MESH:
                if (!snapshot.mesh)
                    return false;
                node_id = scene.addMesh(snapshot.name, cloneMeshData(*snapshot.mesh), parent_id);
                break;
            case lfs::core::NodeType::CROPBOX:
                node_id = scene.addCropBox(snapshot.name, parent_id);
                if (node_id != lfs::core::NULL_NODE && snapshot.cropbox) {
                    scene.setCropBoxData(node_id, *snapshot.cropbox);
                }
                break;
            case lfs::core::NodeType::ELLIPSOID:
                node_id = scene.addEllipsoid(snapshot.name, parent_id);
                if (node_id != lfs::core::NULL_NODE && snapshot.ellipsoid) {
                    scene.setEllipsoidData(node_id, *snapshot.ellipsoid);
                }
                break;
            case lfs::core::NodeType::DATASET:
                node_id = scene.addDataset(snapshot.name);
                break;
            case lfs::core::NodeType::CAMERA_GROUP:
                node_id = scene.addCameraGroup(snapshot.name, parent_id, snapshot.gaussian_count);
                break;
            case lfs::core::NodeType::CAMERA:
                if (!snapshot.camera)
                    return false;
                node_id = scene.addCamera(snapshot.name, parent_id, restoreCamera(*snapshot.camera));
                break;
            case lfs::core::NodeType::KEYFRAME_GROUP:
                node_id = scene.addKeyframeGroup(snapshot.name, parent_id);
                break;
            case lfs::core::NodeType::KEYFRAME:
                if (!snapshot.keyframe)
                    return false;
                node_id = scene.addKeyframe(snapshot.name, parent_id,
                                            std::make_unique<lfs::core::KeyframeData>(*snapshot.keyframe));
                break;
            default:
                throw std::runtime_error("Unsupported node type in scene graph undo");
            }

            if (node_id == lfs::core::NULL_NODE) {
                return false;
            }

            auto* restored = scene.getMutableNode(snapshot.name);
            if (!restored) {
                return false;
            }
            restored->local_transform.setQuiet(snapshot.local_transform);
            restored->visible.setQuiet(snapshot.visible);
            restored->locked.setQuiet(snapshot.locked);
            restored->training_enabled = snapshot.training_enabled;
            restored->gaussian_count.store(snapshot.gaussian_count, std::memory_order_release);
            restored->centroid = snapshot.centroid;
            restored->transform_dirty = true;

            if (snapshot.source_path) {
                scene_manager.setPlyPath(snapshot.name, *snapshot.source_path);
            }

            for (const auto& child : snapshot.children) {
                if (!restoreNodeSnapshot(scene_manager, child)) {
                    return false;
                }
            }

            return true;
        }

        void restoreNodeSelection(SceneManager& scene_manager,
                                  const std::vector<std::string>& selected_node_names) {
            scene_manager.clearSelection();
            if (selected_node_names.size() == 1) {
                scene_manager.selectNode(selected_node_names.front());
                return;
            }
            if (!selected_node_names.empty()) {
                scene_manager.selectNodes(selected_node_names);
            }
        }
    } // namespace

    SceneGraphNodeSnapshot::SceneGraphNodeSnapshot() = default;

    SceneGraphNodeSnapshot::SceneGraphNodeSnapshot(const SceneGraphNodeSnapshot& other)
        : name(other.name),
          parent_name(other.parent_name),
          type(other.type),
          local_transform(other.local_transform),
          visible(other.visible),
          locked(other.locked),
          training_enabled(other.training_enabled),
          gaussian_count(other.gaussian_count),
          centroid(other.centroid),
          payload_device(other.payload_device),
          source_path(other.source_path),
          camera(other.camera),
          children(other.children) {
        if (other.model) {
            model = cloneSplatData(*other.model);
        }
        if (other.point_cloud) {
            point_cloud = clonePointCloud(*other.point_cloud);
        }
        if (other.mesh) {
            mesh = cloneMeshData(*other.mesh);
        }
        if (other.cropbox) {
            cropbox = std::make_unique<lfs::core::CropBoxData>(*other.cropbox);
        }
        if (other.ellipsoid) {
            ellipsoid = std::make_unique<lfs::core::EllipsoidData>(*other.ellipsoid);
        }
        if (other.keyframe) {
            keyframe = std::make_unique<lfs::core::KeyframeData>(*other.keyframe);
        }
    }

    SceneGraphNodeSnapshot::SceneGraphNodeSnapshot(SceneGraphNodeSnapshot&& other) noexcept = default;

    SceneGraphNodeSnapshot& SceneGraphNodeSnapshot::operator=(const SceneGraphNodeSnapshot& other) {
        if (this == &other) {
            return *this;
        }

        SceneGraphNodeSnapshot copy(other);
        *this = std::move(copy);
        return *this;
    }

    SceneGraphNodeSnapshot& SceneGraphNodeSnapshot::operator=(SceneGraphNodeSnapshot&& other) noexcept = default;

    SceneGraphNodeSnapshot::~SceneGraphNodeSnapshot() = default;

    UndoMemoryBreakdown TensorSwapStorage::memoryBreakdown() const {
        UndoMemoryBreakdown total;
        total += tensorMemory(indices);
        total += tensorMemory(stored_values);
        return total;
    }

    void TensorSwapStorage::offloadToCPU() {
        offloadTensor(indices);
        offloadTensor(stored_values);
    }

    void TensorSwapStorage::restoreToDevice() {
        restoreTensorToDevice(indices, device);
        restoreTensorToDevice(stored_values, device);
    }

    SceneSnapshot::SceneSnapshot(SceneManager& scene, std::string name)
        : scene_(scene),
          name_(std::move(name)) {}

    void SceneSnapshot::captureDeletedMasks(
        std::unordered_map<std::string, TensorPresenceSnapshot>& target) {
        target.clear();

        for (const auto* node : scene_.getScene().getNodes()) {
            if (!node || !node->model) {
                continue;
            }

            TensorPresenceSnapshot snapshot;
            snapshot.total_size = node->model->size();
            snapshot.device = node->model->means_raw().device();
            snapshot.present = node->model->has_deleted_mask();
            if (snapshot.present) {
                snapshot.tensor = std::make_shared<lfs::core::Tensor>(node->model->deleted().clone());
            }
            target[node->name] = std::move(snapshot);
        }
    }

    void SceneSnapshot::captureSelection() {
        selection_before_ = scene_.getScene().captureSelectionState();
        captured_ = captured_ | ModifiesFlag::SELECTION;
    }

    void SceneSnapshot::captureTransforms(const std::vector<std::string>& nodes) {
        for (const auto& node_name : nodes) {
            transforms_before_[node_name] = scene_.getNodeTransform(node_name);
        }
        captured_ = captured_ | ModifiesFlag::TRANSFORMS;
    }

    bool SceneSnapshot::captureTransformsBefore(const std::vector<std::string>& nodes,
                                                const std::vector<glm::mat4>& transforms) {
        if (nodes.size() != transforms.size()) {
            LOG_ERROR("Cannot capture transform snapshot: {} node names but {} transforms",
                      nodes.size(), transforms.size());
            return false;
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            transforms_before_[nodes[i]] = transforms[i];
        }
        captured_ = captured_ | ModifiesFlag::TRANSFORMS;
        return true;
    }

    void SceneSnapshot::captureTopology() {
        captureDeletedMasks(deleted_masks_before_);
        combined_deleted_before_.reset();
        combined_deleted_storage_ = {};
        if (scene_.getScene().isConsolidated()) {
            if (const auto* combined = scene_.getScene().getCombinedModel()) {
                TensorPresenceSnapshot snapshot;
                snapshot.total_size = combined->size();
                snapshot.device = combined->means_raw().device();
                snapshot.present = combined->has_deleted_mask();
                if (snapshot.present) {
                    snapshot.tensor =
                        std::make_shared<lfs::core::Tensor>(combined->deleted().clone());
                }
                combined_deleted_before_ = std::move(snapshot);
            }
        }
        captured_ = captured_ | ModifiesFlag::TOPOLOGY;
    }

    void SceneSnapshot::compactSelection() {
        selection_after_metadata_ = scene_.getScene().captureSelectionStateMetadata();

        const auto selection_after = scene_.getScene().getSelectionMask();
        const size_t total_size = std::max({scene_.getScene().getTotalGaussianCount(),
                                            tensorNumel(selection_before_.mask),
                                            tensorNumel(selection_after)});
        const auto fallback_device =
            (selection_after && selection_after->is_valid())
                ? selection_after->device()
                : ((selection_before_.mask && selection_before_.mask->is_valid())
                       ? selection_before_.mask->device()
                       : lfs::core::Device::CUDA);
        selection_mask_storage_ = buildTensorSwapStorage(
            selection_before_.mask,
            selection_after.get(),
            total_size,
            fallback_device,
            lfs::core::DataType::UInt8,
            selection_before_.has_selection,
            selection_after_metadata_.has_selection);
        selection_before_.mask.reset();
    }

    void SceneSnapshot::compactTopology() {
        deleted_mask_storage_.clear();
        combined_deleted_storage_ = {};

        for (const auto& [node_name, before] : deleted_masks_before_) {
            const auto* node = scene_.getScene().getNode(node_name);
            if (!node || !node->model) {
                continue;
            }

            const auto* after_tensor =
                node->model->has_deleted_mask() ? &node->model->deleted() : nullptr;
            const size_t model_size = static_cast<size_t>(node->model->size());
            const size_t total_size =
                std::max({before.total_size, tensorNumel(before.tensor), tensorNumel(after_tensor), model_size});

            auto storage = buildTensorSwapStorage(
                before.tensor,
                after_tensor,
                total_size,
                before.device,
                lfs::core::DataType::Bool,
                before.present,
                node->model->has_deleted_mask());
            if (storage.hasChanges()) {
                deleted_mask_storage_.emplace(node_name, std::move(storage));
            }
        }

        deleted_masks_before_.clear();

        if (combined_deleted_before_) {
            const auto& before = *combined_deleted_before_;
            const auto* combined = scene_.getScene().isConsolidated()
                                       ? scene_.getScene().getCombinedModel()
                                       : nullptr;
            const auto* after_tensor =
                (combined && combined->has_deleted_mask()) ? &combined->deleted() : nullptr;
            const size_t model_size = combined ? static_cast<size_t>(combined->size()) : 0;
            const size_t total_size =
                std::max({before.total_size,
                          tensorNumel(before.tensor),
                          tensorNumel(after_tensor),
                          model_size});

            combined_deleted_storage_ = buildTensorSwapStorage(
                before.tensor,
                after_tensor,
                total_size,
                before.device,
                lfs::core::DataType::Bool,
                before.present,
                combined && combined->has_deleted_mask());
        }

        combined_deleted_before_.reset();
    }

    void SceneSnapshot::captureAfter() {
        if (hasFlag(captured_, ModifiesFlag::SELECTION)) {
            compactSelection();
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, _] : transforms_before_) {
                transforms_after_[node_name] = scene_.getNodeTransform(node_name);
            }
        }

        if (hasFlag(captured_, ModifiesFlag::TOPOLOGY)) {
            compactTopology();
        }
    }

    bool SceneSnapshot::hasChanges() const {
        if (hasFlag(captured_, ModifiesFlag::SELECTION) &&
            (selection_mask_storage_.hasChanges() ||
             !selectionMetadataEqual(selection_before_, selection_after_metadata_))) {
            return true;
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS) && transforms_before_ != transforms_after_) {
            return true;
        }

        if (hasFlag(captured_, ModifiesFlag::TOPOLOGY) &&
            (!deleted_mask_storage_.empty() || combined_deleted_storage_.hasChanges())) {
            return true;
        }

        return false;
    }

    void SceneSnapshot::applySelection(const bool undo_direction) {
        const auto target_metadata = undo_direction
                                         ? lfs::core::Scene::SelectionStateMetadata{
                                               .groups = selection_before_.groups,
                                               .active_group_id = selection_before_.active_group_id,
                                               .next_group_id = selection_before_.next_group_id,
                                               .has_selection = selection_before_.has_selection,
                                           }
                                         : selection_after_metadata_;

        if (selection_mask_storage_.mode == TensorSwapStorageMode::SPARSE &&
            scene_.getScene().getTotalGaussianCount() != selection_mask_storage_.total_size) {
            throw std::runtime_error("Cannot replay sparse selection history after topology changed");
        }

        auto current_selection = scene_.getScene().getSelectionMask();
        const size_t total_size = std::max({selection_mask_storage_.total_size,
                                            scene_.getScene().getTotalGaussianCount(),
                                            tensorNumel(current_selection)});
        auto working_mask = materializeMaskTensor(
            current_selection, total_size, selection_mask_storage_.device, lfs::core::DataType::UInt8);

        if (selection_mask_storage_.hasChanges()) {
            applyTensorSwapStorage(working_mask, selection_mask_storage_);
        }

        lfs::core::Scene::SelectionStateSnapshot snapshot;
        snapshot.groups = target_metadata.groups;
        snapshot.active_group_id = target_metadata.active_group_id;
        snapshot.next_group_id = target_metadata.next_group_id;
        snapshot.has_selection = target_metadata.has_selection;
        if (target_metadata.has_selection) {
            snapshot.mask = std::make_shared<lfs::core::Tensor>(std::move(working_mask));
        }
        scene_.getScene().restoreSelectionState(snapshot);
    }

    void SceneSnapshot::applyTopology(const bool undo_direction) {
        bool restored_any = false;

        for (auto& [node_name, storage] : deleted_mask_storage_) {
            auto* node = scene_.getScene().getMutableNode(node_name);
            if (!node || !node->model) {
                continue;
            }

            if (storage.mode == TensorSwapStorageMode::SPARSE &&
                node->model->size() != storage.total_size) {
                throw std::runtime_error("Cannot replay sparse topology history after gaussian count changed");
            }

            const bool target_present = undo_direction ? storage.before_present : storage.after_present;
            const lfs::core::Tensor* current_deleted =
                node->model->has_deleted_mask() ? &node->model->deleted() : nullptr;
            const size_t model_size = static_cast<size_t>(node->model->size());
            const size_t total_size =
                std::max({storage.total_size, model_size, tensorNumel(current_deleted)});
            auto working_mask = materializeMaskTensor(
                current_deleted, total_size, storage.device, lfs::core::DataType::Bool);

            if (storage.hasChanges()) {
                applyTensorSwapStorage(working_mask, storage);
            }

            if (target_present) {
                node->model->deleted() = std::move(working_mask);
            } else {
                node->model->deleted() = lfs::core::Tensor{};
            }
            restored_any = true;
        }

        if (combined_deleted_storage_.hasChanges()) {
            if (!scene_.getScene().isConsolidated()) {
                throw std::runtime_error("Cannot replay consolidated topology history after scene unconsolidated");
            }

            auto* combined = const_cast<lfs::core::SplatData*>(scene_.getScene().getCombinedModel());
            if (!combined) {
                throw std::runtime_error("Cannot restore consolidated deleted mask without combined model");
            }

            if (combined_deleted_storage_.mode == TensorSwapStorageMode::SPARSE &&
                combined->size() != combined_deleted_storage_.total_size) {
                throw std::runtime_error("Cannot replay consolidated topology history after gaussian count changed");
            }

            const bool target_present =
                undo_direction ? combined_deleted_storage_.before_present : combined_deleted_storage_.after_present;
            const lfs::core::Tensor* current_deleted =
                combined->has_deleted_mask() ? &combined->deleted() : nullptr;
            const size_t model_size = static_cast<size_t>(combined->size());
            const size_t total_size =
                std::max({combined_deleted_storage_.total_size, model_size, tensorNumel(current_deleted)});
            auto working_mask = materializeMaskTensor(
                current_deleted, total_size, combined_deleted_storage_.device, lfs::core::DataType::Bool);

            applyTensorSwapStorage(working_mask, combined_deleted_storage_);

            if (target_present) {
                combined->deleted() = std::move(working_mask);
            } else {
                combined->deleted() = lfs::core::Tensor{};
            }
            restored_any = true;
        }

        if (restored_any) {
            scene_.getScene().notifyMutation(lfs::core::Scene::MutationType::MODEL_CHANGED);
        }
    }

    void SceneSnapshot::undo() {
        if (hasFlag(captured_, ModifiesFlag::SELECTION)) {
            applySelection(true);
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, transform] : transforms_before_) {
                scene_.setNodeTransform(node_name, transform);
            }
        }

        if (hasFlag(captured_, ModifiesFlag::TOPOLOGY)) {
            applyTopology(true);
        }
    }

    void SceneSnapshot::redo() {
        if (hasFlag(captured_, ModifiesFlag::SELECTION)) {
            applySelection(false);
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, transform] : transforms_after_) {
                scene_.setNodeTransform(node_name, transform);
            }
        }

        if (hasFlag(captured_, ModifiesFlag::TOPOLOGY)) {
            applyTopology(false);
        }
    }

    size_t SceneSnapshot::estimatedBytes() const {
        size_t total = selection_mask_storage_.estimatedBytes();
        for (const auto& [_, storage] : deleted_mask_storage_) {
            total += storage.estimatedBytes();
        }
        total += combined_deleted_storage_.estimatedBytes();
        for (const auto& [node_name, _] : transforms_before_) {
            total += sizeof(glm::mat4) + node_name.size();
        }
        for (const auto& [node_name, _] : transforms_after_) {
            total += sizeof(glm::mat4) + node_name.size();
        }
        return total;
    }

    DirtyMask SceneSnapshot::dirtyFlags() const {
        return snapshotDirtyFlags(captured_);
    }

    UndoMemoryBreakdown SceneSnapshot::memoryBreakdown() const {
        UndoMemoryBreakdown total = selection_mask_storage_.memoryBreakdown();
        for (const auto& [_, storage] : deleted_mask_storage_) {
            total += storage.memoryBreakdown();
        }
        total += combined_deleted_storage_.memoryBreakdown();
        for (const auto& [node_name, _] : transforms_before_) {
            total.cpu_bytes += sizeof(glm::mat4) + node_name.size();
        }
        for (const auto& [node_name, _] : transforms_after_) {
            total.cpu_bytes += sizeof(glm::mat4) + node_name.size();
        }
        return total;
    }

    void SceneSnapshot::offloadToCPU() {
        selection_mask_storage_.offloadToCPU();
        for (auto& [_, storage] : deleted_mask_storage_) {
            storage.offloadToCPU();
        }
        combined_deleted_storage_.offloadToCPU();
    }

    void SceneSnapshot::restoreToPreferredDevice() {
        selection_mask_storage_.restoreToDevice();
        for (auto& [_, storage] : deleted_mask_storage_) {
            storage.restoreToDevice();
        }
        combined_deleted_storage_.restoreToDevice();
    }

    UndoMetadata SceneSnapshot::metadata() const {
        return UndoMetadata{
            .id = "scene.snapshot",
            .label = name_,
            .source = "core",
            .scope = snapshotScope(captured_),
        };
    }

    bool pushSceneSnapshotIfChanged(std::unique_ptr<SceneSnapshot> snapshot) {
        if (!snapshot || !snapshot->hasChanges()) {
            return false;
        }
        undoHistory().push(std::move(snapshot));
        return true;
    }

    TensorUndoEntry::TensorUndoEntry(std::string name,
                                     UndoMetadata metadata,
                                     std::string target_name,
                                     lfs::core::Tensor before,
                                     TensorAccessor accessor)
        : name_(std::move(name)),
          metadata_(std::move(metadata)),
          target_name_(std::move(target_name)),
          accessor_(std::move(accessor)),
          before_(std::move(before)) {
        if (!metadata_.label.size()) {
            metadata_.label = name_;
        }
        if (!before_.is_valid()) {
            return;
        }

        tensor_shape_ = before_.shape();
        element_count_ = before_.numel();
        dtype_ = before_.dtype();
        storage_.device = before_.device();
        storage_.dtype = dtype_;
        storage_.total_size = element_count_;
        storage_.before_present = true;
    }

    void TensorUndoEntry::captureAfter() {
        captured_after_ = true;
        auto* current = accessor_ ? accessor_() : nullptr;
        if (!before_.is_valid() || !current || !current->is_valid()) {
            LOG_WARN("Tensor undo capture skipped for '{}': missing tensor target '{}'",
                     name_, target_name_);
            return;
        }
        if (current->dtype() != dtype_ || current->numel() != element_count_) {
            LOG_WARN("Tensor undo capture skipped for '{}': incompatible tensor state on '{}'",
                     name_, target_name_);
            return;
        }
        if (element_count_ > static_cast<size_t>(std::numeric_limits<int>::max())) {
            LOG_WARN("Tensor undo capture skipped for '{}': tensor '{}' exceeds supported reshape size",
                     name_, target_name_);
            return;
        }

        const int flat_size = static_cast<int>(element_count_);
        auto before_flat = before_.contiguous().reshape({flat_size});
        auto after_flat = current->contiguous().reshape({flat_size});
        auto before_ptr = std::make_shared<lfs::core::Tensor>(std::move(before_flat));
        storage_ = buildTensorSwapStorage(before_ptr,
                                          &after_flat,
                                          element_count_,
                                          current->device(),
                                          dtype_,
                                          true,
                                          true);
        before_ = lfs::core::Tensor{};
    }

    bool TensorUndoEntry::hasChanges() const {
        return captured_after_ && storage_.hasChanges();
    }

    void TensorUndoEntry::apply() {
        auto* current = accessor_ ? accessor_() : nullptr;
        if (!current || !current->is_valid()) {
            throw std::runtime_error("Missing tensor target for undo entry '" + target_name_ + "'");
        }
        if (current->dtype() != dtype_ || current->numel() != element_count_) {
            throw std::runtime_error("Incompatible tensor target for undo entry '" + target_name_ + "'");
        }
        if (element_count_ > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Tensor target for undo entry '" + target_name_ + "' exceeds supported reshape size");
        }

        const int flat_size = static_cast<int>(element_count_);
        auto flat = current->contiguous().reshape({flat_size});
        applyTensorSwapStorage(flat, storage_);
        *current = flat.reshape(tensor_shape_).contiguous();
    }

    void TensorUndoEntry::undo() {
        apply();
    }

    void TensorUndoEntry::redo() {
        apply();
    }

    DirtyMask TensorUndoEntry::dirtyFlags() const {
        return DirtyFlag::SPLATS;
    }

    CropBoxUndoEntry::CropBoxUndoEntry(SceneManager& scene,
                                       RenderingManager* rendering_manager,
                                       std::string node_name,
                                       lfs::core::CropBoxData before,
                                       glm::mat4 transform_before,
                                       bool show_before,
                                       bool use_before)
        : scene_(scene),
          rendering_manager_(rendering_manager),
          node_name_(std::move(node_name)),
          before_(std::move(before)),
          transform_before_(transform_before),
          show_before_(show_before),
          use_before_(use_before) {
        captureAfter();
    }

    void CropBoxUndoEntry::captureAfter() {
        const auto* node = scene_.getScene().getNode(node_name_);
        if (!node || !node->cropbox) {
            LOG_WARN("CropBox node '{}' removed during undo capture", node_name_);
            after_ = before_;
            transform_after_ = transform_before_;
            show_after_ = show_before_;
            use_after_ = use_before_;
            return;
        }
        after_ = *node->cropbox;
        transform_after_ = scene_.getNodeTransform(node_name_);
        if (rendering_manager_) {
            const auto settings = rendering_manager_->getSettings();
            show_after_ = settings.show_crop_box;
            use_after_ = settings.use_crop_box;
        } else {
            show_after_ = show_before_;
            use_after_ = use_before_;
        }
    }

    void CropBoxUndoEntry::undo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->cropbox) {
            *node->cropbox = before_;
            scene_.setNodeTransform(node_name_, transform_before_);
        }
        if (rendering_manager_) {
            auto settings = rendering_manager_->getSettings();
            settings.show_crop_box = show_before_;
            settings.use_crop_box = use_before_;
            rendering_manager_->updateSettings(settings);
        }
    }

    void CropBoxUndoEntry::redo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->cropbox) {
            *node->cropbox = after_;
            scene_.setNodeTransform(node_name_, transform_after_);
        }
        if (rendering_manager_) {
            auto settings = rendering_manager_->getSettings();
            settings.show_crop_box = show_after_;
            settings.use_crop_box = use_after_;
            rendering_manager_->updateSettings(settings);
        }
    }

    bool CropBoxUndoEntry::hasChanges() const {
        return !cropBoxesEqual(before_, after_) ||
               transform_before_ != transform_after_ ||
               show_before_ != show_after_ ||
               use_before_ != use_after_;
    }

    UndoMetadata CropBoxUndoEntry::metadata() const {
        return UndoMetadata{
            .id = "cropbox.transform",
            .label = "Crop Box Transform",
            .source = "ui",
            .scope = "cropbox",
        };
    }

    DirtyMask CropBoxUndoEntry::dirtyFlags() const {
        return DirtyFlag::SPLATS | DirtyFlag::OVERLAY;
    }

    EllipsoidUndoEntry::EllipsoidUndoEntry(SceneManager& scene,
                                           RenderingManager* rendering_manager,
                                           std::string node_name,
                                           lfs::core::EllipsoidData before,
                                           glm::mat4 transform_before,
                                           bool show_before,
                                           bool use_before)
        : scene_(scene),
          rendering_manager_(rendering_manager),
          node_name_(std::move(node_name)),
          before_(std::move(before)),
          transform_before_(transform_before),
          show_before_(show_before),
          use_before_(use_before) {
        captureAfter();
    }

    void EllipsoidUndoEntry::captureAfter() {
        const auto* node = scene_.getScene().getNode(node_name_);
        if (!node || !node->ellipsoid) {
            LOG_WARN("Ellipsoid node '{}' removed during undo capture", node_name_);
            after_ = before_;
            transform_after_ = transform_before_;
            show_after_ = show_before_;
            use_after_ = use_before_;
            return;
        }
        after_ = *node->ellipsoid;
        transform_after_ = scene_.getNodeTransform(node_name_);
        if (rendering_manager_) {
            const auto settings = rendering_manager_->getSettings();
            show_after_ = settings.show_ellipsoid;
            use_after_ = settings.use_ellipsoid;
        } else {
            show_after_ = show_before_;
            use_after_ = use_before_;
        }
    }

    void EllipsoidUndoEntry::undo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->ellipsoid) {
            *node->ellipsoid = before_;
            scene_.setNodeTransform(node_name_, transform_before_);
        }
        if (rendering_manager_) {
            auto settings = rendering_manager_->getSettings();
            settings.show_ellipsoid = show_before_;
            settings.use_ellipsoid = use_before_;
            rendering_manager_->updateSettings(settings);
        }
    }

    void EllipsoidUndoEntry::redo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->ellipsoid) {
            *node->ellipsoid = after_;
            scene_.setNodeTransform(node_name_, transform_after_);
        }
        if (rendering_manager_) {
            auto settings = rendering_manager_->getSettings();
            settings.show_ellipsoid = show_after_;
            settings.use_ellipsoid = use_after_;
            rendering_manager_->updateSettings(settings);
        }
    }

    bool EllipsoidUndoEntry::hasChanges() const {
        return !ellipsoidsEqual(before_, after_) ||
               transform_before_ != transform_after_ ||
               show_before_ != show_after_ ||
               use_before_ != use_after_;
    }

    UndoMetadata EllipsoidUndoEntry::metadata() const {
        return UndoMetadata{
            .id = "ellipsoid.transform",
            .label = "Ellipsoid Transform",
            .source = "ui",
            .scope = "ellipsoid",
        };
    }

    DirtyMask EllipsoidUndoEntry::dirtyFlags() const {
        return DirtyFlag::SPLATS | DirtyFlag::OVERLAY;
    }

    PropertyChangeUndoEntry::PropertyChangeUndoEntry(std::string property_path,
                                                     std::any before,
                                                     std::any after,
                                                     std::function<void(const std::any&)> applier)
        : property_path_(std::move(property_path)),
          label_(propertyUndoLabel(property_path_)),
          before_(std::move(before)),
          after_(std::move(after)),
          applier_(std::move(applier)),
          estimated_bytes_(estimateAnyBytes(before_) + estimateAnyBytes(after_)),
          updated_at_(std::chrono::steady_clock::now()) {}

    void PropertyChangeUndoEntry::undo() {
        applier_(before_);
    }

    void PropertyChangeUndoEntry::redo() {
        applier_(after_);
    }

    UndoMetadata PropertyChangeUndoEntry::metadata() const {
        return UndoMetadata{
            .id = property_path_,
            .label = label_,
            .source = "property",
            .scope = propertyUndoScope(property_path_),
        };
    }

    bool PropertyChangeUndoEntry::tryMerge(const UndoEntry& incoming) {
        const auto* next = dynamic_cast<const PropertyChangeUndoEntry*>(&incoming);
        if (!next || next->property_path_ != property_path_ || next->updated_at_ < updated_at_) {
            return false;
        }

        if ((next->updated_at_ - updated_at_) > PROPERTY_COALESCE_WINDOW) {
            return false;
        }

        after_ = next->after_;
        updated_at_ = next->updated_at_;
        estimated_bytes_ = estimateAnyBytes(before_) + estimateAnyBytes(after_);
        return true;
    }

    DirtyMask PropertyChangeUndoEntry::dirtyFlags() const {
        return propertyDirtyFlags(property_path_);
    }

    std::vector<SceneGraphNodeMetadataSnapshot> SceneGraphMetadataEntry::captureNodes(
        const SceneManager& scene_manager,
        const std::vector<std::string>& node_names) {
        std::vector<SceneGraphNodeMetadataSnapshot> snapshots;
        std::set<std::string> seen;
        snapshots.reserve(node_names.size());
        for (const auto& node_name : node_names) {
            if (node_name.empty() || !seen.insert(node_name).second) {
                continue;
            }
            const auto* node = scene_manager.getScene().getNode(node_name);
            if (!node) {
                continue;
            }
            snapshots.push_back(captureNodeMetadataSnapshot(scene_manager, *node));
        }
        return snapshots;
    }

    SceneGraphMetadataEntry::SceneGraphMetadataEntry(SceneManager& scene,
                                                     std::string name,
                                                     std::vector<SceneGraphNodeMetadataDiff> diffs)
        : scene_(scene),
          name_(std::move(name)),
          diffs_(std::move(diffs)),
          updated_at_(std::chrono::steady_clock::now()) {}

    void SceneGraphMetadataEntry::apply(const bool use_after_state) {
        lfs::core::Scene::Transaction txn(scene_.getScene());

        struct AppliedMetadataSnapshot {
            SceneGraphNodeMetadataSnapshot original;
            std::vector<std::string> candidates;
        };

        std::vector<AppliedMetadataSnapshot> applied;
        applied.reserve(diffs_.size());

        for (const auto& diff : diffs_) {
            const auto& target = use_after_state ? diff.after : diff.before;
            const auto& alternate = use_after_state ? diff.before : diff.after;
            const std::vector<std::string> candidates{target.name, alternate.name};
            const auto current_name = resolveExistingNodeName(scene_.getScene(), candidates);
            if (current_name.empty()) {
                throw std::runtime_error("Cannot restore scene node metadata for '" + target.name + "'");
            }

            const auto* current_node = scene_.getScene().getNode(current_name);
            if (!current_node) {
                throw std::runtime_error("Cannot restore scene node metadata for '" + target.name + "'");
            }

            AppliedMetadataSnapshot rollback_state{
                .original = captureNodeMetadataSnapshot(scene_, *current_node),
                .candidates = {current_name, target.name, alternate.name},
            };

            try {
                applyNodeMetadataSnapshot(scene_, target, rollback_state.candidates, true);
            } catch (const HistoryCorruptionError&) {
                throw;
            } catch (...) {
                try {
                    for (auto it = applied.rbegin(); it != applied.rend(); ++it) {
                        applyNodeMetadataSnapshot(scene_, it->original, it->candidates, false);
                    }
                } catch (const std::exception& rollback_error) {
                    throw HistoryCorruptionError(
                        "Failed to rollback scene graph metadata entry '" + name_ + "': " +
                        std::string(rollback_error.what()));
                } catch (...) {
                    throw HistoryCorruptionError(
                        "Failed to rollback scene graph metadata entry '" + name_ + "': unknown exception");
                }
                throw;
            }

            applied.push_back(std::move(rollback_state));
        }
    }

    void SceneGraphMetadataEntry::undo() {
        apply(false);
    }

    void SceneGraphMetadataEntry::redo() {
        apply(true);
    }

    size_t SceneGraphMetadataEntry::estimatedBytes() const {
        size_t total = 0;
        for (const auto& diff : diffs_) {
            total += estimateMetadataBytes(diff.before);
            total += estimateMetadataBytes(diff.after);
        }
        return total;
    }

    bool SceneGraphMetadataEntry::tryMerge(const UndoEntry& incoming) {
        const auto* next = dynamic_cast<const SceneGraphMetadataEntry*>(&incoming);
        if (!next || next->name_ != name_ || next->updated_at_ < updated_at_) {
            return false;
        }

        if ((next->updated_at_ - updated_at_) > PROPERTY_COALESCE_WINDOW) {
            return false;
        }

        if (diffs_.size() != next->diffs_.size()) {
            return false;
        }

        const auto merge_key = [](const SceneGraphNodeMetadataDiff& diff) {
            std::array<std::string, 2> names{diff.before.name, diff.after.name};
            std::sort(names.begin(), names.end());
            return names;
        };

        for (size_t i = 0; i < diffs_.size(); ++i) {
            if (merge_key(diffs_[i]) != merge_key(next->diffs_[i])) {
                return false;
            }
        }

        for (size_t i = 0; i < diffs_.size(); ++i) {
            diffs_[i].after = next->diffs_[i].after;
        }
        updated_at_ = next->updated_at_;
        return true;
    }

    UndoMetadata SceneGraphMetadataEntry::metadata() const {
        return UndoMetadata{
            .id = "scene_graph.metadata",
            .label = name_,
            .source = "core",
            .scope = "scene_graph",
        };
    }

    DirtyMask SceneGraphMetadataEntry::dirtyFlags() const {
        return sceneGraphMetadataDirtyFlags();
    }

    SceneGraphStateSnapshot SceneGraphPatchEntry::captureState(const SceneManager& scene_manager,
                                                               const std::vector<std::string>& root_names,
                                                               const SceneGraphCaptureOptions options) {
        SceneGraphStateSnapshot snapshot;
        if (options.include_selected_nodes) {
            snapshot.selected_node_names = scene_manager.getSelectedNodeNames();
        }
        if (options.include_scene_context) {
            snapshot.context = SceneGraphContextSnapshot{
                .content_type = static_cast<int>(scene_manager.getContentType()),
                .dataset_path = scene_manager.getDatasetPath(),
                .training_model_node_name = scene_manager.getScene().getTrainingModelNodeName(),
            };
        }

        std::vector<std::string> unique_names;
        unique_names.reserve(root_names.size());
        std::set<std::string> seen;
        for (const auto& name : root_names) {
            if (!name.empty() && seen.insert(name).second) {
                unique_names.push_back(name);
            }
        }

        const std::set<std::string> requested(unique_names.begin(), unique_names.end());
        for (const auto& name : unique_names) {
            const auto* node = scene_manager.getScene().getNode(name);
            if (!node) {
                continue;
            }

            bool covered_by_parent = false;
            for (auto parent_id = node->parent_id; parent_id != lfs::core::NULL_NODE;) {
                const auto* parent = scene_manager.getScene().getNodeById(parent_id);
                if (!parent) {
                    break;
                }
                if (requested.contains(parent->name)) {
                    covered_by_parent = true;
                    break;
                }
                parent_id = parent->parent_id;
            }

            if (!covered_by_parent) {
                snapshot.roots.push_back(captureNodeSnapshot(scene_manager, *node, options.mode));
            }
        }

        return snapshot;
    }

    SceneGraphPatchEntry::SceneGraphPatchEntry(SceneManager& scene,
                                               std::string name,
                                               SceneGraphStateSnapshot before,
                                               SceneGraphStateSnapshot after)
        : scene_(scene),
          name_(std::move(name)),
          before_(std::move(before)),
          after_(std::move(after)) {}

    void SceneGraphPatchEntry::applyState(const SceneGraphStateSnapshot& desired,
                                          const SceneGraphStateSnapshot& current) {
        auto& scene = scene_.getScene();

        {
            lfs::core::Scene::Transaction txn(scene);

            std::set<std::string> names_to_remove;
            for (const auto& name : rootNames(desired))
                names_to_remove.insert(name);
            for (const auto& name : rootNames(current))
                names_to_remove.insert(name);

            for (const auto& name : names_to_remove) {
                if (scene.getNode(name)) {
                    scene.removeNode(name, false);
                }
            }

            for (const auto& root : desired.roots) {
                if (!restoreNodeSnapshot(scene_, root)) {
                    throw std::runtime_error("Failed to restore scene graph snapshot for '" + root.name + "'");
                }
            }
        }

        for (const auto& root : current.roots) {
            emitRemovedEvents(root);
        }
        for (const auto& root : desired.roots) {
            emitAddedEvents(scene_, root);
        }

        if (desired.context) {
            scene.setTrainingModelNode(desired.context->training_model_node_name);
            scene_.setDatasetPath(desired.context->dataset_path);
            scene_.changeContentType(static_cast<SceneManager::ContentType>(desired.context->content_type));

            if (scene.getNodeCount() == 0 &&
                desired.context->content_type == static_cast<int>(SceneManager::ContentType::Empty)) {
                python::set_application_scene(nullptr);
                SceneCleared{.from_history = true}.emit();
            } else {
                python::set_application_scene(&scene);
            }
        }

        if (desired.selected_node_names) {
            restoreNodeSelection(scene_, *desired.selected_node_names);
        }
    }

    void SceneGraphPatchEntry::undo() {
        applyState(before_, after_);
    }

    void SceneGraphPatchEntry::redo() {
        applyState(after_, before_);
    }

    size_t SceneGraphPatchEntry::estimatedBytes() const {
        size_t total = 0;
        for (const auto& root : before_.roots) {
            total += estimateSnapshotBytes(root);
        }
        for (const auto& root : after_.roots) {
            total += estimateSnapshotBytes(root);
        }
        return total;
    }

    UndoMemoryBreakdown SceneGraphPatchEntry::memoryBreakdown() const {
        UndoMemoryBreakdown total;
        for (const auto& root : before_.roots) {
            total += snapshotMemoryBreakdown(root);
        }
        for (const auto& root : after_.roots) {
            total += snapshotMemoryBreakdown(root);
        }
        if (before_.selected_node_names) {
            for (const auto& name : *before_.selected_node_names) {
                total.cpu_bytes += name.size();
            }
        }
        if (after_.selected_node_names) {
            for (const auto& name : *after_.selected_node_names) {
                total.cpu_bytes += name.size();
            }
        }
        if (before_.context) {
            total.cpu_bytes += before_.context->dataset_path.native().size() +
                               before_.context->training_model_node_name.size() + sizeof(int);
        }
        if (after_.context) {
            total.cpu_bytes += after_.context->dataset_path.native().size() +
                               after_.context->training_model_node_name.size() + sizeof(int);
        }
        return total;
    }

    void SceneGraphPatchEntry::offloadToCPU() {
        for (auto& root : before_.roots) {
            offloadSnapshot(root);
        }
        for (auto& root : after_.roots) {
            offloadSnapshot(root);
        }
    }

    void SceneGraphPatchEntry::restoreToPreferredDevice() {
        for (auto& root : before_.roots) {
            restoreSnapshot(root);
        }
        for (auto& root : after_.roots) {
            restoreSnapshot(root);
        }
    }

    UndoMetadata SceneGraphPatchEntry::metadata() const {
        return UndoMetadata{
            .id = "scene_graph.patch",
            .label = name_,
            .source = "core",
            .scope = "scene_graph",
        };
    }

    DirtyMask SceneGraphPatchEntry::dirtyFlags() const {
        DirtyMask flags = DirtyFlag::SPLATS | DirtyFlag::MESH | DirtyFlag::OVERLAY;
        if (before_.selected_node_names || after_.selected_node_names) {
            flags |= DirtyFlag::SELECTION;
        }
        if (before_.context || after_.context) {
            return DirtyFlag::ALL;
        }
        return flags;
    }

} // namespace lfs::vis::op
