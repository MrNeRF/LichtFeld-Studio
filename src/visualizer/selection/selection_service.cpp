/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_service.hpp"
#include "core/camera.hpp"
#include "core/cuda/selection_ops.hpp"
#include "core/logger.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cuda_event_pool.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "gui/gui_manager.hpp"
#include "internal/viewport.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "rendering/selection_ops.hpp"
#include "scene/scene_manager.hpp"
#include "selection_group_mask.hpp"
#include "training/training_manager.hpp"
#include "visualizer_impl.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <functional>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/trigonometric.hpp>
#include <limits>
#include <optional>
#include <shared_mutex>

namespace lfs::vis {

    namespace {
        constexpr float POLYGON_CLOSE_DISTANCE_PX = 12.0f;
        constexpr float POLYGON_CURSOR_APPEND_EPSILON_PX = 0.5f;
        constexpr float POLYGON_VERTEX_HIT_RADIUS_PX = 8.0f;
        constexpr float POLYGON_EDGE_HIT_RADIUS_PX = 10.0f;
        constexpr float INVALID_SCREEN_POSITION = -1.0e8f;
        constexpr float HOVER_PICK_RADIUS_PX = 12.0f;
        constexpr float RING_PICK_PADDING_PX = 4.0f;
        constexpr float MIN_VOLUME_SELECTION_RADIUS = 1.0e-3f;

        [[nodiscard]] glm::vec2 screenToRender(const glm::vec2& screen, const SelectionService::ViewportInfo& info) {
            const float scale_x = static_cast<float>(info.render_width) / info.width;
            const float scale_y = static_cast<float>(info.render_height) / info.height;
            return {
                (screen.x - info.x) * scale_x,
                (screen.y - info.y) * scale_y,
            };
        }

        [[nodiscard]] std::vector<std::pair<float, float>> screenPointsToRender(
            const std::vector<glm::vec2>& points, const SelectionService::ViewportInfo& info) {
            std::vector<std::pair<float, float>> render_points;
            render_points.reserve(points.size());
            for (const auto& point : points) {
                const auto render = screenToRender(point, info);
                render_points.emplace_back(render.x, render.y);
            }
            return render_points;
        }

        void hashCombine(std::size_t& seed, const std::size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
        }

        void hashFloat(std::size_t& seed, const float value) {
            std::uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            hashCombine(seed, std::hash<std::uint32_t>{}(bits));
        }

        void hashVec3(std::size_t& seed, const glm::vec3& value) {
            hashFloat(seed, value.x);
            hashFloat(seed, value.y);
            hashFloat(seed, value.z);
        }

        void hashMat3(std::size_t& seed, const glm::mat3& value) {
            const float* const ptr = glm::value_ptr(value);
            for (int i = 0; i < 9; ++i) {
                hashFloat(seed, ptr[i]);
            }
        }

        void hashMat4(std::size_t& seed, const glm::mat4& value) {
            const float* const ptr = glm::value_ptr(value);
            for (int i = 0; i < 16; ++i) {
                hashFloat(seed, ptr[i]);
            }
        }

        void hashTensorIdentity(std::size_t& seed, const core::Tensor* const tensor) {
            const bool valid = tensor && tensor->is_valid();
            hashCombine(seed, std::hash<bool>{}(valid));
            if (!valid) {
                return;
            }
            hashCombine(seed, tensor->debug_id());
            hashCombine(seed, reinterpret_cast<std::size_t>(tensor->data_ptr()));
            hashCombine(seed, tensor->bytes());
            hashCombine(seed, tensor->numel());
            hashCombine(seed, std::hash<int>{}(static_cast<int>(tensor->dtype())));
            hashCombine(seed, std::hash<int>{}(static_cast<int>(tensor->device())));
        }

        [[nodiscard]] std::size_t makeScreenPositionCacheSignature(
            const SceneRenderState& scene_state,
            const rendering::ViewportData& viewport,
            const bool equirectangular,
            const std::uint64_t projection_generation) {
            std::size_t seed = 0;
            hashCombine(seed, std::hash<std::uint64_t>{}(projection_generation));
            hashCombine(seed, reinterpret_cast<std::size_t>(scene_state.combined_model));
            const std::size_t model_count = scene_state.combined_model
                                                ? static_cast<std::size_t>(scene_state.combined_model->size())
                                                : 0u;
            hashCombine(seed, model_count);
            if (scene_state.combined_model) {
                hashTensorIdentity(seed, &scene_state.combined_model->means());
            }
            hashTensorIdentity(seed, scene_state.transform_indices.get());
            hashCombine(seed, scene_state.model_transforms.size());
            for (const auto& transform : scene_state.model_transforms) {
                hashMat4(seed, transform);
            }
            hashCombine(seed, scene_state.node_visibility_mask.size());
            for (const bool visible : scene_state.node_visibility_mask) {
                hashCombine(seed, std::hash<bool>{}(visible));
            }
            hashMat3(seed, viewport.rotation);
            hashVec3(seed, viewport.translation);
            hashCombine(seed, std::hash<int>{}(viewport.size.x));
            hashCombine(seed, std::hash<int>{}(viewport.size.y));
            hashFloat(seed, viewport.focal_length_mm);
            hashCombine(seed, std::hash<bool>{}(viewport.orthographic));
            hashFloat(seed, viewport.ortho_scale);
            hashCombine(seed, std::hash<bool>{}(equirectangular));
            return seed;
        }

        [[nodiscard]] core::Tensor& uploadFloat2PointsToBuffer(
            const std::vector<glm::vec2>& points,
            std::vector<float>& host_buffer,
            core::Tensor& device_buffer) {
            host_buffer.resize(points.size() * 2);
            for (size_t i = 0; i < points.size(); ++i) {
                host_buffer[i * 2] = points[i].x;
                host_buffer[i * 2 + 1] = points[i].y;
            }

            const bool needs_realloc = !device_buffer.is_valid() ||
                                       device_buffer.device() != core::Device::CUDA ||
                                       device_buffer.dtype() != core::DataType::Float32 ||
                                       device_buffer.shape().rank() != 2 ||
                                       device_buffer.size(0) != points.size() ||
                                       device_buffer.size(1) != 2;
            if (needs_realloc) {
                device_buffer = core::Tensor::empty({points.size(), size_t{2}},
                                                    core::Device::CUDA,
                                                    core::DataType::Float32);
            }

            auto host_view = core::Tensor::from_blob(host_buffer.data(),
                                                     {points.size(), size_t{2}},
                                                     core::Device::CPU,
                                                     core::DataType::Float32);
            device_buffer.copy_from(host_view);
            return device_buffer;
        }

        [[nodiscard]] std::optional<std::shared_lock<std::shared_mutex>> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager) {
            std::optional<std::shared_lock<std::shared_mutex>> lock;
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    lock.emplace(trainer->getRenderMutex());
                }
            }
            return lock;
        }

        [[nodiscard]] float distanceSquaredToSegment(const glm::vec2 point,
                                                     const glm::vec2 segment_start,
                                                     const glm::vec2 segment_end,
                                                     float* out_t = nullptr) {
            const glm::vec2 delta = segment_end - segment_start;
            const float length_sq = glm::dot(delta, delta);
            const float t =
                (length_sq > 0.0f)
                    ? glm::clamp(glm::dot(point - segment_start, delta) / length_sq, 0.0f, 1.0f)
                    : 0.0f;
            if (out_t) {
                *out_t = t;
            }

            const glm::vec2 closest = segment_start + delta * t;
            const glm::vec2 offset = point - closest;
            return glm::dot(offset, offset);
        }

        [[nodiscard]] core::Tensor ensureCudaBoolMask(const core::Tensor& mask) {
            auto result = (mask.dtype() == core::DataType::Bool) ? mask : mask.to(core::DataType::Bool);
            if (result.device() != core::Device::CUDA) {
                result = result.cuda();
            }
            return result;
        }

        [[nodiscard]] core::Tensor& ensureCudaByteScratchBuffer(core::Tensor& buffer, const size_t size) {
            const bool needs_realloc = !buffer.is_valid() ||
                                       buffer.device() != core::Device::CUDA ||
                                       buffer.dtype() != core::DataType::UInt8 ||
                                       buffer.numel() != size;
            if (needs_realloc) {
                buffer = core::Tensor::empty({size}, core::Device::CUDA, core::DataType::UInt8);
            }
            return buffer;
        }

        [[nodiscard]] core::Tensor& acquireSelectionOutputBuffer(std::array<core::Tensor, 2>& buffers,
                                                                 size_t& next_index,
                                                                 const size_t size) {
            auto& buffer = ensureCudaByteScratchBuffer(buffers[next_index], size);
            next_index = (next_index + 1) % buffers.size();
            return buffer;
        }

        [[nodiscard]] core::Scene::SelectionGroupCounts cachedSelectionGroupCounts(const core::Scene& scene) {
            core::Scene::SelectionGroupCounts counts{};
            for (const auto& group : scene.getSelectionGroups()) {
                counts[group.id] = group.count;
            }
            return counts;
        }

        [[nodiscard]] core::Scene::SelectionGroupCounts applySelectionGroupDeltas(
            core::Scene::SelectionGroupCounts counts,
            const std::array<int32_t, 256>& group_deltas) {
            for (size_t i = 1; i < counts.size(); ++i) {
                const int32_t delta = group_deltas[i];
                if (delta < 0) {
                    const size_t decrement = static_cast<size_t>(-delta);
                    counts[i] = decrement >= counts[i] ? 0 : counts[i] - decrement;
                } else if (delta > 0) {
                    const size_t increment = static_cast<size_t>(delta);
                    counts[i] = counts[i] > std::numeric_limits<size_t>::max() - increment
                                    ? std::numeric_limits<size_t>::max()
                                    : counts[i] + increment;
                }
            }
            return counts;
        }

        [[nodiscard]] size_t activeSelectionGaussianCount(const SceneManager* const scene_manager) {
            if (!scene_manager) {
                return 0;
            }
            if (const auto* const model = scene_manager->getModelForRendering()) {
                return static_cast<size_t>(model->size());
            }
            return scene_manager->getScene().getTotalGaussianCount();
        }

        [[nodiscard]] const core::Tensor* selectionMaskForSize(
            const std::shared_ptr<core::Tensor>& mask,
            const size_t expected_size) {
            if (!mask || !mask->is_valid() || mask->numel() != expected_size) {
                return nullptr;
            }
            return mask.get();
        }

        [[nodiscard]] const core::Tensor* selectionMaskForSize(
            const core::Tensor* const mask,
            const size_t expected_size) {
            if (!mask || !mask->is_valid() || mask->numel() != expected_size) {
                return nullptr;
            }
            return mask;
        }

        [[nodiscard]] bool nodeMaskRestrictsSelection(const std::vector<bool>& node_mask) {
            return std::any_of(node_mask.begin(), node_mask.end(), [](const bool enabled) { return !enabled; });
        }

        [[nodiscard]] bool copySelectionIfSameSize(const core::Tensor& source, core::Tensor& output) {
            if (!source.is_valid() || !output.is_valid() || source.numel() != output.numel()) {
                return false;
            }
            if (source.device() == core::Device::CUDA &&
                output.device() == core::Device::CUDA &&
                source.dtype() == output.dtype() &&
                source.is_contiguous() &&
                output.is_contiguous()) {
                const cudaStream_t source_stream = source.stream();
                const cudaStream_t output_stream = output.stream();

                // Pooled event edges both ways: copy on the source stream after
                // the output's pending work, then hand the result back to the
                // output stream. record_stream keeps the allocator from
                // recycling the output before the cross-stream write retires.
                lfs::core::bridgeStreams(output_stream, source_stream);

                if (const cudaError_t status = cudaMemcpyAsync(output.data_ptr(),
                                                               source.data_ptr(),
                                                               source.bytes(),
                                                               cudaMemcpyDeviceToDevice,
                                                               source_stream);
                    status != cudaSuccess) {
                    LOG_WARN("SelectionService: async selection copy failed: {} ({})",
                             cudaGetErrorName(status),
                             cudaGetErrorString(status));
                    return false;
                }
                output.record_stream(source_stream);

                lfs::core::bridgeStreams(source_stream, output_stream);
                return true;
            }
            output.copy_from(source);
            return true;
        }

        [[nodiscard]] bool selectionMaskHasAny(const core::Tensor& selection) {
            return selection.is_valid() && selection.numel() > 0 && selection.count_nonzero() > 0;
        }

        [[nodiscard]] core::Tensor visibleNodeScopeMask(
            lfs::core::Scene& scene,
            const size_t visible_count,
            const std::vector<bool>& node_mask) {
            auto scope = core::Tensor::ones({visible_count}, core::Device::CUDA, core::DataType::Bool);
            if (!nodeMaskRestrictsSelection(node_mask)) {
                return scope;
            }

            const auto transform_indices = scene.getTransformIndices();
            if (!transform_indices || !transform_indices->is_valid() ||
                transform_indices->numel() != visible_count) {
                return {};
            }

            rendering::filter_selection_by_node_mask(scope, *transform_indices, node_mask);
            return scope;
        }

        [[nodiscard]] core::Tensor expandSelectionToSceneMask(
            SceneManager* const scene_manager,
            const core::Tensor& selection,
            const SelectionMode mode,
            const uint8_t group_id,
            const core::Tensor* existing_mask,
            const std::vector<bool>& node_mask) {
            if (!scene_manager || !selection.is_valid()) {
                return {};
            }

            auto& scene = scene_manager->getScene();
            const size_t full_count = scene.getSelectionGaussianCount();
            const bool preserves_active_group =
                mode == SelectionMode::Replace &&
                existing_mask && existing_mask->is_valid() &&
                existing_mask->numel() == full_count;
            const bool scoped_replace = preserves_active_group && nodeMaskRestrictsSelection(node_mask);

            if (selection.numel() == full_count) {
                if (scoped_replace) {
                    const size_t visible_count = activeSelectionGaussianCount(scene_manager);
                    if (visible_count == full_count) {
                        auto scope = visibleNodeScopeMask(scene, visible_count, node_mask);
                        if (!scope.is_valid()) {
                            return {};
                        }
                        auto active_group = existing_mask->eq(group_id);
                        if (active_group.device() != core::Device::CUDA) {
                            active_group = active_group.cuda();
                        }
                        return selection.where(scope, active_group);
                    }
                }
                return selection;
            }

            const size_t visible_count = activeSelectionGaussianCount(scene_manager);
            if (selection.numel() != visible_count || full_count == 0) {
                return {};
            }

            const auto visible_indices = scene.getVisibleSelectionIndices();
            if (!visible_indices || !visible_indices->is_valid() ||
                visible_indices->numel() != visible_count) {
                return {};
            }

            core::Tensor expanded;
            if (preserves_active_group) {
                expanded = existing_mask->eq(group_id);
                if (expanded.device() != core::Device::CUDA) {
                    expanded = expanded.cuda();
                }
            } else {
                expanded = core::Tensor::zeros({full_count}, core::Device::CUDA, core::DataType::Bool);
            }

            const core::Tensor* visible_selection = &selection;
            core::Tensor scoped_selection;
            if (scoped_replace) {
                auto scope = visibleNodeScopeMask(scene, visible_count, node_mask);
                if (!scope.is_valid()) {
                    return {};
                }
                const auto active_group_visible = expanded.index_select(0, *visible_indices).contiguous();
                scoped_selection = selection.where(scope, active_group_visible);
                visible_selection = &scoped_selection;
            }

            expanded.index_copy_(0, *visible_indices, *visible_selection);
            return expanded;
        }

        [[nodiscard]] core::Tensor intersectWithActiveSelection(
            SceneManager* const scene_manager,
            const core::Tensor& selection,
            const core::Tensor* const existing_mask,
            const uint8_t group_id) {
            if (!scene_manager || !selection.is_valid()) {
                return {};
            }

            const size_t selection_count = selection.numel();
            if (!existing_mask || !existing_mask->is_valid()) {
                return core::Tensor::zeros({selection_count}, core::Device::CUDA, core::DataType::Bool);
            }

            auto& scene = scene_manager->getScene();
            const size_t full_count = scene.getSelectionGaussianCount();
            auto active_group = existing_mask->eq(group_id);
            if (active_group.device() != core::Device::CUDA) {
                active_group = active_group.cuda();
            }

            if (selection_count == full_count) {
                return selection.logical_and(active_group);
            }

            const size_t visible_count = activeSelectionGaussianCount(scene_manager);
            const auto visible_indices = scene.getVisibleSelectionIndices();
            if (selection_count != visible_count ||
                !visible_indices ||
                !visible_indices->is_valid() ||
                visible_indices->numel() != visible_count) {
                return {};
            }

            const auto active_visible = active_group.index_select(0, *visible_indices).contiguous();
            return selection.logical_and(active_visible);
        }

        [[nodiscard]] rendering::ViewportData viewportDataFromCamera(const core::Camera& camera) {
            const auto rotation_cpu = camera.R().cpu().to(core::DataType::Float32);
            const auto position_cpu = camera.cam_position().cpu().to(core::DataType::Float32);
            const float* const rotation = rotation_cpu.ptr<float>();
            const float* const position = position_cpu.ptr<float>();

            glm::mat4 scene_transform(1.0f);
            if (auto* const scene_manager = services().sceneOrNull()) {
                if (const auto transform =
                        scene_manager->getScene().getCameraSceneTransformByUid(camera.uid())) {
                    scene_transform = rendering::dataWorldTransformToVisualizerWorld(*transform);
                }
            }

            const auto pose = rendering::visualizerCameraPoseFromDataCameraToWorld(
                glm::transpose(rendering::mat3FromRowMajor3x3(rotation)),
                glm::vec3(position[0], position[1], position[2]),
                scene_transform);

            const int width = std::max(camera.image_width(), camera.camera_width());
            const int height = std::max(camera.image_height(), camera.camera_height());

            return rendering::ViewportData{
                .rotation = pose.rotation,
                .translation = pose.translation,
                .size = glm::ivec2(width, height),
                .focal_length_mm = rendering::vFovToFocalLength(glm::degrees(camera.FoVy())),
                .orthographic = false,
                .ortho_scale = 1.0f,
            };
        }

        [[nodiscard]] core::Tensor uploadModelTransformsToCuda(const std::vector<glm::mat4>& model_transforms) {
            std::vector<float> transform_data(model_transforms.size() * 16);
            for (size_t i = 0; i < model_transforms.size(); ++i) {
                const auto& transform = model_transforms[i];
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        transform_data[i * 16 + row * 4 + col] = transform[col][row];
                    }
                }
            }
            return core::Tensor::from_vector(
                       transform_data,
                       {model_transforms.size(), size_t{4}, size_t{4}},
                       core::Device::CPU)
                .cuda();
        }

        [[nodiscard]] rendering::ViewportData viewportDataFromViewer(
            const Viewport& viewport,
            const SelectionService::ViewportInfo& info,
            const RenderSettings& settings) {
            return rendering::ViewportData{
                .rotation = viewport.camera.R,
                .translation = viewport.camera.t,
                .size = glm::ivec2(info.render_width, info.render_height),
                .focal_length_mm = settings.focal_length_mm,
                .orthographic = settings.orthographic,
                .ortho_scale = settings.ortho_scale,
            };
        }

        [[nodiscard]] rendering::FrameView frameViewFromViewport(const rendering::ViewportData& viewport,
                                                                 const glm::vec3& background_color,
                                                                 const float far_plane = rendering::DEFAULT_FAR_PLANE) {
            return rendering::FrameView{
                .rotation = viewport.rotation,
                .translation = viewport.translation,
                .size = viewport.size,
                .focal_length_mm = viewport.focal_length_mm,
                .intrinsics_override = std::nullopt,
                .far_plane = far_plane,
                .orthographic = viewport.orthographic,
                .ortho_scale = viewport.ortho_scale,
                .background_color = background_color,
            };
        }

        template <typename RenderableT>
        [[nodiscard]] const RenderableT* findRenderableByNodeId(const std::vector<RenderableT>& items,
                                                                const core::NodeId node_id) {
            if (node_id == core::NULL_NODE) {
                return nullptr;
            }
            const auto it = std::find_if(items.begin(), items.end(),
                                         [node_id](const auto& item) { return item.node_id == node_id; });
            return (it == items.end()) ? nullptr : &(*it);
        }

        [[nodiscard]] std::optional<core::Tensor> tryBuildVksplatSelectionMask(
            SceneManager* const scene_manager,
            RenderingManager* const rendering_manager,
            const rendering::FrameView& frame_view,
            const bool equirectangular,
            const RenderingManager::VksplatSelectionMaskShape shape,
            const std::vector<glm::vec4>& primitives,
            std::uint32_t* const picked_ring_id_out = nullptr) {
            if (!scene_manager || !rendering_manager || primitives.empty()) {
                return std::nullopt;
            }

            auto result = rendering_manager->buildVksplatSelectionMask(
                *scene_manager, frame_view, equirectangular, shape, primitives, {}, picked_ring_id_out);
            if (result) {
                return std::move(*result);
            }

            const auto settings = rendering_manager->getSettings();
            if (lfs::rendering::isVkSplatBackend(settings.raster_backend)) {
                LOG_DEBUG("SelectionService: VkSplat selection query unavailable, falling back to screen-position path: {}",
                          result.error());
            }
            return std::nullopt;
        }

        [[nodiscard]] std::optional<core::Tensor> tryBuildVksplatPolygonSelectionMask(
            SceneManager* const scene_manager,
            RenderingManager* const rendering_manager,
            const rendering::FrameView& frame_view,
            const bool equirectangular,
            const std::vector<glm::vec2>& polygon_vertices) {
            if (!scene_manager || !rendering_manager || polygon_vertices.size() < 3) {
                return std::nullopt;
            }

            auto result = rendering_manager->buildVksplatSelectionMask(
                *scene_manager,
                frame_view,
                equirectangular,
                RenderingManager::VksplatSelectionMaskShape::Polygon,
                {},
                polygon_vertices);
            if (result) {
                return std::move(*result);
            }

            const auto settings = rendering_manager->getSettings();
            if (lfs::rendering::isVkSplatBackend(settings.raster_backend)) {
                LOG_DEBUG("SelectionService: VkSplat polygon selection query unavailable, falling back to screen-position path: {}",
                          result.error());
            }
            return std::nullopt;
        }

        [[nodiscard]] std::vector<glm::vec4> buildBrushPrimitives(
            const std::vector<glm::vec2>& points,
            const float radius,
            const SelectionService::ViewportInfo& info) {
            std::vector<glm::vec4> primitives;
            if (points.empty() || !info.valid()) {
                return primitives;
            }

            const float scale_x = static_cast<float>(info.render_width) / info.width;
            const float scaled_radius = radius * scale_x;
            constexpr float STEP_FACTOR = 0.5f;
            constexpr int MAX_BRUSH_STEPS = 128;

            for (size_t i = 0; i < points.size(); ++i) {
                const glm::vec2 from = (i == 0) ? points[i] : points[i - 1];
                const glm::vec2 to = points[i];
                const glm::vec2 delta = to - from;
                const float step_spacing = std::max(radius * STEP_FACTOR, 1.0f);
                const int num_steps = std::min(
                    MAX_BRUSH_STEPS,
                    std::max(1, static_cast<int>(std::ceil(glm::length(delta) / step_spacing))));
                for (int step = 0; step < num_steps; ++step) {
                    const float t = (num_steps == 1) ? 1.0f
                                                     : static_cast<float>(step + 1) / static_cast<float>(num_steps);
                    const glm::vec2 sample = from + delta * t;
                    const auto render = screenToRender(sample, info);
                    primitives.emplace_back(render.x, render.y, scaled_radius * scaled_radius, 0.0f);
                }
            }

            return primitives;
        }

        [[nodiscard]] std::shared_ptr<core::Tensor> projectGaussianScreenPositions(
            const core::SplatData& model,
            const rendering::ViewportData& viewport,
            const bool equirectangular,
            const rendering::GaussianSceneState& scene) {
            if (equirectangular || viewport.size.x <= 0 || viewport.size.y <= 0 ||
                (viewport.orthographic && viewport.ortho_scale <= 0.0f)) {
                return nullptr;
            }

            const size_t count = static_cast<size_t>(model.size());
            if (count == 0) {
                return nullptr;
            }

            try {
                auto means = model.get_means();
                if (!means.is_valid() || means.ndim() != 2 || means.size(0) != count || means.size(1) != 3) {
                    return nullptr;
                }
                if (means.dtype() != core::DataType::Float32) {
                    means = means.to(core::DataType::Float32);
                }
                if (means.device() == core::Device::CUDA) {
                    try {
                        core::Tensor model_transforms_cuda;
                        const core::Tensor* model_transforms_ptr = nullptr;
                        if (scene.model_transforms && !scene.model_transforms->empty()) {
                            model_transforms_cuda = uploadModelTransformsToCuda(*scene.model_transforms);
                            model_transforms_ptr = &model_transforms_cuda;
                        }

                        core::Tensor transform_indices_cuda;
                        const core::Tensor* transform_indices_ptr = nullptr;
                        if (scene.transform_indices && scene.transform_indices->is_valid() &&
                            scene.transform_indices->numel() >= count) {
                            transform_indices_cuda = *scene.transform_indices;
                            if (transform_indices_cuda.dtype() != core::DataType::Int32) {
                                transform_indices_cuda = transform_indices_cuda.to(core::DataType::Int32);
                            }
                            if (transform_indices_cuda.device() != core::Device::CUDA) {
                                transform_indices_cuda = transform_indices_cuda.cuda();
                            }
                            if (!transform_indices_cuda.is_contiguous()) {
                                transform_indices_cuda = transform_indices_cuda.contiguous();
                            }
                            transform_indices_ptr = &transform_indices_cuda;
                        }

                        const auto [fx, fy] =
                            rendering::computePixelFocalLengths(viewport.size, viewport.focal_length_mm);
                        const std::array<float, 9> view_rotation_rows{
                            viewport.rotation[0].x,
                            viewport.rotation[0].y,
                            viewport.rotation[0].z,
                            viewport.rotation[1].x,
                            viewport.rotation[1].y,
                            viewport.rotation[1].z,
                            viewport.rotation[2].x,
                            viewport.rotation[2].y,
                            viewport.rotation[2].z,
                        };
                        const std::array<float, 3> translation{
                            viewport.translation.x,
                            viewport.translation.y,
                            viewport.translation.z,
                        };

                        return std::make_shared<core::Tensor>(
                            rendering::project_screen_positions_tensor(
                                means,
                                viewport.size.x,
                                viewport.size.y,
                                view_rotation_rows,
                                translation,
                                fx,
                                fy,
                                viewport.orthographic,
                                viewport.ortho_scale,
                                model_transforms_ptr,
                                transform_indices_ptr,
                                scene.node_visibility_mask));
                    } catch (const std::exception& e) {
                        LOG_DEBUG("SelectionService: CUDA screen-position projection unavailable, falling back to CPU: {}",
                                  e.what());
                    }
                }
                means = means.cpu().contiguous();
                const float* const means_ptr = means.ptr<float>();
                if (!means_ptr) {
                    return nullptr;
                }

                core::Tensor transform_indices_cpu;
                const std::int32_t* transform_indices = nullptr;
                if (scene.transform_indices && scene.transform_indices->is_valid() &&
                    scene.transform_indices->numel() >= count) {
                    transform_indices_cpu = *scene.transform_indices;
                    if (transform_indices_cpu.dtype() != core::DataType::Int32) {
                        transform_indices_cpu = transform_indices_cpu.to(core::DataType::Int32);
                    }
                    transform_indices_cpu = transform_indices_cpu.cpu().contiguous();
                    transform_indices = transform_indices_cpu.ptr<std::int32_t>();
                }

                static const std::vector<glm::mat4> empty_transforms;
                const std::vector<glm::mat4>* const transforms_ptr = scene.model_transforms;
                const std::vector<glm::mat4>& transforms = transforms_ptr ? *transforms_ptr : empty_transforms;

                std::vector<float> positions(count * 2, INVALID_SCREEN_POSITION);
                for (size_t i = 0; i < count; ++i) {
                    const int transform_index = transform_indices ? transform_indices[i] : 0;
                    if (!scene.node_visibility_mask.empty() && transform_indices) {
                        if (transform_index < 0 ||
                            static_cast<size_t>(transform_index) >= scene.node_visibility_mask.size() ||
                            !scene.node_visibility_mask[static_cast<size_t>(transform_index)]) {
                            continue;
                        }
                    }

                    glm::vec3 world_point(
                        means_ptr[i * 3 + 0],
                        means_ptr[i * 3 + 1],
                        means_ptr[i * 3 + 2]);
                    if (!transforms.empty()) {
                        const int clamped_index =
                            std::clamp(transform_index, 0, static_cast<int>(transforms.size()) - 1);
                        world_point = glm::vec3(
                            transforms[static_cast<size_t>(clamped_index)] * glm::vec4(world_point, 1.0f));
                    }

                    const auto projected = rendering::projectWorldPoint(
                        viewport.rotation,
                        viewport.translation,
                        viewport.size,
                        world_point,
                        viewport.focal_length_mm,
                        viewport.orthographic,
                        viewport.ortho_scale);
                    if (!projected || !std::isfinite(projected->x) || !std::isfinite(projected->y)) {
                        continue;
                    }
                    positions[i * 2] = projected->x;
                    positions[i * 2 + 1] = projected->y;
                }

                return std::make_shared<core::Tensor>(
                    core::Tensor::from_vector(
                        positions,
                        {count, size_t{2}},
                        core::Device::CPU)
                        .cuda()
                        .contiguous());
            } catch (const std::exception& e) {
                LOG_WARN("SelectionService: failed to project Gaussian screen positions: {}", e.what());
                return nullptr;
            }
        }

        [[nodiscard]] std::optional<int> pickProjectedGaussian(
            const core::Tensor& screen_positions,
            const glm::vec2 cursor_pos,
            const float radius_px = HOVER_PICK_RADIUS_PX) {
            if (!screen_positions.is_valid() || screen_positions.ndim() != 2 ||
                screen_positions.size(1) != 2) {
                return std::nullopt;
            }

            if (screen_positions.device() == core::Device::CUDA &&
                screen_positions.dtype() == core::DataType::Float32) {
                try {
                    const int picked = rendering::pick_projected_gaussian_tensor(
                        screen_positions, cursor_pos.x, cursor_pos.y, radius_px);
                    return picked >= 0 ? std::optional<int>{picked} : std::nullopt;
                } catch (const std::exception& e) {
                    LOG_DEBUG("SelectionService: CUDA projected pick unavailable, falling back to CPU scan: {}",
                              e.what());
                }
            }

            auto cpu = screen_positions;
            if (cpu.dtype() != core::DataType::Float32) {
                cpu = cpu.to(core::DataType::Float32);
            }
            cpu = cpu.cpu().contiguous();
            const float* const data = cpu.ptr<float>();
            if (!data) {
                return std::nullopt;
            }

            const float max_dist_sq = radius_px * radius_px;
            float best_dist_sq = max_dist_sq;
            int best_index = -1;
            const size_t count = cpu.size(0);
            for (size_t i = 0; i < count; ++i) {
                const glm::vec2 pos(data[i * 2], data[i * 2 + 1]);
                if (pos.x < INVALID_SCREEN_POSITION * 0.5f ||
                    pos.y < INVALID_SCREEN_POSITION * 0.5f ||
                    !std::isfinite(pos.x) || !std::isfinite(pos.y)) {
                    continue;
                }
                const glm::vec2 delta = pos - cursor_pos;
                const float dist_sq = glm::dot(delta, delta);
                if (dist_sq <= best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_index = static_cast<int>(i);
                }
            }
            return best_index >= 0 ? std::optional<int>{best_index} : std::nullopt;
        }

    } // namespace

    SelectionService::SelectionService(SceneManager* scene_manager, RenderingManager* rendering_manager)
        : scene_manager_(scene_manager),
          rendering_manager_(rendering_manager) {
        assert(scene_manager_);
        assert(rendering_manager_);
    }

    SelectionService::~SelectionService() = default;

    SelectionResult SelectionService::selectBrush(float x, float y, float radius, SelectionMode mode,
                                                  int camera_index) {
        LOG_TIMER("SelectionService::selectBrush");
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }
        const auto filters = defaultFilterState();
        const auto settings = rendering_manager_->getSettings();
        const std::vector<glm::vec4> primitives{{x, y, radius * radius, 0.0f}};
        if (const auto frame_view = resolveCommandFrameView(camera_index)) {
            if (auto selection = tryBuildVksplatSelectionMask(
                    scene_manager_, rendering_manager_, *frame_view, settings.equirectangular,
                    RenderingManager::VksplatSelectionMaskShape::Brush, primitives)) {
                return commitSelection(*selection, mode, effectiveNodeMask(true), filters, "selection.brush");
            }
        }

        const auto screen_positions = resolveCommandScreenPositions(camera_index);
        if (!screen_positions || !screen_positions->is_valid()) {
            return {false, 0, "No screen positions"};
        }

        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, screen_positions->size(0));
        rendering::brush_select_tensor(*screen_positions, x, y, radius, selection);
        return commitSelection(selection, mode, effectiveNodeMask(true), filters, "selection.brush");
    }

    SelectionResult SelectionService::selectRect(float x0, float y0, float x1, float y1, SelectionMode mode,
                                                 int camera_index) {
        LOG_TIMER("SelectionService::selectRect");
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }
        const auto filters = defaultFilterState();
        const auto settings = rendering_manager_->getSettings();
        const std::vector<glm::vec4> primitives{{
            std::min(x0, x1),
            std::min(y0, y1),
            std::max(x0, x1),
            std::max(y0, y1),
        }};
        if (const auto frame_view = resolveCommandFrameView(camera_index)) {
            if (auto selection = tryBuildVksplatSelectionMask(
                    scene_manager_, rendering_manager_, *frame_view, settings.equirectangular,
                    RenderingManager::VksplatSelectionMaskShape::Rectangle, primitives)) {
                return commitSelection(*selection, mode, effectiveNodeMask(true), filters, "selection.rect");
            }
        }

        const auto screen_positions = resolveCommandScreenPositions(camera_index);
        if (!screen_positions || !screen_positions->is_valid()) {
            return {false, 0, "No screen positions"};
        }

        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, screen_positions->size(0));
        rendering::rect_select_tensor(*screen_positions,
                                      std::min(x0, x1),
                                      std::min(y0, y1),
                                      std::max(x0, x1),
                                      std::max(y0, y1),
                                      selection);
        return commitSelection(selection, mode, effectiveNodeMask(true), filters, "selection.rect");
    }

    std::optional<rendering::FrameView> SelectionService::resolveCommandFrameView(int camera_index) const {
        if (!scene_manager_ || !rendering_manager_) {
            return std::nullopt;
        }
        const auto settings = rendering_manager_->getSettings();
        if (camera_index >= 0) {
            const auto cameras = scene_manager_->getScene().getAllCameras();
            if (camera_index < static_cast<int>(cameras.size()) && cameras[camera_index]) {
                return frameViewFromViewport(
                    viewportDataFromCamera(*cameras[camera_index]),
                    settings.background_color);
            }
            return std::nullopt;
        }
        const auto context = resolveViewerViewportContext();
        if (!context || !context->valid()) {
            return std::nullopt;
        }
        Viewport projection_viewport = *context->viewport;
        projection_viewport.windowSize = {context->info.render_width, context->info.render_height};
        return frameViewFromViewport(
            viewportDataFromViewer(projection_viewport, context->info, settings),
            settings.background_color,
            settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
    }

    SelectionResult SelectionService::selectPolygon(const std::vector<glm::vec2>& vertices,
                                                    SelectionMode mode, int camera_index) {
        LOG_TIMER("SelectionService::selectPolygon");
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }
        if (vertices.size() < 3) {
            return {false, 0, "Polygon requires at least 3 vertices"};
        }

        const auto filters = defaultFilterState();
        const auto settings = rendering_manager_->getSettings();

        if (const auto frame_view = resolveCommandFrameView(camera_index)) {
            if (auto selection = tryBuildVksplatPolygonSelectionMask(
                    scene_manager_, rendering_manager_, *frame_view, settings.equirectangular, vertices)) {
                return commitSelection(*selection, mode, effectiveNodeMask(true), filters, "selection.polygon");
            }
        }

        const auto screen_positions = resolveCommandScreenPositions(camera_index);
        if (!screen_positions || !screen_positions->is_valid()) {
            return {false, 0, "No screen positions"};
        }

        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, screen_positions->size(0));
        auto& polygon = uploadFloat2PointsToBuffer(vertices, polygon_vertex_host_buffer_, polygon_vertex_device_buffer_);
        rendering::polygon_select_tensor(*screen_positions, polygon, selection);
        return commitSelection(selection, mode, effectiveNodeMask(true), filters, "selection.polygon");
    }

    SelectionResult SelectionService::selectLasso(const std::vector<glm::vec2>& vertices,
                                                  const SelectionMode mode, const int camera_index) {
        LOG_TIMER("SelectionService::selectLasso");
        auto result = selectPolygon(vertices, mode, camera_index);
        if (result.success) {
            return result;
        }
        if (result.error == "Polygon requires at least 3 vertices") {
            result.error = "Lasso requires at least 3 points";
        }
        return result;
    }

    SelectionResult SelectionService::selectRing(const float x, const float y, const SelectionMode mode,
                                                 const int camera_index) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto filters = defaultFilterState();
        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            return {false, 0, "No gaussians"};
        }

        const auto settings = rendering_manager_->getSettings();
        std::optional<rendering::FrameView> frame_view;
        glm::vec2 query_point{x, y};
        float query_padding = RING_PICK_PADDING_PX;
        if (camera_index >= 0) {
            frame_view = resolveCommandFrameView(camera_index);
        } else if (const auto context = resolveViewerViewportContext(glm::vec2{x, y});
                   context && context->valid()) {
            Viewport projection_viewport = *context->viewport;
            projection_viewport.windowSize = {context->info.render_width, context->info.render_height};
            frame_view = frameViewFromViewport(
                viewportDataFromViewer(projection_viewport, context->info, settings),
                settings.background_color,
                settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
            query_point = screenToRender(glm::vec2{x, y}, context->info);
            query_padding = RING_PICK_PADDING_PX *
                            (static_cast<float>(context->info.render_width) / context->info.width);
        }
        const std::vector<glm::vec4> primitives{{query_point.x, query_point.y, query_padding, 0.0f}};
        if (frame_view) {
            std::uint32_t picked_ring_id = std::numeric_limits<std::uint32_t>::max();
            if (auto selection = tryBuildVksplatSelectionMask(
                    scene_manager_, rendering_manager_, *frame_view, settings.equirectangular,
                    RenderingManager::VksplatSelectionMaskShape::Ring, primitives, &picked_ring_id);
                selection) {
                if (selectionMaskHasAny(*selection)) {
                    return commitSelection(*selection, mode, effectiveNodeMask(true), filters, "selection.ring");
                }
                return {false, 0, "No hovered gaussian"};
            }
        }

        const auto hovered_id = resolveCommandHoveredGaussianId(x, y, camera_index, filters);
        if (hovered_id && *hovered_id >= 0 && static_cast<size_t>(*hovered_id) < total) {
            auto& selection = resetBoolScratchBuffer(command_selection_buffer_, total);
            rendering::set_selection_element(selection.ptr<bool>(), *hovered_id, true);
            return commitSelection(selection, mode, effectiveNodeMask(true), filters, "selection.ring");
        }

        return {false, 0, "No hovered gaussian"};
    }

    SelectionResult SelectionService::selectByColorAt(const float x, const float y, const SelectionMode mode,
                                                      const SelectionFilterState filters,
                                                      const int camera_index) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto hovered_id = resolveCommandHoveredGaussianId(x, y, camera_index, filters);
        if (!hovered_id) {
            return {false, 0, "No hovered gaussian"};
        }

        auto& scene = scene_manager_->getScene();
        auto* const model = scene.getCombinedModel();
        if (!model) {
            return {false, 0, "No model"};
        }

        const auto& sh0 = model->sh0();
        if (!sh0.is_valid() || *hovered_id < 0 || static_cast<size_t>(*hovered_id) >= sh0.size(0)) {
            return {false, 0, "Invalid color reference"};
        }

        auto sh0_cpu = sh0.cpu();
        const float* const sh0_data = sh0_cpu.ptr<float>();
        if (!sh0_data) {
            return {false, 0, "Invalid color data"};
        }

        constexpr float SH_C0 = 0.28209479177387814f;
        const size_t ref_offset = static_cast<size_t>(*hovered_id) * 3;
        const float ref_r = std::clamp(0.5f + sh0_data[ref_offset] * SH_C0, 0.0f, 1.0f);
        const float ref_g = std::clamp(0.5f + sh0_data[ref_offset + 1] * SH_C0, 0.0f, 1.0f);
        const float ref_b = std::clamp(0.5f + sh0_data[ref_offset + 2] * SH_C0, 0.0f, 1.0f);

        constexpr float COLOR_THRESHOLD = 0.2f;
        const auto group_id = scene.getActiveSelectionGroup();
        auto mask = core::cuda::select_by_color(sh0, ref_r, ref_g, ref_b, COLOR_THRESHOLD, group_id);

        return commitSelection(mask,
                               mode,
                               effectiveNodeMask(filters.restrict_to_selected_nodes),
                               filters,
                               "selection.by_color");
    }

    SelectionResult SelectionService::selectBoxVolume(const SelectionMode mode,
                                                      const SelectionCommitOptions options) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto gizmo = rendering_manager_->getGizmoState();
        if (!gizmo.cropbox_active) {
            return {false, 0, "No active box selection volume"};
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            return {true, 0, {}};
        }

        const glm::mat4 world_to_box = glm::inverse(gizmo.cropbox_transform);
        const float* const transform_ptr = glm::value_ptr(world_to_box);
        const auto crop_t = core::Tensor::from_vector(std::vector<float>(transform_ptr, transform_ptr + 16), {4, 4});
        const auto crop_min =
            core::Tensor::from_vector({gizmo.cropbox_min.x, gizmo.cropbox_min.y, gizmo.cropbox_min.z}, {3});
        const auto crop_max =
            core::Tensor::from_vector({gizmo.cropbox_max.x, gizmo.cropbox_max.y, gizmo.cropbox_max.z}, {3});

        auto selection = core::Tensor::ones({total}, core::Device::CUDA, core::DataType::Bool);
        applyCropFilter(selection, &crop_t, &crop_min, &crop_max, nullptr, nullptr, false);

        auto filters = defaultFilterState();
        filters.crop_filter = false;
        return commitSelection(selection,
                               mode,
                               effectiveNodeMask(filters.restrict_to_selected_nodes),
                               filters,
                               "selection.box",
                               options);
    }

    SelectionResult SelectionService::selectSphereVolume(const SelectionMode mode,
                                                         const SelectionCommitOptions options) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const auto gizmo = rendering_manager_->getGizmoState();
        if (!gizmo.ellipsoid_active) {
            return {false, 0, "No active sphere selection volume"};
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            return {true, 0, {}};
        }

        const glm::mat4 world_to_ellipsoid = glm::inverse(gizmo.ellipsoid_transform);
        const float* const transform_ptr = glm::value_ptr(world_to_ellipsoid);
        const auto ellip_t =
            core::Tensor::from_vector(std::vector<float>(transform_ptr, transform_ptr + 16), {4, 4});
        const auto ellip_radii = core::Tensor::from_vector(
            {gizmo.ellipsoid_radii.x, gizmo.ellipsoid_radii.y, gizmo.ellipsoid_radii.z}, {3});

        auto selection = core::Tensor::ones({total}, core::Device::CUDA, core::DataType::Bool);
        applyCropFilter(selection, nullptr, nullptr, nullptr, &ellip_t, &ellip_radii, false);

        auto filters = defaultFilterState();
        filters.crop_filter = false;
        return commitSelection(selection,
                               mode,
                               effectiveNodeMask(filters.restrict_to_selected_nodes),
                               filters,
                               "selection.sphere",
                               options);
    }

    SelectionResult SelectionService::selectAllFiltered() {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            return {true, 0, {}};
        }

        const auto filters = defaultFilterState();
        auto selection = core::Tensor::ones({total}, core::Device::CUDA, core::DataType::Bool);
        return commitSelection(selection,
                               SelectionMode::Replace,
                               effectiveNodeMask(filters.restrict_to_selected_nodes),
                               filters,
                               "selection.all.filtered");
    }

    SelectionResult SelectionService::invertFiltered() {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            return {true, 0, {}};
        }

        const auto filters = defaultFilterState();
        const auto node_mask = effectiveNodeMask(filters.restrict_to_selected_nodes);
        auto filter_mask = core::Tensor::ones({total}, core::Device::CUDA, core::DataType::Bool);
        applyFilters(filter_mask, filters, node_mask);

        const auto& scene = scene_manager_->getScene();
        const uint8_t group_id = scene.getActiveSelectionGroup();
        const auto existing_mask = scene.getVisibleSelectionMask();
        const auto* existing = selectionMaskForSize(existing_mask, total);
        const auto current_active = existing
                                        ? existing->eq(group_id)
                                        : core::Tensor::zeros({total}, core::Device::CUDA, core::DataType::Bool);
        const auto any_selected = existing
                                      ? existing->gt(0.0f)
                                      : core::Tensor::zeros({total}, core::Device::CUDA, core::DataType::Bool);
        const auto other_selected = any_selected.logical_and(current_active.logical_not());
        const auto toggle_mask = filter_mask.logical_and(other_selected.logical_not());
        const auto inverted = current_active.logical_xor(toggle_mask);

        return commitSelection(
            inverted, SelectionMode::Replace, {}, SelectionFilterState{}, "selection.invert.filtered");
    }

    SelectionResult SelectionService::applyMask(const std::vector<uint8_t>& mask, SelectionMode mode) {
        if (!scene_manager_) {
            return {false, 0, "Missing scene manager"};
        }

        const size_t visible_total = activeSelectionGaussianCount(scene_manager_);
        const size_t full_total = scene_manager_->getScene().getSelectionGaussianCount();
        if (full_total == 0 || (mask.size() != visible_total && mask.size() != full_total)) {
            return {false, 0, "Mask size mismatch"};
        }

        auto tensor_mask = core::Tensor::empty({mask.size()}, core::Device::CPU, core::DataType::UInt8);
        std::memcpy(tensor_mask.ptr<uint8_t>(), mask.data(), mask.size() * sizeof(uint8_t));
        return applyMask(tensor_mask, mode);
    }

    SelectionResult SelectionService::applyMask(const core::Tensor& mask, SelectionMode mode) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const size_t visible_total = activeSelectionGaussianCount(scene_manager_);
        const size_t full_total = scene_manager_->getScene().getSelectionGaussianCount();
        if (full_total == 0 || (mask.numel() != visible_total && mask.numel() != full_total)) {
            return {false, 0, "Mask size mismatch"};
        }

        return commitSelection(mask, mode, {}, SelectionFilterState{}, "selection.mask");
    }

    SelectionResult SelectionService::previewMask(const core::Tensor& mask, SelectionMode mode) {
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        const size_t visible_total = activeSelectionGaussianCount(scene_manager_);
        const size_t full_total = scene_manager_->getScene().getSelectionGaussianCount();
        if (full_total == 0 || (mask.numel() != visible_total && mask.numel() != full_total)) {
            return {false, 0, "Mask size mismatch"};
        }

        SelectionCommitOptions options;
        options.push_undo = false;
        return commitSelection(mask, mode, {}, SelectionFilterState{}, "selection.preview", options);
    }

    void SelectionService::beginStroke() {
        if (!scene_manager_) {
            return;
        }

        const size_t n = activeSelectionGaussianCount(scene_manager_);
        if (n == 0) {
            return;
        }

        const auto existing = scene_manager_->getScene().getSelectionMask();
        selection_before_stroke_ =
            (existing && existing->is_valid()) ? std::make_shared<core::Tensor>(existing->clone()) : nullptr;

        try {
            (void)resetBoolScratchBuffer(stroke_selection_, n);
        } catch (const std::exception& e) {
            LOG_WARN("SelectionService: could not allocate stroke selection buffer: {}", e.what());
            selection_before_stroke_.reset();
            stroke_selection_ = {};
            stroke_active_ = false;
            return;
        }
        stroke_active_ = true;
    }

    core::Tensor* SelectionService::getStrokeSelection() {
        return stroke_active_ ? &stroke_selection_ : nullptr;
    }

    void SelectionService::applyCropFilterToStroke() {
        if (!stroke_active_ || !stroke_selection_.is_valid()) {
            return;
        }
        applyCropFilter(stroke_selection_);
    }

    SelectionResult SelectionService::finalizeStroke(SelectionMode mode, const std::vector<bool>& node_mask) {
        if (!stroke_active_ || !stroke_selection_.is_valid()) {
            return {false, 0, "No active stroke"};
        }

        const auto result = commitSelection(stroke_selection_, mode,
                                            node_mask.empty() ? effectiveNodeMask(true) : node_mask,
                                            defaultFilterState(), "selection.stroke");

        selection_before_stroke_.reset();
        stroke_selection_ = core::Tensor();
        stroke_active_ = false;

        if (rendering_manager_) {
            rendering_manager_->clearSelectionPreviews();
            rendering_manager_->markDirty(DirtyFlag::SELECTION);
        }

        return result;
    }

    void SelectionService::cancelStroke() {
        if (selection_before_stroke_ && scene_manager_) {
            scene_manager_->getScene().setSelectionMask(std::make_shared<core::Tensor>(selection_before_stroke_->clone()));
        }
        selection_before_stroke_.reset();
        stroke_selection_ = core::Tensor();
        stroke_active_ = false;

        if (rendering_manager_) {
            rendering_manager_->clearSelectionPreviews();
            rendering_manager_->markDirty(DirtyFlag::SELECTION);
        }
    }

    size_t SelectionService::getTotalGaussianCount() const {
        return activeSelectionGaussianCount(scene_manager_);
    }

    bool SelectionService::hasScreenPositions() const {
        const auto screen_positions = getScreenPositions();
        return screen_positions && screen_positions->is_valid();
    }

    std::shared_ptr<core::Tensor> SelectionService::getScreenPositions() const {
        if (testing_screen_positions_ && testing_screen_positions_->is_valid()) {
            return testing_screen_positions_;
        }
        if (!rendering_manager_) {
            return nullptr;
        }

        const auto context = resolveViewerViewportContext();
        if (!context || !context->info.valid()) {
            return nullptr;
        }

        return getScreenPositionsForContext(*context);
    }

    void SelectionService::setTestingScreenPositions(std::shared_ptr<core::Tensor> screen_positions) {
        testing_screen_positions_ = std::move(screen_positions);
    }

    void SelectionService::setTestingScreenPositionsForCamera(const int camera_index,
                                                              std::shared_ptr<core::Tensor> screen_positions) {
        if (camera_index < 0) {
            return;
        }
        if (screen_positions && screen_positions->is_valid()) {
            testing_camera_screen_positions_[camera_index] = std::move(screen_positions);
            return;
        }
        testing_camera_screen_positions_.erase(camera_index);
    }

    void SelectionService::setTestingViewport(ViewportInfo viewport) {
        testing_viewport_ = std::move(viewport);
    }

    void SelectionService::setTestingHoveredGaussianId(std::optional<int> hovered_gaussian_id) {
        testing_hovered_gaussian_id_ = hovered_gaussian_id;
    }

    void SelectionService::clearTestingOverrides() {
        testing_screen_positions_.reset();
        testing_camera_screen_positions_.clear();
        testing_viewport_.reset();
        testing_hovered_gaussian_id_.reset();
        viewport_screen_positions_.fill(nullptr);
        viewport_screen_position_keys_ = {};
    }

    std::optional<SelectionService::ViewerViewportContext> SelectionService::resolveViewerViewportContext(
        const std::optional<glm::vec2> screen_point,
        const std::optional<SplitViewPanelId> panel_override) const {
        ViewerViewportContext context;
        context.panel = panel_override.value_or(SplitViewPanelId::Left);

        if (testing_viewport_ && testing_viewport_->valid()) {
            static Viewport testing_viewport_source(1, 1);
            context.info = *testing_viewport_;
            context.viewport = &testing_viewport_source;
            return context;
        }

        auto* const gm = services().guiOrNull();
        if (!rendering_manager_ || !gm || !gm->getViewer()) {
            return std::nullopt;
        }

        const auto viewport_pos = gm->getViewportPos();
        const auto viewport_size = gm->getViewportSize();
        const auto panel = rendering_manager_->resolveViewerPanel(
            gm->getViewer()->getViewport(),
            {viewport_pos.x, viewport_pos.y},
            {viewport_size.x, viewport_size.y},
            screen_point,
            panel_override);
        if (!panel) {
            return std::nullopt;
        }

        context.panel = panel->panel;
        context.info = ViewportInfo{
            .x = panel->x,
            .y = panel->y,
            .width = panel->width,
            .height = panel->height,
            .render_width = panel->render_width,
            .render_height = panel->render_height,
        };
        context.viewport = panel->viewport;
        return context.info.valid() ? std::optional<ViewerViewportContext>(context) : std::nullopt;
    }

    std::shared_ptr<core::Tensor> SelectionService::getScreenPositionsForContext(
        const ViewerViewportContext& context) const {
        if (testing_screen_positions_ && testing_screen_positions_->is_valid()) {
            return testing_screen_positions_;
        }
        if (!context.info.valid()) {
            return nullptr;
        }
        if (!scene_manager_ || !rendering_manager_ || !context.viewport) {
            return nullptr;
        }

        const size_t panel_index = splitViewPanelIndex(context.panel);
        const auto settings = rendering_manager_->getSettings();
        Viewport projection_viewport = *context.viewport;
        projection_viewport.windowSize = {context.info.render_width, context.info.render_height};
        const auto viewport = viewportDataFromViewer(projection_viewport, context.info, settings);

        auto render_lock = acquireLiveModelRenderLock(scene_manager_);
        auto scene_state = scene_manager_->buildRenderState();
        if (!scene_state.combined_model || scene_state.combined_model->size() == 0) {
            viewport_screen_positions_[panel_index].reset();
            viewport_screen_position_keys_[panel_index] = {};
            return nullptr;
        }

        const ScreenPositionCacheKey key{
            .valid = true,
            .signature = makeScreenPositionCacheSignature(
                scene_state,
                viewport,
                settings.equirectangular,
                rendering_manager_->getViewportProjectionGeneration()),
        };
        if (viewport_screen_position_keys_[panel_index] == key &&
            viewport_screen_positions_[panel_index] &&
            viewport_screen_positions_[panel_index]->is_valid()) {
            return viewport_screen_positions_[panel_index];
        }

        auto screen_positions = projectGaussianScreenPositions(
            *scene_state.combined_model,
            viewport,
            settings.equirectangular,
            {.model_transforms = &scene_state.model_transforms,
             .transform_indices = scene_state.transform_indices,
             .node_visibility_mask = scene_state.node_visibility_mask});
        viewport_screen_positions_[panel_index] = screen_positions;
        viewport_screen_position_keys_[panel_index] =
            (screen_positions && screen_positions->is_valid()) ? key : ScreenPositionCacheKey{};
        return screen_positions;
    }

    bool SelectionService::beginInteractiveSelection(const SelectionShape shape, const SelectionMode mode,
                                                     const glm::vec2 start_pos, const float brush_radius,
                                                     const SelectionFilterState filters) {
        if (!scene_manager_ || !rendering_manager_) {
            return false;
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            return false;
        }

        cancelInteractiveSelection();

        interactive_selection_ = {};
        interactive_selection_.active = true;
        interactive_selection_.shape = shape;
        interactive_selection_.mode = mode;
        interactive_selection_.generation = ++interactive_selection_generation_;
        interactive_selection_.filters = filters;
        interactive_selection_.brush_radius = brush_radius;
        interactive_selection_.start_pos = start_pos;
        interactive_selection_.cursor_pos = start_pos;
        interactive_selection_.viewport_context = resolveViewerViewportContext(start_pos);
        if (!interactive_selection_.viewport_context || !interactive_selection_.viewport_context->info.valid()) {
            interactive_selection_ = {};
            return false;
        }
        try {
            (void)resetBoolScratchBuffer(interactive_selection_.working_selection, total);
        } catch (const std::exception& e) {
            LOG_WARN("SelectionService: could not allocate interactive selection preview buffers: {}", e.what());
            interactive_selection_ = {};
            return false;
        }

        switch (shape) {
        case SelectionShape::Brush:
        case SelectionShape::Lasso:
        case SelectionShape::Polygon:
        case SelectionShape::Rings:
            interactive_selection_.points.push_back(start_pos);
            break;
        case SelectionShape::Box:
        case SelectionShape::Sphere:
            interactive_selection_.volume_center_world = resolveInteractivePolygonWorldPoint(start_pos);
            if (!interactive_selection_.volume_center_world) {
                interactive_selection_ = {};
                return false;
            }
            break;
        case SelectionShape::Rectangle:
            break;
        }

        if (shape == SelectionShape::Polygon) {
            if (const auto world_point = resolveInteractivePolygonWorldPoint(start_pos)) {
                interactive_selection_.polygon_world_points.push_back(*world_point);
            }
        }

        refreshInteractivePreview();
        return true;
    }

    void SelectionService::updateInteractiveSelection(const glm::vec2 cursor_pos) {
        if (!interactive_selection_.active) {
            return;
        }

        auto& session = interactive_selection_;
        session.cursor_pos = cursor_pos;

        switch (session.shape) {
        case SelectionShape::Brush:
            if (session.points.empty() || glm::distance(session.points.back(), cursor_pos) > 1.0f) {
                session.points.push_back(cursor_pos);
            }
            break;
        case SelectionShape::Lasso:
            if (session.points.empty() || glm::distance(session.points.back(), cursor_pos) > 3.0f) {
                session.points.push_back(cursor_pos);
            }
            break;
        case SelectionShape::Rectangle:
        case SelectionShape::Rings:
        case SelectionShape::Box:
        case SelectionShape::Sphere:
            break;
        case SelectionShape::Polygon:
            if (session.dragged_polygon_vertex >= 0 &&
                static_cast<size_t>(session.dragged_polygon_vertex) < session.points.size()) {
                session.points[session.dragged_polygon_vertex] = cursor_pos;
                if (static_cast<size_t>(session.dragged_polygon_vertex) < session.polygon_world_points.size()) {
                    if (const auto world_point = resolveInteractivePolygonWorldPoint(cursor_pos)) {
                        session.polygon_world_points[session.dragged_polygon_vertex] = *world_point;
                    }
                }
            }
            break;
        }

        session.preview_dirty = true;
    }

    bool SelectionService::appendInteractivePolygonVertex(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || session.polygon_closed) {
            return false;
        }

        session.cursor_pos = point;
        glm::vec2 close_anchor = session.points.front();
        if (!session.polygon_world_points.empty()) {
            if (const auto projected = projectInteractivePolygonWorldPoint(session.polygon_world_points.front())) {
                close_anchor = *projected;
            }
        }

        if (session.points.size() >= 3 &&
            glm::distance(close_anchor, point) < POLYGON_CLOSE_DISTANCE_PX) {
            session.polygon_closed = true;
        } else {
            session.points.push_back(point);
            if (const auto world_point = resolveInteractivePolygonWorldPoint(point)) {
                session.polygon_world_points.push_back(*world_point);
            }
        }

        session.preview_dirty = true;
        return true;
    }

    bool SelectionService::beginInteractivePolygonVertexDrag(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed) {
            return false;
        }

        const int vertex = findInteractivePolygonVertexAt(point);
        if (vertex < 0) {
            return false;
        }

        session.dragged_polygon_vertex = vertex;
        session.cursor_pos = point;
        session.preview_dirty = true;
        return true;
    }

    bool SelectionService::insertInteractivePolygonVertex(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed) {
            return false;
        }

        const int edge = findInteractivePolygonEdgeAt(point);
        if (edge < 0) {
            return false;
        }

        const size_t insert_at = static_cast<size_t>(edge + 1);
        session.points.insert(session.points.begin() + static_cast<std::ptrdiff_t>(insert_at), point);

        if (!session.polygon_world_points.empty()) {
            const auto world_point = resolveInteractivePolygonWorldPoint(point);
            if (!world_point || session.polygon_world_points.size() + 1 != session.points.size()) {
                session.points.erase(session.points.begin() + static_cast<std::ptrdiff_t>(insert_at));
                return false;
            }

            session.polygon_world_points.insert(
                session.polygon_world_points.begin() + static_cast<std::ptrdiff_t>(insert_at), *world_point);
        }

        session.dragged_polygon_vertex = static_cast<int>(insert_at);
        session.cursor_pos = point;
        session.preview_dirty = true;
        return true;
    }

    void SelectionService::endInteractivePolygonVertexDrag() {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon) {
            return;
        }
        session.dragged_polygon_vertex = -1;
        session.preview_dirty = true;
    }

    bool SelectionService::removeInteractivePolygonVertex(const glm::vec2 point) {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed ||
            session.points.size() <= 3) {
            return false;
        }

        const int vertex = findInteractivePolygonVertexAt(point);
        if (vertex < 0) {
            return false;
        }

        if (!session.polygon_world_points.empty()) {
            if (static_cast<size_t>(vertex) >= session.polygon_world_points.size()) {
                return false;
            }
            session.polygon_world_points.erase(
                session.polygon_world_points.begin() + static_cast<std::ptrdiff_t>(vertex));
        }
        session.points.erase(session.points.begin() + static_cast<std::ptrdiff_t>(vertex));

        session.dragged_polygon_vertex = -1;
        session.cursor_pos = point;
        session.preview_dirty = true;
        return true;
    }

    bool SelectionService::undoInteractivePolygonVertex() {
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || session.points.empty()) {
            return false;
        }
        if (session.points.size() <= 1) {
            return false;
        }

        session.points.pop_back();
        if (session.polygon_world_points.size() >= session.points.size() + 1) {
            session.polygon_world_points.pop_back();
        }
        if (session.dragged_polygon_vertex >= static_cast<int>(session.points.size())) {
            session.dragged_polygon_vertex = -1;
        }
        session.polygon_closed = false;
        session.preview_dirty = true;
        return true;
    }

    SelectionResult SelectionService::finishInteractiveSelection() {
        auto& session = interactive_selection_;
        if (!session.active) {
            return {false, 0, "No active interactive selection"};
        }

        core::Tensor selection;
        if (!buildSelectionMaskForInteractiveSession(selection)) {
            return {false, 0, "Interactive selection is incomplete"};
        }

        const char* undo_name = "select.stroke";
        if (session.shape == SelectionShape::Box) {
            undo_name = "selection.box";
        } else if (session.shape == SelectionShape::Sphere) {
            undo_name = "selection.sphere";
        }

        const auto result = commitSelection(selection, session.mode,
                                            effectiveNodeMask(session.filters.restrict_to_selected_nodes),
                                            session.filters, undo_name);
        clearInteractivePreviewState();
        interactive_selection_ = {};
        return result;
    }

    void SelectionService::cancelInteractiveSelection() {
        clearInteractivePreviewState();
        interactive_selection_ = {};
    }

    void SelectionService::updatePassiveRingHoverPreview(const glm::vec2 cursor_pos,
                                                         const SelectionMode mode,
                                                         const SelectionFilterState filters) {
        if (!scene_manager_ || !rendering_manager_ || interactive_selection_.active) {
            return;
        }

        const auto context = resolveViewerViewportContext(cursor_pos);
        if (!context || !context->valid()) {
            rendering_manager_->clearCursorPreviewState();
            return;
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            rendering_manager_->clearCursorPreviewState();
            return;
        }

        auto& selection = resetBoolScratchBuffer(command_selection_buffer_, total);
        int picked_ring_id = -1;
        const auto exact_hit = buildRingSelectionForContext(*context, cursor_pos, selection, &picked_ring_id);
        bool hit = exact_hit.value_or(false);
        if (!exact_hit.has_value()) {
            const auto hovered_id = renderHoveredGaussianIdForViewerContext(*context, cursor_pos, filters);
            if (hovered_id && *hovered_id >= 0 && static_cast<size_t>(*hovered_id) < selection.numel()) {
                rendering::set_selection_element(selection.ptr<bool>(), *hovered_id, true);
                picked_ring_id = *hovered_id;
                hit = true;
            }
        }
        applyFilters(selection, filters, effectiveNodeMask(filters.restrict_to_selected_nodes));
        if (!selectionMaskHasAny(selection)) {
            picked_ring_id = -1;
            hit = false;
        }

        const auto render_cursor = screenToRender(cursor_pos, context->info);
        const bool add_mode = mode != SelectionMode::Remove;
        rendering_manager_->setCursorPreviewState(
            true, render_cursor.x, render_cursor.y, 0.0f, add_mode, nullptr, false, 0.0f,
            context->panel, picked_ring_id);
        if (hit && picked_ring_id >= 0) {
            rendering_manager_->setPreviewSelection(&command_selection_buffer_, add_mode);
        } else {
            rendering_manager_->clearPreviewSelection();
        }
        rendering_manager_->markDirty(DirtyFlag::SELECTION);
    }

    void SelectionService::refreshInteractivePreview() {
        auto& session = interactive_selection_;
        if (!session.active || !rendering_manager_) {
            return;
        }

        const bool continuous_refresh = (session.shape == SelectionShape::Polygon) ||
                                        (session.shape == SelectionShape::Rings);
        if (!session.preview_dirty && !continuous_refresh) {
            return;
        }
        LOG_TIMER("SelectionService::refreshInteractivePreview");

        if (!session.viewport_context || !session.viewport_context->info.valid()) {
            return;
        }
        const auto& context = *session.viewport_context;
        const auto& info = context.info;
        const bool add_mode = (session.mode != SelectionMode::Remove);

        {
            LOG_TIMER("SelectionService::refreshInteractivePreview.geometry");

            rendering_manager_->clearRectPreview();
            rendering_manager_->clearPolygonPreview();
            rendering_manager_->clearLassoPreview();
            rendering_manager_->clearPreviewSelection();
            if (session.shape != SelectionShape::Brush && session.shape != SelectionShape::Rings) {
                rendering_manager_->clearCursorPreviewState();
            }

            switch (session.shape) {
            case SelectionShape::Brush: {
                const auto render_cursor = screenToRender(session.cursor_pos, info);
                const float radius = session.brush_radius * (static_cast<float>(info.render_width) / info.width);
                rendering_manager_->setCursorPreviewState(
                    true, render_cursor.x, render_cursor.y, radius, add_mode, nullptr, false, 0.0f, context.panel);
                break;
            }
            case SelectionShape::Rectangle: {
                const auto render_start = screenToRender(session.start_pos, info);
                const auto render_end = screenToRender(session.cursor_pos, info);
                rendering_manager_->setRectPreview(
                    render_start.x, render_start.y, render_end.x, render_end.y, add_mode, context.panel, true);
                break;
            }
            case SelectionShape::Polygon: {
                if (!session.polygon_world_points.empty()) {
                    rendering_manager_->setPolygonPreviewWorldSpace(
                        session.polygon_world_points,
                        shouldClosePolygonPreview(),
                        add_mode,
                        context.panel);
                } else {
                    rendering_manager_->setPolygonPreview(
                        screenPointsToRender(getPolygonPreviewPoints(), info),
                        shouldClosePolygonPreview(),
                        add_mode,
                        context.panel);
                }
                break;
            }
            case SelectionShape::Lasso:
                rendering_manager_->setLassoPreview(
                    screenPointsToRender(session.points, info), add_mode, context.panel, true);
                break;
            case SelectionShape::Rings: {
                const auto render_cursor = screenToRender(session.cursor_pos, info);
                rendering_manager_->setCursorPreviewState(
                    true, render_cursor.x, render_cursor.y, 0.0f, add_mode, nullptr, false, 0.0f,
                    context.panel, -1);
                break;
            }
            case SelectionShape::Box:
            case SelectionShape::Sphere:
                if (const auto geometry = buildInteractiveVolumeGeometry()) {
                    publishInteractiveVolumeGeometry(*geometry);
                }
                break;
            }
        }

        {
            LOG_TIMER("SelectionService::refreshInteractivePreview.live_mask");
            core::Tensor selection;
            int picked_ring_id = -1;
            const bool has_preview_selection =
                (session.shape == SelectionShape::Brush)
                    ? buildInteractiveBrushPreviewIncremental()
                    : buildSelectionMaskForInteractiveSession(selection, true, &picked_ring_id);
            if (has_preview_selection) {
                rendering_manager_->setPreviewSelection(&interactive_selection_.working_selection, add_mode);
            }
            if (session.shape == SelectionShape::Rings) {
                const auto render_cursor = screenToRender(session.cursor_pos, info);
                rendering_manager_->setCursorPreviewState(
                    true, render_cursor.x, render_cursor.y, 0.0f, add_mode, nullptr, false, 0.0f,
                    context.panel, picked_ring_id);
            }
        }

        rendering_manager_->markDirty(DirtyFlag::SELECTION);
        session.preview_dirty = false;
    }

    SelectionResult SelectionService::commitSelection(const core::Tensor& selection, const SelectionMode mode,
                                                      const std::vector<bool>& node_mask,
                                                      const SelectionFilterState& filters,
                                                      const char* undo_name,
                                                      const SelectionCommitOptions options) {
        LOG_TIMER("SelectionService::commitSelection");
        if (!scene_manager_ || !rendering_manager_) {
            return {false, 0, "Missing managers"};
        }

        auto selection_mask = [&] {
            LOG_TIMER("commitSelection.ensureCudaBoolMask");
            return ensureCudaBoolMask(selection);
        }();
        if (!selection_mask.is_valid()) {
            return {false, 0, "Invalid selection mask"};
        }

        if (selection_mask.device() == core::Device::CUDA) {
            LOG_TIMER("commitSelection.sync_selection_stream");
            try {
                selection_mask.sync_to_stream(core::getCurrentCUDAStream());
            } catch (const std::exception& e) {
                return {false, 0, e.what()};
            }
        }

        {
            LOG_TIMER("commitSelection.applyFilters");
            applyFilters(selection_mask, filters, node_mask);
        }

        auto& scene = scene_manager_->getScene();
        const auto existing_mask = scene.getSelectionMask();
        const uint8_t group_id = scene.getActiveSelectionGroup();
        const size_t full_count = scene.getSelectionGaussianCount();
        const core::Tensor* const base_full_mask = selectionMaskForSize(options.base_selection, full_count);
        const core::Tensor* const existing_full_mask =
            base_full_mask ? base_full_mask : selectionMaskForSize(existing_mask, full_count);
        const bool using_base_selection = base_full_mask != nullptr;

        const bool intersect_mode = (mode == SelectionMode::Intersect);
        if (intersect_mode) {
            LOG_TIMER("commitSelection.intersect_active_selection");
            selection_mask = intersectWithActiveSelection(scene_manager_, selection_mask, existing_full_mask, group_id);
            if (!selection_mask.is_valid()) {
                return {false, 0, "Selection size mismatch"};
            }
        }

        const SelectionMode apply_mode = intersect_mode ? SelectionMode::Replace : mode;
        const size_t selection_count = selection_mask.numel();
        const bool add_mode = (apply_mode != SelectionMode::Remove);
        const bool replace_mode = (apply_mode == SelectionMode::Replace);
        const bool node_scope_restricts = nodeMaskRestrictsSelection(node_mask);
        const bool scoped_replace = replace_mode && existing_full_mask && node_scope_restricts;

        std::shared_ptr<core::Tensor> commit_transform_indices;
        const std::vector<bool>* commit_node_mask = nullptr;
        if (node_scope_restricts) {
            commit_transform_indices = scene.getTransformIndices();
            if (commit_transform_indices &&
                commit_transform_indices->is_valid() &&
                commit_transform_indices->numel() == selection_count) {
                commit_node_mask = &node_mask;
            }
        }

        core::Tensor scene_selection_mask;
        std::shared_ptr<core::Tensor> visible_indices;
        bool use_indexed_commit = false;
        if (selection_count == full_count) {
            if (!scoped_replace || commit_node_mask) {
                LOG_TIMER("commitSelection.expandSelectionToSceneMask.deferred_full_scene");
                scene_selection_mask = selection_mask;
            }
        } else {
            const size_t visible_count = activeSelectionGaussianCount(scene_manager_);
            const auto candidate_visible_indices = scene.getVisibleSelectionIndices();
            if (selection_count == visible_count && full_count > 0 &&
                candidate_visible_indices &&
                candidate_visible_indices->is_valid() &&
                candidate_visible_indices->numel() == selection_count &&
                (!scoped_replace || commit_node_mask)) {
                LOG_TIMER("commitSelection.expandSelectionToSceneMask.deferred_visible_indices");
                visible_indices = candidate_visible_indices;
                use_indexed_commit = true;
            }
        }

        if (!scene_selection_mask.is_valid() && !use_indexed_commit) {
            scene_selection_mask = [&] {
                LOG_TIMER("commitSelection.expandSelectionToSceneMask");
                return expandSelectionToSceneMask(
                    scene_manager_, selection_mask, apply_mode, group_id,
                    existing_full_mask,
                    node_mask);
            }();
        }
        if (!scene_selection_mask.is_valid() && !use_indexed_commit) {
            return {false, 0, "Selection size mismatch"};
        }
        const size_t n = use_indexed_commit ? full_count : scene_selection_mask.numel();

        auto locked_groups = [&] {
            LOG_TIMER("commitSelection.upload_locked_group_mask");
            return selection::upload_locked_group_mask(
                scene, locked_groups_device_mask_, locked_groups_host_mask_, locked_groups_host_mask_valid_);
        }();
        if (!locked_groups) {
            return {false, 0, locked_groups.error()};
        }

        const core::Tensor empty_mask;
        const auto* existing_ptr = using_base_selection
                                       ? selectionMaskForSize(base_full_mask, n)
                                       : selectionMaskForSize(existing_mask, n);
        const auto& existing_ref = existing_ptr ? *existing_ptr : empty_mask;
        auto& output_mask = acquireSelectionOutputBuffer(selection_output_buffers_, selection_output_buffer_index_, n);
        const std::vector<bool> empty_node_mask;
        const auto& final_node_mask = commit_node_mask ? *commit_node_mask : empty_node_mask;
        const bool count_groups_in_apply = !use_indexed_commit;
        const bool can_apply_group_deltas =
            count_groups_in_apply && !using_base_selection && (!existing_ptr || !scene.selectionGroupCountsDirty());
        const auto base_group_counts = can_apply_group_deltas
                                           ? cachedSelectionGroupCounts(scene)
                                           : core::Scene::SelectionGroupCounts{};

        if (use_indexed_commit) {
            LOG_TIMER("commitSelection.apply_selection_group_indexed_tensor_mask");
            rendering::apply_selection_group_indexed_tensor_mask(
                selection_mask, *visible_indices, existing_ref, output_mask, group_id, *locked_groups,
                add_mode, commit_transform_indices.get(), final_node_mask, replace_mode);
        } else {
            LOG_TIMER("commitSelection.apply_selection_group_tensor_mask");
            rendering::apply_selection_group_tensor_mask(
                scene_selection_mask, existing_ref, output_mask, group_id, *locked_groups,
                add_mode, commit_transform_indices.get(), final_node_mask, replace_mode,
                &selection_group_counts_scratch_);
        }

        core::Scene::SelectionGroupCounts group_counts{};
        bool selection_change_known = false;
        size_t selection_changed_count = 0;
        try {
            LOG_TIMER("commitSelection.count_selection_groups");
            if (count_groups_in_apply) {
                const auto delta_result =
                    rendering::read_selection_group_delta_result(selection_group_counts_scratch_);
                selection_changed_count = delta_result.changed_count;
                selection_change_known = true;
                if (can_apply_group_deltas) {
                    group_counts = applySelectionGroupDeltas(base_group_counts, delta_result.group_deltas);
                } else {
                    group_counts = rendering::count_selection_groups(output_mask, selection_group_counts_scratch_);
                }
            } else {
                group_counts = rendering::count_selection_groups(output_mask, selection_group_counts_scratch_);
            }
        } catch (const std::exception& e) {
            return {false, 0, e.what()};
        }
        size_t selected_count = 0;
        for (size_t i = 1; i < group_counts.size(); ++i) {
            selected_count += group_counts[i];
        }

        std::unique_ptr<op::SceneSnapshot> entry;
        if (options.push_undo) {
            LOG_TIMER("commitSelection.snapshot_captureSelection");
            entry = std::make_unique<op::SceneSnapshot>(*scene_manager_, undo_name);
            if (selection_change_known && !using_base_selection) {
                entry->setSelectionChangeHint(selection_changed_count > 0, true);
            }
            entry->captureSelection();
        }

        // Snapshot the selection result before reusing the rotating output buffer.
        auto new_selection = [&] {
            LOG_TIMER("commitSelection.clone_output_mask");
            return std::make_shared<core::Tensor>(output_mask.clone());
        }();
        {
            LOG_TIMER("commitSelection.setSelectionMask");
            scene.setSelectionMaskWithGroupCounts(new_selection, selected_count, group_counts);
        }
        if (const auto normalized_selection = scene.getSelectionMask();
            normalized_selection && normalized_selection->is_valid()) {
            selected_count = normalized_selection->count_nonzero();
        } else {
            selected_count = 0;
        }

        if (entry) {
            {
                LOG_TIMER("commitSelection.snapshot_captureAfter");
                entry->captureAfter();
            }
            {
                LOG_TIMER("commitSelection.pushSceneSnapshot");
                op::pushSceneSnapshotIfChanged(std::move(entry));
            }
        }

        rendering_manager_->markDirty(DirtyFlag::SELECTION);
        return {true, selected_count, {}};
    }

    std::shared_ptr<core::Tensor> SelectionService::resolveCommandScreenPositions(const int camera_index) const {
        if (camera_index >= 0) {
            if (const auto it = testing_camera_screen_positions_.find(camera_index);
                it != testing_camera_screen_positions_.end() &&
                it->second &&
                it->second->is_valid()) {
                return it->second;
            }
            if (auto remote_positions = renderScreenPositionsForCamera(camera_index);
                remote_positions && remote_positions->is_valid()) {
                return remote_positions;
            }
        }
        return getScreenPositions();
    }

    std::shared_ptr<core::Tensor> SelectionService::renderScreenPositionsForCamera(const int camera_index) const {
        if (!scene_manager_ || !rendering_manager_ || camera_index < 0) {
            return nullptr;
        }

        auto render_lock = acquireLiveModelRenderLock(scene_manager_);
        auto cameras = scene_manager_->getScene().getAllCameras();
        if (camera_index >= static_cast<int>(cameras.size()) || !cameras[camera_index]) {
            return nullptr;
        }

        auto scene_state = scene_manager_->buildRenderState();
        if (!scene_state.combined_model || scene_state.combined_model->size() == 0) {
            return nullptr;
        }

        const auto settings = rendering_manager_->getSettings();
        const auto viewport = viewportDataFromCamera(*cameras[camera_index]);
        return projectGaussianScreenPositions(
            *scene_state.combined_model,
            viewport,
            settings.equirectangular,
            {.model_transforms = &scene_state.model_transforms,
             .transform_indices = scene_state.transform_indices,
             .node_visibility_mask = scene_state.node_visibility_mask});
    }

    std::shared_ptr<core::Tensor> SelectionService::renderScreenPositionsForCurrentViewport() const {
        const auto context = resolveViewerViewportContext();
        if (!context) {
            return nullptr;
        }
        return getScreenPositionsForContext(*context);
    }

    std::optional<int> SelectionService::resolveCommandHoveredGaussianId(const float x, const float y,
                                                                         const int camera_index,
                                                                         const SelectionFilterState& filters) {
        if (testing_hovered_gaussian_id_.has_value()) {
            return testing_hovered_gaussian_id_;
        }

        if (camera_index >= 0) {
            if (auto hovered_id = renderHoveredGaussianIdForCamera(x, y, camera_index, filters);
                hovered_id.has_value()) {
                return hovered_id;
            }
        }

        return renderHoveredGaussianIdForCurrentViewport(x, y, filters);
    }

    std::optional<int> SelectionService::renderHoveredGaussianIdForCamera(const float x, const float y,
                                                                          const int camera_index,
                                                                          const SelectionFilterState& filters) {
        if (!scene_manager_ || camera_index < 0) {
            return std::nullopt;
        }

        auto cameras = scene_manager_->getScene().getAllCameras();
        if (camera_index >= static_cast<int>(cameras.size()) || !cameras[camera_index]) {
            return std::nullopt;
        }

        return renderHoveredGaussianId(viewportDataFromCamera(*cameras[camera_index]), {x, y}, filters);
    }

    std::optional<int> SelectionService::renderHoveredGaussianIdForViewerContext(
        const ViewerViewportContext& context,
        const glm::vec2 cursor_pos,
        const SelectionFilterState& filters) const {
        if (!context.valid()) {
            return std::nullopt;
        }

        const auto screen_positions = getScreenPositionsForContext(context);
        if (!screen_positions || !screen_positions->is_valid()) {
            return std::nullopt;
        }
        return pickHoveredGaussianIdFromScreenPositions(
            *screen_positions,
            screenToRender(cursor_pos, context.info),
            filters);
    }

    std::optional<int> SelectionService::renderHoveredGaussianIdForCurrentViewport(
        const float x, const float y, const SelectionFilterState& filters) {
        const auto context = resolveViewerViewportContext(glm::vec2{x, y});
        if (!context) {
            return std::nullopt;
        }
        return renderHoveredGaussianIdForViewerContext(*context, {x, y}, filters);
    }

    std::optional<int> SelectionService::renderHoveredGaussianId(const rendering::ViewportData& viewport,
                                                                 const glm::vec2 cursor_pos,
                                                                 const SelectionFilterState& filters) const {
        if (!scene_manager_ || !rendering_manager_) {
            return std::nullopt;
        }

        auto render_lock = acquireLiveModelRenderLock(scene_manager_);
        auto scene_state = scene_manager_->buildRenderState();
        if (!scene_state.combined_model || scene_state.combined_model->size() == 0) {
            return std::nullopt;
        }

        const auto settings = rendering_manager_->getSettings();
        auto screen_positions = projectGaussianScreenPositions(
            *scene_state.combined_model,
            viewport,
            settings.equirectangular,
            {.model_transforms = &scene_state.model_transforms,
             .transform_indices = scene_state.transform_indices,
             .node_visibility_mask = scene_state.node_visibility_mask});
        render_lock.reset();
        if (!screen_positions || !screen_positions->is_valid()) {
            return std::nullopt;
        }

        return pickHoveredGaussianIdFromScreenPositions(*screen_positions, cursor_pos, filters);
    }

    std::optional<int> SelectionService::pickHoveredGaussianIdFromScreenPositions(
        const core::Tensor& screen_positions,
        const glm::vec2 cursor_pos,
        const SelectionFilterState& filters) const {
        if (!scene_manager_) {
            return std::nullopt;
        }

        const auto hovered_result = pickProjectedGaussian(screen_positions, cursor_pos);
        if (!hovered_result) {
            return std::nullopt;
        }

        const int hovered_id = *hovered_result;
        if (hovered_id < 0 ||
            static_cast<size_t>(hovered_id) >= activeSelectionGaussianCount(scene_manager_)) {
            return std::nullopt;
        }

        if (filters.crop_filter || filters.depth_filter || filters.restrict_to_selected_nodes) {
            auto candidate = core::Tensor::zeros(
                {activeSelectionGaussianCount(scene_manager_)},
                core::Device::CUDA,
                core::DataType::Bool);
            rendering::set_selection_element(candidate.ptr<bool>(), hovered_id, true);
            applyFilters(candidate, filters, effectiveNodeMask(filters.restrict_to_selected_nodes));
            const auto candidate_cpu = candidate.cpu().contiguous();
            if (!candidate_cpu.ptr<bool>()[hovered_id]) {
                return std::nullopt;
            }
        }

        return hovered_id;
    }

    core::Tensor& SelectionService::resetBoolScratchBuffer(core::Tensor& buffer, const size_t size) {
        const bool needs_realloc = !buffer.is_valid() ||
                                   buffer.device() != core::Device::CUDA ||
                                   buffer.dtype() != core::DataType::Bool ||
                                   buffer.numel() != size;
        if (needs_realloc) {
            buffer = core::Tensor::zeros({size}, core::Device::CUDA, core::DataType::Bool);
            return buffer;
        }

        buffer.fill_(0.0f, buffer.stream());
        return buffer;
    }

    bool SelectionService::buildInteractiveBrushPreviewIncremental() {
        LOG_TIMER("SelectionService::buildInteractiveBrushPreviewIncremental");
        auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Brush || !scene_manager_ || !rendering_manager_) {
            return false;
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0 || session.points.empty()) {
            return false;
        }

        const bool needs_working_realloc =
            !session.working_selection.is_valid() ||
            session.working_selection.device() != core::Device::CUDA ||
            session.working_selection.dtype() != core::DataType::Bool ||
            session.working_selection.numel() != total;
        const auto node_mask = effectiveNodeMask(session.filters.restrict_to_selected_nodes);
        const bool node_scope_changed =
            session.preview_brush_point_count > 0 &&
            node_mask != session.live_preview_node_mask;
        if (needs_working_realloc) {
            session.working_selection = core::Tensor::zeros({total}, core::Device::CUDA, core::DataType::Bool);
            session.preview_brush_point_count = 0;
        } else if (session.preview_brush_point_count > session.points.size() || node_scope_changed) {
            session.working_selection.zero_();
            session.preview_brush_point_count = 0;
        }

        if (session.preview_brush_point_count == session.points.size()) {
            return session.preview_brush_point_count > 0;
        }

        const size_t first_point =
            (session.preview_brush_point_count == 0) ? 0 : (session.preview_brush_point_count - 1);
        if (first_point >= session.points.size()) {
            return session.preview_brush_point_count > 0;
        }

        std::vector<glm::vec2> delta_points;
        delta_points.reserve(session.points.size() - first_point);
        delta_points.insert(delta_points.end(), session.points.begin() + static_cast<std::ptrdiff_t>(first_point),
                            session.points.end());

        const bool needs_delta_realloc =
            !session.live_delta_selection.is_valid() ||
            session.live_delta_selection.device() != core::Device::CUDA ||
            session.live_delta_selection.dtype() != core::DataType::Bool ||
            session.live_delta_selection.numel() != total;
        if (needs_delta_realloc) {
            session.live_delta_selection = core::Tensor::zeros({total}, core::Device::CUDA, core::DataType::Bool);
        }
        auto& delta_selection = session.live_delta_selection;
        delta_selection.fill_(0.0f, delta_selection.stream());
        {
            LOG_TIMER("SelectionService::buildInteractiveBrushPreviewIncremental.brush_delta");
            if (!buildBrushSelection(delta_points, session.brush_radius, delta_selection)) {
                return session.preview_brush_point_count > 0;
            }
        }

        {
            LOG_TIMER("SelectionService::buildInteractiveBrushPreviewIncremental.applyFilters");
            applyFilters(delta_selection, session.filters, node_mask);
        }

        if (session.preview_brush_point_count == 0) {
            LOG_TIMER("SelectionService::buildInteractiveBrushPreviewIncremental.copy_initial");
            session.working_selection.copy_from(delta_selection);
        } else {
            LOG_TIMER("SelectionService::buildInteractiveBrushPreviewIncremental.merge");
            rendering::merge_selection_mask_or(session.working_selection, delta_selection);
        }

        session.preview_brush_point_count = session.points.size();
        session.live_preview_node_mask = node_mask;
        return true;
    }

    std::optional<SelectionService::ViewportInfo> SelectionService::resolveViewportInfo() const {
        const auto context = resolveViewerViewportContext();
        if (!context || !context->info.valid()) {
            return std::nullopt;
        }
        return context->info;
    }

    bool SelectionService::buildSelectionMaskForInteractiveSession(core::Tensor& selection_out,
                                                                   const bool include_polygon_cursor,
                                                                   int* const picked_ring_id_out) {
        LOG_TIMER("SelectionService::buildSelectionMaskForInteractiveSession");
        if (picked_ring_id_out) {
            *picked_ring_id_out = -1;
        }
        auto& session = interactive_selection_;
        if (!session.active || !scene_manager_ || !rendering_manager_) {
            return false;
        }

        const size_t total = activeSelectionGaussianCount(scene_manager_);
        if (total == 0) {
            return false;
        }

        bool success = false;
        switch (session.shape) {
        case SelectionShape::Brush:
            success = buildInteractiveBrushPreviewIncremental();
            success = success && session.preview_brush_point_count == session.points.size() &&
                      session.working_selection.is_valid() && session.working_selection.numel() == total;
            if (success) {
                selection_out = session.working_selection;
            }
            break;
        case SelectionShape::Rectangle:
            selection_out = resetBoolScratchBuffer(session.working_selection, total);
            success = buildRectangleSelection(session.start_pos, session.cursor_pos, selection_out);
            break;
        case SelectionShape::Polygon: {
            selection_out = resetBoolScratchBuffer(session.working_selection, total);
            if (!session.polygon_world_points.empty()) {
                success = buildWorldPolygonSelection(session.polygon_world_points, selection_out);
            } else {
                const auto polygon_points = include_polygon_cursor ? getPolygonPreviewPoints() : session.points;
                success = buildPolygonSelection(polygon_points, selection_out);
            }
            break;
        }
        case SelectionShape::Lasso:
            selection_out = resetBoolScratchBuffer(session.working_selection, total);
            success = buildPolygonSelection(session.points, selection_out);
            break;
        case SelectionShape::Rings:
            selection_out = resetBoolScratchBuffer(session.working_selection, total);
            success = buildRingSelection(
                session.cursor_pos, selection_out, true, !include_polygon_cursor, picked_ring_id_out);
            break;
        case SelectionShape::Box:
        case SelectionShape::Sphere:
            selection_out = resetBoolScratchBuffer(session.working_selection, total);
            if (const auto geometry = buildInteractiveVolumeGeometry()) {
                success = buildVolumeSelection(*geometry, selection_out);
            }
            break;
        }

        if (!success) {
            return false;
        }

        applyFilters(selection_out, session.filters, effectiveNodeMask(session.filters.restrict_to_selected_nodes));
        return true;
    }

    bool SelectionService::buildBrushSelection(const std::vector<glm::vec2>& points, const float radius,
                                               core::Tensor& selection_out) const {
        LOG_TIMER("SelectionService::buildBrushSelection");
        if (points.empty()) {
            return false;
        }

        const auto& session = interactive_selection_;
        if (!scene_manager_ || !rendering_manager_ || !session.viewport_context ||
            !session.viewport_context->info.valid() || !session.viewport_context->viewport) {
            return false;
        }
        const auto& info = session.viewport_context->info;

        const auto primitives = buildBrushPrimitives(points, radius, info);
        if (primitives.empty()) {
            return false;
        }
        const auto settings = rendering_manager_->getSettings();
        if (!testing_screen_positions_ && !testing_viewport_) {
            Viewport projection_viewport = *session.viewport_context->viewport;
            projection_viewport.windowSize = {info.render_width, info.render_height};
            const auto frame_view = frameViewFromViewport(
                viewportDataFromViewer(projection_viewport, info, settings),
                settings.background_color,
                settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
            if (auto selection = tryBuildVksplatSelectionMask(
                    scene_manager_, rendering_manager_, frame_view, settings.equirectangular,
                    RenderingManager::VksplatSelectionMaskShape::Brush, primitives);
                selection && copySelectionIfSameSize(*selection, selection_out)) {
                return true;
            }
        }

        const auto screen_positions = getScreenPositionsForContext(*session.viewport_context);
        if (!screen_positions || !screen_positions->is_valid() || screen_positions->size(0) != selection_out.numel()) {
            return false;
        }

        const float scale_x = static_cast<float>(info.render_width) / info.width;
        const float scaled_radius = radius * scale_x;
        constexpr float STEP_FACTOR = 0.5f;
        constexpr int MAX_BRUSH_STEPS = 128;

        for (size_t i = 0; i < points.size(); ++i) {
            const glm::vec2 from = (i == 0) ? points[i] : points[i - 1];
            const glm::vec2 to = points[i];
            const glm::vec2 delta = to - from;
            const float step_spacing = std::max(radius * STEP_FACTOR, 1.0f);
            const int num_steps = std::min(
                MAX_BRUSH_STEPS,
                std::max(1, static_cast<int>(std::ceil(glm::length(delta) / step_spacing))));
            for (int step = 0; step < num_steps; ++step) {
                const float t = (num_steps == 1) ? 1.0f
                                                 : static_cast<float>(step + 1) / static_cast<float>(num_steps);
                const glm::vec2 sample = from + delta * t;
                const auto render = screenToRender(sample, info);
                rendering::brush_select_tensor(*screen_positions, render.x, render.y, scaled_radius, selection_out);
            }
        }

        return true;
    }

    bool SelectionService::buildRectangleSelection(const glm::vec2 start, const glm::vec2 end,
                                                   core::Tensor& selection_out) const {
        LOG_TIMER("SelectionService::buildRectangleSelection");
        const auto& session = interactive_selection_;
        if (!scene_manager_ || !rendering_manager_ || !session.viewport_context ||
            !session.viewport_context->info.valid() || !session.viewport_context->viewport) {
            return false;
        }
        const auto& info = session.viewport_context->info;

        const auto render_start = screenToRender(start, info);
        const auto render_end = screenToRender(end, info);
        const std::vector<glm::vec4> primitives{{
            std::min(render_start.x, render_end.x),
            std::min(render_start.y, render_end.y),
            std::max(render_start.x, render_end.x),
            std::max(render_start.y, render_end.y),
        }};
        const auto settings = rendering_manager_->getSettings();
        if (!testing_screen_positions_ && !testing_viewport_) {
            Viewport projection_viewport = *session.viewport_context->viewport;
            projection_viewport.windowSize = {info.render_width, info.render_height};
            const auto frame_view = frameViewFromViewport(
                viewportDataFromViewer(projection_viewport, info, settings),
                settings.background_color,
                settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
            if (auto selection = tryBuildVksplatSelectionMask(
                    scene_manager_, rendering_manager_, frame_view, settings.equirectangular,
                    RenderingManager::VksplatSelectionMaskShape::Rectangle, primitives);
                selection && copySelectionIfSameSize(*selection, selection_out)) {
                return true;
            }
        }

        const auto screen_positions = getScreenPositionsForContext(*session.viewport_context);
        if (!screen_positions || !screen_positions->is_valid() || screen_positions->size(0) != selection_out.numel()) {
            return false;
        }

        rendering::rect_select_tensor(*screen_positions,
                                      std::min(render_start.x, render_end.x),
                                      std::min(render_start.y, render_end.y),
                                      std::max(render_start.x, render_end.x),
                                      std::max(render_start.y, render_end.y),
                                      selection_out);
        return true;
    }

    bool SelectionService::buildPolygonSelection(const std::vector<glm::vec2>& points,
                                                 core::Tensor& selection_out) const {
        LOG_TIMER("SelectionService::buildPolygonSelection");
        if (points.size() < 3) {
            return false;
        }

        const auto& session = interactive_selection_;
        if (!scene_manager_ || !rendering_manager_ || !session.viewport_context ||
            !session.viewport_context->info.valid() || !session.viewport_context->viewport) {
            return false;
        }
        const auto& info = session.viewport_context->info;

        std::vector<glm::vec2> render_points;
        render_points.reserve(points.size());
        for (const auto& point : points) {
            render_points.push_back(screenToRender(point, info));
        }

        const auto settings = rendering_manager_->getSettings();
        if (!testing_screen_positions_ && !testing_viewport_) {
            Viewport projection_viewport = *session.viewport_context->viewport;
            projection_viewport.windowSize = {info.render_width, info.render_height};
            const auto frame_view = frameViewFromViewport(
                viewportDataFromViewer(projection_viewport, info, settings),
                settings.background_color,
                settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
            if (auto selection = tryBuildVksplatPolygonSelectionMask(
                    scene_manager_, rendering_manager_, frame_view, settings.equirectangular, render_points);
                selection && copySelectionIfSameSize(*selection, selection_out)) {
                return true;
            }
        }

        const auto screen_positions = getScreenPositionsForContext(*session.viewport_context);
        if (!screen_positions || !screen_positions->is_valid() || screen_positions->size(0) != selection_out.numel()) {
            return false;
        }

        auto& polygon =
            uploadFloat2PointsToBuffer(render_points, polygon_vertex_host_buffer_, polygon_vertex_device_buffer_);
        rendering::polygon_select_tensor(*screen_positions, polygon, selection_out);
        return true;
    }

    bool SelectionService::buildWorldPolygonSelection(const std::vector<glm::vec3>& world_points,
                                                      core::Tensor& selection_out) const {
        LOG_TIMER("SelectionService::buildWorldPolygonSelection");
        if (world_points.size() < 3) {
            return false;
        }

        const auto& session = interactive_selection_;
        if (!scene_manager_ || !rendering_manager_ || !session.viewport_context ||
            !session.viewport_context->viewport || !session.viewport_context->info.valid()) {
            return false;
        }

        Viewport projection_viewport = *session.viewport_context->viewport;
        projection_viewport.windowSize = {
            session.viewport_context->info.render_width,
            session.viewport_context->info.render_height};
        const auto settings = rendering_manager_->getSettings();

        std::vector<glm::vec2> render_points;
        render_points.reserve(world_points.size());
        for (const auto& wp : world_points) {
            const auto projected = rendering::projectWorldPoint(
                projection_viewport.camera.R,
                projection_viewport.camera.t,
                {session.viewport_context->info.render_width, session.viewport_context->info.render_height},
                wp,
                settings.focal_length_mm,
                settings.orthographic,
                settings.ortho_scale);
            if (!projected) {
                return false;
            }
            render_points.emplace_back(projected->x, projected->y);
        }

        if (!testing_screen_positions_ && !testing_viewport_) {
            const auto frame_view = frameViewFromViewport(
                viewportDataFromViewer(projection_viewport, session.viewport_context->info, settings),
                settings.background_color,
                settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
            if (auto selection = tryBuildVksplatPolygonSelectionMask(
                    scene_manager_, rendering_manager_, frame_view, settings.equirectangular, render_points);
                selection && copySelectionIfSameSize(*selection, selection_out)) {
                return true;
            }
        }

        const auto screen_positions = getScreenPositionsForContext(*session.viewport_context);
        if (!screen_positions || !screen_positions->is_valid() || screen_positions->size(0) != selection_out.numel()) {
            return false;
        }

        auto& polygon =
            uploadFloat2PointsToBuffer(render_points, polygon_vertex_host_buffer_, polygon_vertex_device_buffer_);
        rendering::polygon_select_tensor(*screen_positions, polygon, selection_out);
        return true;
    }

    bool SelectionService::buildRingSelection(const glm::vec2 cursor_pos, core::Tensor& selection_out,
                                              const bool try_exact_ring_pick,
                                              const bool require_exact_ring_hit,
                                              int* const picked_ring_id_out) const {
        if (picked_ring_id_out) {
            *picked_ring_id_out = -1;
        }
        if (!rendering_manager_) {
            return false;
        }

        const auto& session = interactive_selection_;
        int hovered_id = testing_hovered_gaussian_id_.value_or(-1);
        if (hovered_id >= 0 && static_cast<size_t>(hovered_id) < selection_out.numel()) {
            rendering::set_selection_element(selection_out.ptr<bool>(), hovered_id, true);
            if (picked_ring_id_out) {
                *picked_ring_id_out = hovered_id;
            }
            return true;
        }

        if (try_exact_ring_pick &&
            scene_manager_ &&
            session.viewport_context &&
            session.viewport_context->info.valid() &&
            session.viewport_context->viewport &&
            !testing_screen_positions_ &&
            !testing_viewport_) {
            if (const auto exact_hit =
                    buildRingSelectionForContext(*session.viewport_context, cursor_pos, selection_out, picked_ring_id_out)) {
                return *exact_hit || !require_exact_ring_hit;
            }
        }

        if (!session.viewport_context) {
            return false;
        }
        hovered_id =
            renderHoveredGaussianIdForViewerContext(*session.viewport_context, cursor_pos, session.filters)
                .value_or(-1);
        if (hovered_id < 0 || static_cast<size_t>(hovered_id) >= selection_out.numel()) {
            return !require_exact_ring_hit;
        }

        rendering::set_selection_element(selection_out.ptr<bool>(), hovered_id, true);
        if (picked_ring_id_out) {
            *picked_ring_id_out = hovered_id;
        }
        return true;
    }

    std::optional<bool> SelectionService::buildRingSelectionForContext(const ViewerViewportContext& context,
                                                                       const glm::vec2 cursor_pos,
                                                                       core::Tensor& selection_out,
                                                                       int* const picked_ring_id_out) const {
        if (picked_ring_id_out) {
            *picked_ring_id_out = -1;
        }
        if (!scene_manager_ || !rendering_manager_ || !context.info.valid() || !context.viewport) {
            return std::nullopt;
        }

        const auto& info = context.info;
        const auto settings = rendering_manager_->getSettings();
        const auto render_cursor = screenToRender(cursor_pos, info);
        const float render_padding = RING_PICK_PADDING_PX * (static_cast<float>(info.render_width) / info.width);
        const std::vector<glm::vec4> primitives{{render_cursor.x, render_cursor.y, render_padding, 0.0f}};
        Viewport projection_viewport = *context.viewport;
        projection_viewport.windowSize = {info.render_width, info.render_height};
        const auto frame_view = frameViewFromViewport(
            viewportDataFromViewer(projection_viewport, info, settings),
            settings.background_color,
            settings.depth_clip_enabled ? settings.depth_clip_far : lfs::rendering::DEFAULT_FAR_PLANE);
        std::uint32_t picked_ring_id = std::numeric_limits<std::uint32_t>::max();
        if (auto selection = tryBuildVksplatSelectionMask(
                scene_manager_, rendering_manager_, frame_view, settings.equirectangular,
                RenderingManager::VksplatSelectionMaskShape::Ring, primitives, &picked_ring_id);
            selection && copySelectionIfSameSize(*selection, selection_out)) {
            if (picked_ring_id != std::numeric_limits<std::uint32_t>::max() &&
                picked_ring_id < selection_out.numel() &&
                picked_ring_id_out) {
                *picked_ring_id_out = static_cast<int>(picked_ring_id);
            }
            return selectionMaskHasAny(*selection);
        }
        return std::nullopt;
    }

    std::optional<SelectionService::InteractiveVolumeGeometry>
    SelectionService::buildInteractiveVolumeGeometry() const {
        const auto& session = interactive_selection_;
        if (!rendering_manager_ || !session.active ||
            (session.shape != SelectionShape::Box && session.shape != SelectionShape::Sphere) ||
            !session.viewport_context || !session.viewport_context->viewport ||
            !session.viewport_context->info.valid()) {
            return std::nullopt;
        }

        if (!session.volume_center_world) {
            return std::nullopt;
        }
        const glm::vec3 center_world = *session.volume_center_world;

        const auto& info = session.viewport_context->info;
        Viewport projection_viewport = *session.viewport_context->viewport;
        projection_viewport.windowSize = {info.render_width, info.render_height};

        const auto settings = rendering_manager_->getSettings();
        const auto render_point = screenToRender(session.cursor_pos, info);
        const glm::vec3 forward = rendering::cameraForward(projection_viewport.camera.R);
        float depth = glm::dot(center_world - projection_viewport.camera.t, forward);
        if (!std::isfinite(depth) || depth <= 0.0f) {
            depth = glm::length(center_world - projection_viewport.camera.t);
        }
        if (!std::isfinite(depth) || depth <= 0.0f) {
            return std::nullopt;
        }

        const float ortho_scale = projection_viewport.ortho_scale_override.value_or(settings.ortho_scale);
        const glm::vec3 drag_world = projection_viewport.unprojectPixel(
            render_point.x,
            render_point.y,
            depth,
            settings.focal_length_mm,
            settings.orthographic,
            ortho_scale);
        if (!Viewport::isValidWorldPosition(drag_world)) {
            return std::nullopt;
        }

        const float drag_radius = glm::length(drag_world - center_world);
        if (!std::isfinite(drag_radius)) {
            return std::nullopt;
        }
        const float radius = std::max(drag_radius, MIN_VOLUME_SELECTION_RADIUS);

        glm::mat4 transform(1.0f);
        transform[3] = glm::vec4(center_world, 1.0f);

        InteractiveVolumeGeometry geometry;
        geometry.center_world = center_world;
        geometry.radius = radius;
        geometry.visualizer_transform = transform;
        geometry.box_min = glm::vec3(-radius);
        geometry.box_max = glm::vec3(radius);
        geometry.ellipsoid_radii = glm::vec3(radius);
        return geometry;
    }

    bool SelectionService::buildVolumeSelection(const InteractiveVolumeGeometry& geometry,
                                                core::Tensor& selection_out) const {
        const auto& session = interactive_selection_;
        if (!selection_out.is_valid() ||
            (session.shape != SelectionShape::Box && session.shape != SelectionShape::Sphere)) {
            return false;
        }

        selection_out.fill_(1.0f, selection_out.stream());

        const glm::mat4 world_to_volume = glm::inverse(geometry.visualizer_transform);
        const float* const transform_ptr = glm::value_ptr(world_to_volume);
        const auto transform =
            core::Tensor::from_vector(std::vector<float>(transform_ptr, transform_ptr + 16), {4, 4});

        if (session.shape == SelectionShape::Box) {
            const auto box_min = core::Tensor::from_vector(
                {geometry.box_min.x, geometry.box_min.y, geometry.box_min.z}, {3});
            const auto box_max = core::Tensor::from_vector(
                {geometry.box_max.x, geometry.box_max.y, geometry.box_max.z}, {3});
            applyCropFilter(selection_out, &transform, &box_min, &box_max, nullptr, nullptr, false);
        } else {
            const auto radii = core::Tensor::from_vector(
                {geometry.ellipsoid_radii.x, geometry.ellipsoid_radii.y, geometry.ellipsoid_radii.z}, {3});
            applyCropFilter(selection_out, nullptr, nullptr, nullptr, &transform, &radii, false);
        }

        return true;
    }

    void SelectionService::publishInteractiveVolumeGeometry(const InteractiveVolumeGeometry& geometry) const {
        const auto& session = interactive_selection_;
        if (session.shape != SelectionShape::Box && session.shape != SelectionShape::Sphere) {
            return;
        }

        if (auto* const gui = services().guiOrNull()) {
            const auto mode = session.shape == SelectionShape::Sphere
                                  ? SelectionSubMode::Sphere
                                  : SelectionSubMode::Box;
            gui->gizmo().setSelectionVolumeFromDrag(mode,
                                                    session.mode,
                                                    session.generation,
                                                    geometry.center_world,
                                                    geometry.radius);
            return;
        }

        if (!rendering_manager_) {
            return;
        }

        if (session.shape == SelectionShape::Box) {
            rendering_manager_->setCropboxGizmoState(
                true, geometry.box_min, geometry.box_max, geometry.visualizer_transform, false);
            rendering_manager_->setEllipsoidGizmoActive(false);
        } else {
            rendering_manager_->setEllipsoidGizmoState(
                true, geometry.ellipsoid_radii, geometry.visualizer_transform, false);
            rendering_manager_->setCropboxGizmoActive(false);
        }
        rendering_manager_->markDirty(DirtyFlag::SPLATS | DirtyFlag::OVERLAY);
    }

    std::vector<glm::vec2> SelectionService::getPolygonPreviewPoints() const {
        const auto& session = interactive_selection_;
        std::vector<glm::vec2> preview_points = session.points;
        if (!session.active || session.shape != SelectionShape::Polygon || session.polygon_closed) {
            return preview_points;
        }

        if (preview_points.empty()) {
            preview_points.push_back(session.cursor_pos);
            return preview_points;
        }

        glm::vec2 close_anchor = preview_points.front();
        if (const auto projected = resolveInteractivePolygonDisplayPoint(0)) {
            close_anchor = *projected;
        }

        const bool can_preview_close = preview_points.size() >= 3 &&
                                       glm::distance(close_anchor, session.cursor_pos) < POLYGON_CLOSE_DISTANCE_PX;
        if (!can_preview_close &&
            glm::distance(preview_points.back(), session.cursor_pos) > POLYGON_CURSOR_APPEND_EPSILON_PX) {
            preview_points.push_back(session.cursor_pos);
        }
        return preview_points;
    }

    std::optional<glm::vec2> SelectionService::resolveInteractivePolygonDisplayPoint(const size_t index) const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || index >= session.points.size()) {
            return std::nullopt;
        }

        if (index < session.polygon_world_points.size()) {
            if (const auto projected = projectInteractivePolygonWorldPoint(session.polygon_world_points[index])) {
                return projected;
            }
        }

        return session.points[index];
    }

    int SelectionService::findInteractivePolygonVertexAt(const glm::vec2 screen_point) const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon) {
            return -1;
        }

        const float radius_sq = POLYGON_VERTEX_HIT_RADIUS_PX * POLYGON_VERTEX_HIT_RADIUS_PX;
        for (size_t i = 0; i < session.points.size(); ++i) {
            if (const auto vertex = resolveInteractivePolygonDisplayPoint(i)) {
                const glm::vec2 delta = screen_point - *vertex;
                if (glm::dot(delta, delta) <= radius_sq) {
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    }

    int SelectionService::findInteractivePolygonEdgeAt(const glm::vec2 screen_point) const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon || !session.polygon_closed ||
            session.points.size() < 3) {
            return -1;
        }

        const float edge_radius_sq = POLYGON_EDGE_HIT_RADIUS_PX * POLYGON_EDGE_HIT_RADIUS_PX;
        const float vertex_radius_sq = POLYGON_VERTEX_HIT_RADIUS_PX * POLYGON_VERTEX_HIT_RADIUS_PX;
        float best_distance_sq = edge_radius_sq;
        int best_edge = -1;

        for (size_t i = 0; i < session.points.size(); ++i) {
            const size_t next = (i + 1) % session.points.size();
            const auto start = resolveInteractivePolygonDisplayPoint(i);
            const auto end = resolveInteractivePolygonDisplayPoint(next);
            if (!start || !end) {
                continue;
            }

            if (glm::dot(screen_point - *start, screen_point - *start) <= vertex_radius_sq ||
                glm::dot(screen_point - *end, screen_point - *end) <= vertex_radius_sq) {
                continue;
            }

            float t = 0.0f;
            const float distance_sq = distanceSquaredToSegment(screen_point, *start, *end, &t);
            if (distance_sq <= best_distance_sq && t > 0.0f && t < 1.0f) {
                best_distance_sq = distance_sq;
                best_edge = static_cast<int>(i);
            }
        }

        return best_edge;
    }

    std::optional<glm::vec3> SelectionService::resolveInteractivePolygonWorldPoint(const glm::vec2 screen_point) const {
        const auto& session = interactive_selection_;
        if (!rendering_manager_ || !session.viewport_context || !session.viewport_context->viewport ||
            !session.viewport_context->info.valid()) {
            return std::nullopt;
        }
        const glm::ivec2 rendered_size = rendering_manager_->getRenderedSize();
        if (rendered_size.x <= 0 || rendered_size.y <= 0) {
            return std::nullopt;
        }

        const auto settings = rendering_manager_->getSettings();
        const auto& info = session.viewport_context->info;
        Viewport projection_viewport = *session.viewport_context->viewport;
        projection_viewport.windowSize = {info.render_width, info.render_height};
        const auto render_point = screenToRender(screen_point, info);
        const float depth = rendering_manager_->getDepthAtPixel(
            static_cast<int>(render_point.x), static_cast<int>(render_point.y), session.viewport_context->panel);
        const float ortho_scale = projection_viewport.ortho_scale_override.value_or(settings.ortho_scale);

        if (depth > 0.0f) {
            const glm::vec3 world = projection_viewport.unprojectPixel(
                render_point.x,
                render_point.y,
                depth,
                settings.focal_length_mm,
                settings.orthographic,
                ortho_scale);
            if (Viewport::isValidWorldPosition(world)) {
                return world;
            }
        }

        const float pivot_distance = glm::length(projection_viewport.camera.pivot - projection_viewport.camera.t);
        const float fallback_distance = pivot_distance > 0.1f ? pivot_distance : 10.0f;
        const glm::vec3 fallback_world =
            projection_viewport.unprojectPixel(
                render_point.x,
                render_point.y,
                fallback_distance,
                settings.focal_length_mm,
                settings.orthographic,
                ortho_scale);
        if (Viewport::isValidWorldPosition(fallback_world)) {
            return fallback_world;
        }

        const glm::vec3 forward = rendering::cameraForward(projection_viewport.camera.R);
        return projection_viewport.camera.t + forward * fallback_distance;
    }

    std::optional<glm::vec2> SelectionService::projectInteractivePolygonWorldPoint(const glm::vec3 world_point) const {
        const auto& session = interactive_selection_;
        if (!rendering_manager_ || !session.viewport_context || !session.viewport_context->viewport ||
            !session.viewport_context->info.valid()) {
            return std::nullopt;
        }

        const auto& info = session.viewport_context->info;
        Viewport projection_viewport = *session.viewport_context->viewport;
        projection_viewport.windowSize = {info.render_width, info.render_height};
        const auto settings = rendering_manager_->getSettings();
        const float ortho_scale = projection_viewport.ortho_scale_override.value_or(settings.ortho_scale);
        const auto projected = rendering::projectWorldPoint(
            projection_viewport.camera.R,
            projection_viewport.camera.t,
            {info.render_width, info.render_height},
            world_point,
            settings.focal_length_mm,
            settings.orthographic,
            ortho_scale);
        if (!projected) {
            return std::nullopt;
        }

        const float scale_x = info.width / static_cast<float>(std::max(info.render_width, 1));
        const float scale_y = info.height / static_cast<float>(std::max(info.render_height, 1));
        return glm::vec2(info.x + projected->x * scale_x,
                         info.y + projected->y * scale_y);
    }

    bool SelectionService::shouldClosePolygonPreview() const {
        const auto& session = interactive_selection_;
        if (!session.active || session.shape != SelectionShape::Polygon) {
            return false;
        }
        if (session.polygon_closed) {
            return true;
        }

        glm::vec2 close_anchor = session.points.front();
        if (!session.polygon_world_points.empty()) {
            if (const auto projected = projectInteractivePolygonWorldPoint(session.polygon_world_points.front())) {
                close_anchor = *projected;
            }
        }

        return session.points.size() >= 3 &&
               glm::distance(close_anchor, session.cursor_pos) < POLYGON_CLOSE_DISTANCE_PX;
    }

    void SelectionService::applyFilters(core::Tensor& selection, const SelectionFilterState& filters,
                                        const std::vector<bool>& node_mask) const {
        LOG_TIMER("SelectionService::applyFilters");
        if (!scene_manager_ || !rendering_manager_ || !selection.is_valid()) {
            return;
        }

        if (nodeMaskRestrictsSelection(node_mask)) {
            LOG_TIMER("applyFilters.filter_selection_by_node_mask");
            if (const auto transform_indices = scene_manager_->getScene().getTransformIndices();
                transform_indices && transform_indices->is_valid()) {
                rendering::filter_selection_by_node_mask(selection, *transform_indices, node_mask);
            }
        }

        if (filters.crop_filter) {
            applyCropFilter(selection);
        }
        if (filters.depth_filter) {
            applyDepthFilter(selection);
        }
    }

    void SelectionService::applyCropFilter(core::Tensor& selection,
                                           const core::Tensor* crop_box_transform,
                                           const core::Tensor* crop_box_min,
                                           const core::Tensor* crop_box_max,
                                           const core::Tensor* ellipsoid_transform,
                                           const core::Tensor* ellipsoid_radii,
                                           const bool use_scene_filters) const {
        LOG_TIMER("SelectionService::applyCropFilter");
        if (!scene_manager_ || !selection.is_valid()) {
            return;
        }

        auto render_lock = [&] {
            LOG_TIMER("applyCropFilter.acquireLiveModelRenderLock");
            return acquireLiveModelRenderLock(scene_manager_);
        }();
        const auto* const model = scene_manager_->getModelForRendering();
        if (!model || model->size() == 0) {
            return;
        }

        const auto& means = model->means();
        if (!means.is_valid() || means.size(0) != selection.size(0)) {
            return;
        }

        core::Tensor crop_t;
        core::Tensor crop_min;
        core::Tensor crop_max;
        const core::Tensor* crop_t_ptr = nullptr;
        const core::Tensor* crop_min_ptr = nullptr;
        const core::Tensor* crop_max_ptr = nullptr;
        bool crop_inverse = false;

        const auto render_state = [&] {
            LOG_TIMER("applyCropFilter.buildRenderState");
            return scene_manager_->buildRenderState();
        }();
        if (crop_box_transform && crop_box_min && crop_box_max) {
            crop_t_ptr = crop_box_transform;
            crop_min_ptr = crop_box_min;
            crop_max_ptr = crop_box_max;
        } else if (use_scene_filters) {
            if (const auto* const cb =
                    findRenderableByNodeId(render_state.cropboxes, scene_manager_->getActiveSelectionCropBoxId());
                cb && cb->data) {
                const glm::mat4 inv_transform = glm::inverse(cb->world_transform);
                const float* const t_ptr = glm::value_ptr(inv_transform);
                crop_t = core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4});
                crop_min = core::Tensor::from_vector({cb->data->min.x, cb->data->min.y, cb->data->min.z}, {3});
                crop_max = core::Tensor::from_vector({cb->data->max.x, cb->data->max.y, cb->data->max.z}, {3});
                crop_t_ptr = &crop_t;
                crop_min_ptr = &crop_min;
                crop_max_ptr = &crop_max;
                crop_inverse = cb->data->inverse;
            }
        }

        core::Tensor ellip_t;
        core::Tensor ellip_radii;
        const core::Tensor* ellip_t_ptr = nullptr;
        const core::Tensor* ellip_radii_ptr = nullptr;
        bool ellipsoid_inverse = false;

        if (ellipsoid_transform && ellipsoid_radii) {
            ellip_t_ptr = ellipsoid_transform;
            ellip_radii_ptr = ellipsoid_radii;
        } else if (use_scene_filters) {
            if (const auto* const el =
                    findRenderableByNodeId(render_state.ellipsoids, scene_manager_->getActiveSelectionEllipsoidId());
                el && el->data) {
                const glm::mat4 inv_transform = glm::inverse(el->world_transform);
                const float* const t_ptr = glm::value_ptr(inv_transform);
                ellip_t = core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4});
                ellip_radii = core::Tensor::from_vector({el->data->radii.x, el->data->radii.y, el->data->radii.z}, {3});
                ellip_t_ptr = &ellip_t;
                ellip_radii_ptr = &ellip_radii;
                ellipsoid_inverse = el->data->inverse;
            }
        }

        if (!crop_t_ptr && !ellip_t_ptr) {
            return;
        }

        core::Tensor model_transforms_cuda;
        const core::Tensor* model_transforms_ptr = nullptr;
        if (!render_state.model_transforms.empty()) {
            LOG_TIMER("applyCropFilter.uploadModelTransformsToCuda");
            model_transforms_cuda = uploadModelTransformsToCuda(render_state.model_transforms);
            model_transforms_ptr = &model_transforms_cuda;
        }

        core::Tensor transform_indices_cuda;
        const core::Tensor* transform_indices_ptr = nullptr;
        if (render_state.transform_indices && render_state.transform_indices->is_valid() &&
            render_state.transform_indices->numel() == means.size(0)) {
            if (render_state.transform_indices->device() == core::Device::CUDA) {
                transform_indices_ptr = render_state.transform_indices.get();
            } else {
                LOG_TIMER("applyCropFilter.transform_indices_to_cuda");
                transform_indices_cuda = render_state.transform_indices->cuda();
                transform_indices_ptr = &transform_indices_cuda;
            }
        }

        {
            LOG_TIMER("applyCropFilter.filter_selection_by_crop");
            rendering::filter_selection_by_crop(
                selection, means,
                crop_t_ptr,
                crop_min_ptr,
                crop_max_ptr,
                crop_inverse,
                ellip_t_ptr,
                ellip_radii_ptr,
                ellipsoid_inverse,
                model_transforms_ptr,
                transform_indices_ptr);
        }
    }

    void SelectionService::applyDepthFilter(core::Tensor& selection) const {
        LOG_TIMER("SelectionService::applyDepthFilter");
        if (!scene_manager_ || !rendering_manager_ || !selection.is_valid()) {
            return;
        }

        auto render_lock = [&] {
            LOG_TIMER("applyDepthFilter.acquireLiveModelRenderLock");
            return acquireLiveModelRenderLock(scene_manager_);
        }();
        const auto* const model = scene_manager_->getModelForRendering();
        if (!model || model->size() == 0) {
            return;
        }

        const auto settings = rendering_manager_->getSettings();
        if (!settings.depth_filter_enabled) {
            return;
        }

        const auto& means = model->means();
        if (!means.is_valid() || means.size(0) != selection.size(0)) {
            return;
        }

        const glm::mat4 world_to_filter = settings.depth_filter_transform.inv().toMat4();
        const float* const t_ptr = glm::value_ptr(world_to_filter);
        const auto depth_t = core::Tensor::from_vector(std::vector<float>(t_ptr, t_ptr + 16), {4, 4});
        const auto depth_min = core::Tensor::from_vector(
            {settings.depth_filter_min.x, settings.depth_filter_min.y, settings.depth_filter_min.z}, {3});
        const auto depth_max = core::Tensor::from_vector(
            {settings.depth_filter_max.x, settings.depth_filter_max.y, settings.depth_filter_max.z}, {3});

        const auto render_state = [&] {
            LOG_TIMER("applyDepthFilter.buildRenderState");
            return scene_manager_->buildRenderState();
        }();
        core::Tensor model_transforms_cuda;
        const core::Tensor* model_transforms_ptr = nullptr;
        if (!render_state.model_transforms.empty()) {
            LOG_TIMER("applyDepthFilter.uploadModelTransformsToCuda");
            model_transforms_cuda = uploadModelTransformsToCuda(render_state.model_transforms);
            model_transforms_ptr = &model_transforms_cuda;
        }

        core::Tensor transform_indices_cuda;
        const core::Tensor* transform_indices_ptr = nullptr;
        if (render_state.transform_indices && render_state.transform_indices->is_valid() &&
            render_state.transform_indices->numel() == means.size(0)) {
            if (render_state.transform_indices->device() == core::Device::CUDA) {
                transform_indices_ptr = render_state.transform_indices.get();
            } else {
                LOG_TIMER("applyDepthFilter.transform_indices_to_cuda");
                transform_indices_cuda = render_state.transform_indices->cuda();
                transform_indices_ptr = &transform_indices_cuda;
            }
        }

        {
            LOG_TIMER("applyDepthFilter.filter_selection_by_crop");
            rendering::filter_selection_by_crop(
                selection, means,
                &depth_t, &depth_min, &depth_max, false,
                nullptr, nullptr, false,
                model_transforms_ptr,
                transform_indices_ptr);
        }
    }

    void SelectionService::clearInteractivePreviewState() {
        if (rendering_manager_) {
            rendering_manager_->clearSelectionPreviews();
            rendering_manager_->markDirty(DirtyFlag::SELECTION);
        }
    }

    std::vector<bool> SelectionService::effectiveNodeMask(const bool restrict_to_selected_nodes) const {
        if (!scene_manager_ || !restrict_to_selected_nodes || !scene_manager_->hasSelectedNode()) {
            return {};
        }
        return scene_manager_->getSelectedNodeMask();
    }

    SelectionFilterState SelectionService::defaultFilterState() const {
        SelectionFilterState filters{};
        if (!rendering_manager_) {
            return filters;
        }

        const auto settings = rendering_manager_->getSettings();
        filters.crop_filter = settings.crop_filter_for_selection;
        filters.depth_filter = settings.depth_filter_enabled;
        filters.restrict_to_selected_nodes = true;
        return filters;
    }

} // namespace lfs::vis
