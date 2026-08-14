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
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/vulkan_scene_upscaler_adapter.hpp"
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::gui {

    enum class VideoExportUpscalerExecution : uint8_t {
        Native,
        SpatialTensor,
        TemporalVulkanResolve,
        VulkanAdapter,
    };

    struct VideoExportUpscalerResources {
        bool vulkan_color = false;
        bool depth = false;
        bool motion_vectors = false;
        bool jitter = false;
        bool history = false;
        bool reactive_mask = false;
        bool exposure = false;
    };

    struct VideoExportVulkanFrameInputs {
        VulkanSceneUpscalerResource color;
        VulkanSceneUpscalerResource depth;
        VulkanSceneUpscalerResource motion;
        glm::ivec2 output_extent{0, 0};
        glm::vec2 jitter_pixels{0.0f};
        glm::vec2 previous_jitter_pixels{0.0f};
        bool history_valid = false;
        float exposure = 1.0f;
        VkSemaphore completion_semaphore = VK_NULL_HANDLE;
        std::uint64_t completion_value = 0;
    };

    struct VideoExportUpscalerContract {
        std::string id;
        SceneUpscalerRequirements requirements;
        VideoExportUpscalerExecution execution = VideoExportUpscalerExecution::Native;
        bool optional = false;
        bool lazy_adapter = false;
    };

    enum class VideoExportUpscalerResourceIssue : uint8_t {
        None,
        VulkanColor,
        Depth,
        MotionVectors,
        Jitter,
        History,
        ReactiveMask,
        Exposure,
    };

    struct VideoExportRenderPlan {
        int output_width = 0;
        int output_height = 0;
        int input_width = 0;
        int input_height = 0;
        bool requires_upscale = false;
        std::string backend;
        int quality = 1;

        bool operator==(const VideoExportRenderPlan&) const = default;
    };

    struct VideoExportMeshSnapshot {
        std::shared_ptr<lfs::core::MeshData> mesh;
        lfs::core::NodeId node_id = lfs::core::NULL_NODE;
        glm::mat4 transform{1.0f};
        bool is_selected = false;
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
        const lfs::vis::SceneManager& scene_manager,
        lfs::io::video::VideoSplatPrecision precision =
            lfs::io::video::VideoSplatPrecision::Float32);

    LFS_VIS_API void refreshVideoExportMeshTransforms(
        VideoExportSceneSnapshot& snapshot,
        const lfs::core::Scene& scene);

    LFS_VIS_API std::expected<lfs::io::video::VideoExportOptions, std::string> validateVideoExportOptions(
        lfs::io::video::VideoExportOptions options);

    [[nodiscard]] LFS_VIS_API std::expected<VideoExportRenderPlan, std::string>
    makeVideoExportRenderPlan(const lfs::io::video::VideoExportOptions& options);

    [[nodiscard]] LFS_VIS_API std::expected<VideoExportRenderPlan, std::string>
    resolveVideoExportRenderPlan(const lfs::io::video::VideoExportOptions& options,
                                 bool supports_reconstruction);

    [[nodiscard]] LFS_VIS_API int videoExportTemporalSampleCount(
        const lfs::io::video::VideoExportOptions& options);

    [[nodiscard]] LFS_VIS_API std::vector<float> videoExportSampleTimes(
        float frame_time,
        float frame_duration,
        float timeline_start,
        float timeline_end,
        const lfs::io::video::VideoExportOptions& options);

    [[nodiscard]] LFS_VIS_API std::optional<VideoExportUpscalerContract>
    videoExportUpscalerContract(
        std::string_view backend,
        const OptionalSceneUpscalerRegistry& optional_registry = optionalSceneUpscalerRegistry());

    [[nodiscard]] LFS_VIS_API VideoExportUpscalerResourceIssue
    validateVideoExportUpscalerResources(
        const VideoExportUpscalerContract& contract,
        const VideoExportUpscalerResources& resources);

    [[nodiscard]] LFS_VIS_API std::string_view videoExportUpscalerResourceIssueMessage(
        VideoExportUpscalerResourceIssue issue);

    [[nodiscard]] LFS_VIS_API VideoExportUpscalerResources videoExportUpscalerResources(
        const VideoExportVulkanFrameInputs& inputs);

} // namespace lfs::vis::gui
