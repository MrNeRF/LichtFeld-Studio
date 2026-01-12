/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::rendering {

    class CameraFrustumRenderer {
    public:
        static constexpr int THUMBNAIL_SIZE = 128;

        CameraFrustumRenderer() = default;
        ~CameraFrustumRenderer();

        Result<void> init();
        Result<void> render(const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
                            const glm::mat4& view,
                            const glm::mat4& projection,
                            float scale = 0.1f,
                            const glm::vec3& train_color = glm::vec3(0.0f, 1.0f, 0.0f),
                            const glm::vec3& eval_color = glm::vec3(1.0f, 0.0f, 0.0f),
                            const glm::mat4& scene_transform = glm::mat4(1.0f),
                            bool equirectangular_view = false);

        Result<int> pickCamera(const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
                               const glm::vec2& mouse_pos,
                               const glm::vec2& viewport_pos,
                               const glm::vec2& viewport_size,
                               const glm::mat4& view,
                               const glm::mat4& projection,
                               float scale = 0.1f,
                               const glm::mat4& scene_transform = glm::mat4(1.0f));

        void setHighlightedCamera(const int index) { highlighted_camera_ = index; }
        [[nodiscard]] int getHighlightedCamera() const { return highlighted_camera_; }

        void setShowImages(const bool show) { show_images_ = show; }
        [[nodiscard]] bool getShowImages() const { return show_images_; }

        void setImageOpacity(const float opacity) { image_opacity_ = std::clamp(opacity, 0.0f, 1.0f); }
        [[nodiscard]] float getImageOpacity() const { return image_opacity_; }

        [[nodiscard]] bool isInitialized() const { return initialized_; }

        void clearThumbnailCache();

    private:
        struct Vertex {
            glm::vec3 position;
            glm::vec2 uv;
        };

        struct InstanceData {
            glm::mat4 transform;
            glm::vec3 color;
            float alpha;
            uint32_t texture_id;
            uint32_t is_validation;
            uint32_t is_equirectangular;
            uint32_t padding; // 16-byte alignment
        };

        struct ThumbnailRequest {
            int camera_uid;
            std::filesystem::path image_path;
            int image_width;
            int image_height;
        };

        struct LoadedThumbnail {
            int camera_uid;
            std::vector<uint8_t> pixel_data;
            int width;
            int height;
        };

        Result<void> createGeometry();
        Result<void> createSphereGeometry();
        Result<void> createPickingFBO();

        void prepareInstances(const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
                              float scale,
                              const glm::vec3& train_color,
                              const glm::vec3& eval_color,
                              bool for_picking,
                              const glm::vec3& view_position,
                              const glm::mat4& scene_transform);

        void updateInstanceVisibility(const glm::vec3& view_position);

        // Thumbnail loading
        [[nodiscard]] GLuint getOrLoadThumbnail(const lfs::core::Camera& camera);
        void startThumbnailLoader();
        void stopThumbnailLoader();
        void thumbnailLoaderWorker();
        void queueThumbnailLoad(const lfs::core::Camera& camera);
        void uploadReadyThumbnails();

        // GL resources - frustum geometry
        ManagedShader shader_;
        ManagedShader shader_lines_;
        VAO vao_;
        VBO vbo_;
        EBO face_ebo_;
        EBO edge_ebo_;
        VBO instance_vbo_;

        // Sphere geometry for equirectangular cameras
        VAO sphere_vao_;
        VBO sphere_vbo_;
        EBO sphere_face_ebo_;
        EBO sphere_edge_ebo_;
        size_t num_sphere_face_indices_ = 0;
        size_t num_sphere_edge_indices_ = 0;

        // Picking
        FBO picking_fbo_;
        Texture picking_color_texture_;
        Texture picking_depth_texture_;
        int picking_fbo_width_ = 0;
        int picking_fbo_height_ = 0;

        // Instance data
        std::vector<InstanceData> cached_instances_;
        std::vector<int> camera_ids_;
        std::vector<glm::vec3> camera_positions_;

        size_t num_face_indices_ = 0;
        size_t num_edge_indices_ = 0;
        int highlighted_camera_ = -1;
        bool initialized_ = false;

        // Cache invalidation
        float last_scale_ = -1.0f;
        glm::vec3 last_train_color_{-1, -1, -1};
        glm::vec3 last_eval_color_{-1, -1, -1};
        glm::vec3 last_view_position_{0, 0, 0};
        glm::mat4 last_scene_transform_{1.0f};

        // Image preview
        bool show_images_ = true;
        float image_opacity_ = 0.8f;

        // Texture array for batched thumbnail rendering
        Texture thumbnail_array_;
        int thumbnail_array_capacity_ = 0;
        int thumbnail_array_count_ = 0;
        std::unordered_map<int, int> uid_to_layer_; // camera_uid -> array layer index
        std::unordered_set<int> thumbnail_pending_;
        std::mutex pending_mutex_;

        // Async loading
        std::queue<ThumbnailRequest> thumbnail_load_queue_;
        std::mutex load_queue_mutex_;
        std::condition_variable load_queue_cv_;

        std::queue<LoadedThumbnail> thumbnail_ready_queue_;
        std::mutex ready_queue_mutex_;

        std::thread thumbnail_loader_thread_;
        std::atomic<bool> thumbnail_loader_running_{false};
    };

} // namespace lfs::rendering
