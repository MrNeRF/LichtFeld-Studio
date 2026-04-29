/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_frustum_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "io/pipelined_image_loader.hpp"
#include "rendering/coordinate_conventions.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace lfs::rendering {

    namespace {
        constexpr float FADE_START_MULTIPLIER = 5.0f;
        constexpr float FADE_END_MULTIPLIER = 0.2f;
        constexpr float MIN_VISIBLE_MULTIPLIER = 0.1f;
        constexpr float MIN_VISIBLE_ALPHA = 0.05f;
        constexpr float MIN_RENDER_ALPHA = 0.01f;
        constexpr float WIREFRAME_WIDTH = 1.5f;
        constexpr int PICKING_SAMPLE_SIZE = 3;
        constexpr int INITIAL_TEXTURE_ARRAY_CAPACITY = 256;
        constexpr float EQUIRECTANGULAR_DISPLAY_FOV = 1.0472f; // 60 degrees

        [[nodiscard]] std::vector<glm::mat4> resolveSceneTransforms(
            const size_t camera_count,
            const glm::mat4& fallback_scene_transform,
            const std::span<const glm::mat4> scene_transforms) {
            if (scene_transforms.size() == camera_count) {
                return std::vector<glm::mat4>(scene_transforms.begin(), scene_transforms.end());
            }
            return std::vector<glm::mat4>(camera_count, fallback_scene_transform);
        }

    } // namespace

    CameraFrustumRenderer::~CameraFrustumRenderer() {
        stopThumbnailLoader();
        if (picking_pbos_[0]) {
            glDeleteBuffers(2, picking_pbos_);
        }
    }

    void CameraFrustumRenderer::clearThumbnailCache() {
        thumbnail_generation_.fetch_add(1, std::memory_order_acq_rel);
        const std::scoped_lock lock(shared_loader_mutex_, pending_mutex_, load_queue_mutex_, ready_queue_mutex_);

        thumbnail_pending_.clear();
        thumbnail_load_queue_ = {};
        thumbnail_ready_queue_ = {};

        thumbnail_array_ = Texture{};
        thumbnail_array_capacity_ = 0;
        thumbnail_array_count_ = 0;
        uid_to_layer_.clear();

        cached_instances_.clear();
        camera_ids_.clear();
        camera_positions_.clear();
        last_scale_ = -1.0f;
        last_scene_transforms_.clear();
        fallback_loader_.reset();
    }

    Result<void> CameraFrustumRenderer::init() {
        auto shader_result = load_shader_with_geometry("camera_frustum", "camera_frustum.vert",
                                                       "camera_frustum.geom", "camera_frustum.frag", false);
        if (!shader_result) {
            return std::unexpected(shader_result.error().what());
        }
        shader_ = std::move(*shader_result);

        auto lines_shader_result = load_shader_with_geometry("camera_frustum_lines", "camera_frustum.vert",
                                                             "camera_frustum_lines.geom", "camera_frustum.frag", false);
        if (!lines_shader_result) {
            return std::unexpected(lines_shader_result.error().what());
        }
        shader_lines_ = std::move(*lines_shader_result);

        if (auto result = createGeometry(); !result) {
            return result;
        }

        if (auto result = createSphereGeometry(); !result) {
            return result;
        }

        auto instance_vbo_result = create_vbo();
        if (!instance_vbo_result) {
            return std::unexpected(instance_vbo_result.error());
        }
        instance_vbo_ = std::move(*instance_vbo_result);

        if (auto result = createPickingFBO(); !result) {
            return result;
        }

        startThumbnailLoader();
        initialized_ = true;
        LOG_INFO("Camera frustum renderer initialized");
        return {};
    }

    Result<void> CameraFrustumRenderer::createGeometry() {
        const std::vector<Vertex> vertices = {
            {{-0.5f, -0.5f, -1.0f}, {0.0f, 0.0f}},
            {{0.5f, -0.5f, -1.0f}, {1.0f, 0.0f}},
            {{0.5f, 0.5f, -1.0f}, {1.0f, 1.0f}},
            {{-0.5f, 0.5f, -1.0f}, {0.0f, 1.0f}},
            {{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f}}};

        const std::vector<unsigned int> face_indices = {
            0, 1, 2, 0, 2, 3,                  // Base
            0, 4, 1, 1, 4, 2, 2, 4, 3, 3, 4, 0 // Sides
        };

        const std::vector<unsigned int> edge_indices = {
            0, 1, 1, 2, 2, 3, 3, 0, // Base
            0, 4, 1, 4, 2, 4, 3, 4  // Apex
        };

        num_face_indices_ = face_indices.size();
        num_edge_indices_ = edge_indices.size();

        auto vao_result = create_vao();
        if (!vao_result)
            return std::unexpected(vao_result.error());

        auto vbo_result = create_vbo();
        if (!vbo_result)
            return std::unexpected(vbo_result.error());
        vbo_ = std::move(*vbo_result);

        auto face_ebo_result = create_vbo();
        if (!face_ebo_result)
            return std::unexpected(face_ebo_result.error());
        face_ebo_ = std::move(*face_ebo_result);

        auto edge_ebo_result = create_vbo();
        if (!edge_ebo_result)
            return std::unexpected(edge_ebo_result.error());
        edge_ebo_ = std::move(*edge_ebo_result);

        VAOBuilder builder(std::move(*vao_result));

        const std::span<const float> vertices_data(
            reinterpret_cast<const float*>(vertices.data()),
            vertices.size() * sizeof(Vertex) / sizeof(float));

        builder.attachVBO(vbo_, vertices_data, GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT, .stride = sizeof(Vertex), .offset = nullptr})
            .setAttribute({.index = 1, .size = 2, .type = GL_FLOAT, .stride = sizeof(Vertex), .offset = reinterpret_cast<const void*>(offsetof(Vertex, uv))});

        builder.attachEBO(face_ebo_, std::span(face_indices), GL_STATIC_DRAW);
        vao_ = builder.build();

        BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
        upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(edge_indices), GL_STATIC_DRAW);

        return {};
    }

    Result<void> CameraFrustumRenderer::createSphereGeometry() {
        constexpr int LAT_SEGS = 16;
        constexpr int LON_SEGS = 24;
        constexpr float RADIUS = 0.5f;

        std::vector<Vertex> vertices;
        std::vector<unsigned int> face_indices;
        std::vector<unsigned int> edge_indices;

        for (int lat = 0; lat <= LAT_SEGS; ++lat) {
            const float theta = static_cast<float>(lat) / LAT_SEGS * glm::pi<float>();
            const float sin_t = std::sin(theta);
            const float cos_t = std::cos(theta);

            for (int lon = 0; lon <= LON_SEGS; ++lon) {
                const float phi = static_cast<float>(lon) / LON_SEGS * 2.0f * glm::pi<float>();
                const glm::vec3 pos(RADIUS * sin_t * std::sin(phi), RADIUS * cos_t, -RADIUS * sin_t * std::cos(phi));
                const glm::vec2 uv(static_cast<float>(lon) / LON_SEGS, 1.0f - static_cast<float>(lat) / LAT_SEGS);
                vertices.push_back({pos, uv});
            }
        }

        vertices.push_back({{0.0f, 0.0f, 0.0f}, {0.5f, 0.5f}});
        const unsigned int apex_idx = static_cast<unsigned int>(vertices.size() - 1);

        for (int lat = 0; lat < LAT_SEGS; ++lat) {
            for (int lon = 0; lon < LON_SEGS; ++lon) {
                const unsigned int curr = lat * (LON_SEGS + 1) + lon;
                const unsigned int next = curr + LON_SEGS + 1;
                face_indices.insert(face_indices.end(), {curr, next, curr + 1, curr + 1, next, next + 1});
            }
        }

        for (int lat = 0; lat <= LAT_SEGS; lat += 4) {
            for (int lon = 0; lon < LON_SEGS; ++lon) {
                const unsigned int curr = lat * (LON_SEGS + 1) + lon;
                edge_indices.insert(edge_indices.end(), {curr, curr + 1});
            }
        }
        for (int lon = 0; lon < LON_SEGS; lon += 4) {
            for (int lat = 0; lat < LAT_SEGS; ++lat) {
                const unsigned int curr = lat * (LON_SEGS + 1) + lon;
                edge_indices.insert(edge_indices.end(), {curr, curr + LON_SEGS + 1});
            }
        }
        for (int lon = 0; lon < LON_SEGS; lon += LON_SEGS / 4) {
            const unsigned int eq_idx = (LAT_SEGS / 2) * (LON_SEGS + 1) + lon;
            edge_indices.insert(edge_indices.end(), {apex_idx, eq_idx});
        }

        num_sphere_face_indices_ = face_indices.size();
        num_sphere_edge_indices_ = edge_indices.size();

        auto vao_result = create_vao();
        if (!vao_result)
            return std::unexpected(vao_result.error());

        auto vbo_result = create_vbo();
        if (!vbo_result)
            return std::unexpected(vbo_result.error());
        sphere_vbo_ = std::move(*vbo_result);

        auto face_ebo_result = create_vbo();
        if (!face_ebo_result)
            return std::unexpected(face_ebo_result.error());
        sphere_face_ebo_ = std::move(*face_ebo_result);

        auto edge_ebo_result = create_vbo();
        if (!edge_ebo_result)
            return std::unexpected(edge_ebo_result.error());
        sphere_edge_ebo_ = std::move(*edge_ebo_result);

        VAOBuilder builder(std::move(*vao_result));
        const std::span<const float> vertices_data(
            reinterpret_cast<const float*>(vertices.data()),
            vertices.size() * sizeof(Vertex) / sizeof(float));

        builder.attachVBO(sphere_vbo_, vertices_data, GL_STATIC_DRAW)
            .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT, .stride = sizeof(Vertex), .offset = nullptr})
            .setAttribute({.index = 1, .size = 2, .type = GL_FLOAT, .stride = sizeof(Vertex), .offset = reinterpret_cast<const void*>(offsetof(Vertex, uv))});

        builder.attachEBO(sphere_face_ebo_, std::span(face_indices), GL_STATIC_DRAW);
        sphere_vao_ = builder.build();

        BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(sphere_edge_ebo_);
        upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(edge_indices), GL_STATIC_DRAW);

        return {};
    }

    Result<void> CameraFrustumRenderer::createPickingFBO() {
        GLuint fbo_id;
        glGenFramebuffers(1, &fbo_id);
        if (fbo_id == 0) {
            return std::unexpected("Failed to create picking FBO");
        }
        picking_fbo_ = FBO(fbo_id);

        picking_fbo_width_ = 256;
        picking_fbo_height_ = 256;

        GLuint color_tex;
        glGenTextures(1, &color_tex);
        picking_color_texture_ = Texture(color_tex);

        glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        GLuint depth_tex;
        glGenTextures(1, &depth_tex);
        picking_depth_texture_ = Texture(depth_tex);

        glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, picking_color_texture_, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, picking_depth_texture_, 0);

        const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        if (status != GL_FRAMEBUFFER_COMPLETE) {
            return std::unexpected("Picking FBO incomplete");
        }

        glGenBuffers(2, picking_pbos_);
        for (auto& pbo : picking_pbos_) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
            glBufferData(GL_PIXEL_PACK_BUFFER,
                         PICKING_SAMPLE_SIZE * PICKING_SAMPLE_SIZE * 3 * sizeof(float),
                         nullptr, GL_STREAM_READ);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        return {};
    }

    void CameraFrustumRenderer::prepareInstances(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        const std::span<const glm::vec3> per_camera_colors,
        const bool for_picking,
        const glm::vec3& view_position,
        const glm::mat4& scene_transform,
        const std::span<const glm::mat4> scene_transforms,
        const std::unordered_set<int>& disabled_uids,
        const std::unordered_set<int>& emphasized_uids) {

        const auto resolved_scene_transforms =
            resolveSceneTransforms(cameras.size(), scene_transform, scene_transforms);

        const bool needs_regeneration =
            cached_instances_.size() != cameras.size() ||
            last_scale_ != scale ||
            last_scene_transforms_ != resolved_scene_transforms;

        if (needs_regeneration) {
            cached_instances_.clear();
            cached_instances_.reserve(cameras.size());
            camera_ids_.clear();
            camera_ids_.reserve(cameras.size());
            camera_positions_.clear();
            camera_positions_.reserve(cameras.size());

            for (size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
                const auto& cam = cameras[camera_index];
                if (!cam) {
                    continue;
                }
                auto R_tensor = cam->R();
                auto T_tensor = cam->T();

                if (!R_tensor.is_valid() || !T_tensor.is_valid())
                    continue;

                if (R_tensor.device() != lfs::core::Device::CPU)
                    R_tensor = R_tensor.cpu();
                if (T_tensor.device() != lfs::core::Device::CPU)
                    T_tensor = T_tensor.cpu();

                glm::mat4 w2c(1.0f);
                auto R_acc = R_tensor.accessor<float, 2>();
                auto T_acc = T_tensor.accessor<float, 1>();

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        w2c[j][i] = R_acc(i, j);
                    }
                    w2c[3][i] = T_acc(i);
                }

                const glm::mat4 transformed_c2w =
                    resolved_scene_transforms[camera_index] * glm::inverse(w2c);
                const glm::mat4 visualizer_c2w = transformed_c2w * DATA_TO_VISUALIZER_CAMERA_AXES_4;
                const glm::vec3 cam_pos = glm::vec3(visualizer_c2w[3]);
                camera_positions_.push_back(cam_pos);

                const float aspect = static_cast<float>(cam->image_width()) / static_cast<float>(cam->image_height());
                const bool is_equirect = cam->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
                const float fov_y = is_equirect ? EQUIRECTANGULAR_DISPLAY_FOV
                                                : lfs::core::focal2fov(cam->focal_y(), cam->image_height());
                const float half_height = std::tan(fov_y * 0.5f);
                const float half_width = half_height * aspect;

                const glm::mat4 fov_scale = glm::scale(
                    glm::mat4(1.0f),
                    glm::vec3(half_width * 2.0f * scale, half_height * 2.0f * scale, scale));
                const glm::mat4 model = visualizer_c2w * fov_scale;

                cached_instances_.push_back(
                    {model, train_color, 1.0f, 0, 0u, is_equirect ? 1u : 0u, 0u, 0u});
                camera_ids_.push_back(cam->uid());
            }
        }

        updateInstanceAppearance(cameras, train_color, eval_color, per_camera_colors,
                                 disabled_uids, emphasized_uids);

        if (!for_picking) {
            updateInstanceVisibility(view_position);
        }

        last_scale_ = scale;
        last_train_color_ = train_color;
        last_eval_color_ = eval_color;
        last_scene_transforms_ = resolved_scene_transforms;
    }

    void CameraFrustumRenderer::updateInstanceAppearance(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        const std::span<const glm::vec3> per_camera_colors,
        const std::unordered_set<int>& disabled_uids,
        const std::unordered_set<int>& emphasized_uids) {

        const bool has_overrides = per_camera_colors.size() == cameras.size();
        size_t instance_index = 0;

        for (size_t i = 0; i < cameras.size() && instance_index < cached_instances_.size(); ++i) {
            const auto& cam = cameras[i];
            if (!cam) {
                continue;
            }
            auto R_tensor = cam->R();
            auto T_tensor = cam->T();
            if (!R_tensor.is_valid() || !T_tensor.is_valid()) {
                continue;
            }
            const bool is_validation = cam->image_name().find("test") != std::string::npos;

            glm::vec3 color = is_validation ? eval_color : train_color;
            if (has_overrides) {
                const glm::vec3 override_color = per_camera_colors[i];
                if (std::isfinite(override_color.x) &&
                    std::isfinite(override_color.y) &&
                    std::isfinite(override_color.z)) {
                    color = override_color;
                }
            }

            cached_instances_[instance_index].color = color;
            cached_instances_[instance_index].is_validation = is_validation ? 1u : 0u;
            cached_instances_[instance_index].is_training_disabled =
                disabled_uids.count(cam->uid()) > 0 ? 1u : 0u;
            cached_instances_[instance_index].is_emphasized =
                emphasized_uids.count(cam->uid()) > 0 ? 1u : 0u;
            ++instance_index;
        }
    }

    void CameraFrustumRenderer::updateInstanceVisibility(const glm::vec3& view_position) {
        if (camera_positions_.size() != cached_instances_.size())
            return;

        const float fade_start = FADE_START_MULTIPLIER * last_scale_;
        const float fade_end = FADE_END_MULTIPLIER * last_scale_;
        const float min_visible = MIN_VISIBLE_MULTIPLIER * last_scale_;

        for (size_t i = 0; i < camera_positions_.size(); ++i) {
            const float distance = glm::length(camera_positions_[i] - view_position);
            float alpha = 1.0f;

            if (distance < min_visible) {
                alpha = 0.0f;
            } else if (distance < fade_end) {
                alpha = MIN_VISIBLE_ALPHA;
            } else if (distance < fade_start) {
                const float t = (distance - fade_end) / (fade_start - fade_end);
                alpha = MIN_VISIBLE_ALPHA + (1.0f - MIN_VISIBLE_ALPHA) * (t * t * (3.0f - 2.0f * t));
            }
            if (cached_instances_[i].is_training_disabled > 0u)
                alpha *= 0.4f;
            cached_instances_[i].alpha = alpha;
        }
    }

    Result<void> CameraFrustumRenderer::render(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const glm::mat4& view,
        const glm::mat4& projection,
        const float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        const std::span<const glm::vec3> per_camera_colors,
        const glm::mat4& scene_transform,
        const std::span<const glm::mat4> scene_transforms,
        const bool equirectangular_view,
        const std::unordered_set<int>& disabled_uids,
        const std::unordered_set<int>& emphasized_uids) {

        if (!initialized_ || cameras.empty())
            return {};

        uploadReadyThumbnails();

        const glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);
        prepareInstances(cameras, scale, train_color, eval_color, per_camera_colors, false,
                         view_position, scene_transform, scene_transforms,
                         disabled_uids, emphasized_uids);

        if (cached_instances_.empty())
            return {};

        std::vector<InstanceData> frustum_instances;
        std::vector<InstanceData> sphere_instances;
        std::vector<int> frustum_indices;
        std::vector<int> sphere_indices;
        std::vector<std::shared_ptr<const lfs::core::Camera>> frustum_cameras;
        std::vector<std::shared_ptr<const lfs::core::Camera>> sphere_cameras;

        for (size_t i = 0; i < cached_instances_.size(); ++i) {
            if (cached_instances_[i].alpha > MIN_RENDER_ALPHA) {
                if (cached_instances_[i].is_equirectangular) {
                    sphere_instances.push_back(cached_instances_[i]);
                    sphere_indices.push_back(static_cast<int>(i));
                    if (i < cameras.size())
                        sphere_cameras.push_back(cameras[i]);
                } else {
                    frustum_instances.push_back(cached_instances_[i]);
                    frustum_indices.push_back(static_cast<int>(i));
                    if (i < cameras.size())
                        frustum_cameras.push_back(cameras[i]);
                }
            }
        }

        if (frustum_instances.empty() && sphere_instances.empty())
            return {};

        GLStateGuard state_guard;
        while (glGetError() != GL_NO_ERROR) {}

        const auto setupInstanceAttributes = [this](std::vector<InstanceData>& instances) {
            BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
            upload_buffer(GL_ARRAY_BUFFER, std::span(instances), GL_DYNAMIC_DRAW);

            for (int i = 0; i < 4; ++i) {
                glEnableVertexAttribArray(2 + i);
                glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                      reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                glVertexAttribDivisor(2 + i, 1);
            }

            glEnableVertexAttribArray(6);
            glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                  reinterpret_cast<void*>(offsetof(InstanceData, color)));
            glVertexAttribDivisor(6, 1);

            glEnableVertexAttribArray(7);
            glVertexAttribIPointer(7, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, texture_id)));
            glVertexAttribDivisor(7, 1);

            glEnableVertexAttribArray(8);
            glVertexAttribIPointer(8, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_validation)));
            glVertexAttribDivisor(8, 1);

            glEnableVertexAttribArray(9);
            glVertexAttribIPointer(9, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_equirectangular)));
            glVertexAttribDivisor(9, 1);

            glEnableVertexAttribArray(10);
            glVertexAttribIPointer(10, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_training_disabled)));
            glVertexAttribDivisor(10, 1);

            glEnableVertexAttribArray(11);
            glVertexAttribIPointer(11, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_emphasized)));
            glVertexAttribDivisor(11, 1);
        };

        const auto cleanupInstanceAttributes = []() {
            for (int i = 2; i <= 11; ++i) {
                glDisableVertexAttribArray(i);
                glVertexAttribDivisor(i, 0);
            }
        };

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const glm::mat4 view_proj = projection * view;

        const auto findFocusedIndex = [this](const std::vector<int>& indices) -> int {
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] == focused_camera_)
                    return static_cast<int>(i);
            }
            return -1;
        };

        const int frustum_focus = findFocusedIndex(frustum_indices);
        const int sphere_focus = findFocusedIndex(sphere_indices);

        if (show_images_) {
            for (size_t i = 0; i < frustum_instances.size() && i < frustum_cameras.size(); ++i)
                frustum_instances[i].texture_id = getOrLoadThumbnail(*frustum_cameras[i]);
            for (size_t i = 0; i < sphere_instances.size() && i < sphere_cameras.size(); ++i)
                sphere_instances[i].texture_id = getOrLoadThumbnail(*sphere_cameras[i]);
        }

        const bool render_textures = show_images_ && thumbnail_array_capacity_ > 0;

        if (render_textures) {
            ShaderScope shader(shader_);
            if (!shader.isBound())
                return std::unexpected("Failed to bind camera frustum shader");

            shader->set("viewProj", view_proj);
            shader->set("view", view);
            shader->set("viewPos", view_position);
            shader->set("pickingMode", false);
            shader->set("equirectangularView", equirectangular_view);
            shader->set("showImages", true);
            shader->set("imageOpacity", image_opacity_);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D_ARRAY, thumbnail_array_);
            shader->set("cameraTextures", 0);

            if (!frustum_instances.empty()) {
                shader->set("focusIndex", frustum_focus);
                VAOBinder vao_bind(vao_);
                setupInstanceAttributes(frustum_instances);
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr,
                                        static_cast<GLsizei>(frustum_instances.size()));
                cleanupInstanceAttributes();
            }

            if (!sphere_instances.empty()) {
                shader->set("focusIndex", sphere_focus);
                VAOBinder vao_bind(sphere_vao_);
                setupInstanceAttributes(sphere_instances);
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(sphere_face_ebo_);
                glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(num_sphere_face_indices_),
                                        GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(sphere_instances.size()));
                cleanupInstanceAttributes();
            }

            glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        }

        {
            ShaderScope shader(shader_lines_);
            if (!shader.isBound())
                return std::unexpected("Failed to bind camera frustum lines shader");

            shader->set("viewProj", view_proj);
            shader->set("view", view);
            shader->set("viewPos", view_position);
            shader->set("pickingMode", false);
            shader->set("equirectangularView", equirectangular_view);
            shader->set("showImages", false);

            glLineWidth(WIREFRAME_WIDTH);

            if (!frustum_instances.empty()) {
                shader->set("focusIndex", frustum_focus);
                VAOBinder vao_bind(vao_);
                setupInstanceAttributes(frustum_instances);
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
                glDrawElementsInstanced(GL_LINES, static_cast<GLsizei>(num_edge_indices_),
                                        GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(frustum_instances.size()));
                cleanupInstanceAttributes();
            }

            if (!sphere_instances.empty()) {
                shader->set("focusIndex", sphere_focus);
                VAOBinder vao_bind(sphere_vao_);
                setupInstanceAttributes(sphere_instances);
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(sphere_edge_ebo_);
                glDrawElementsInstanced(GL_LINES, static_cast<GLsizei>(num_sphere_edge_indices_),
                                        GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(sphere_instances.size()));
                cleanupInstanceAttributes();
            }
        }

        return {};
    }

    Result<int> CameraFrustumRenderer::pickCamera(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const glm::vec2& mouse_pos,
        const glm::vec2& viewport_pos,
        const glm::vec2& viewport_size,
        const glm::mat4& view,
        const glm::mat4& projection,
        const float scale,
        const glm::mat4& scene_transform,
        const std::span<const glm::mat4> scene_transforms) {

        if (!initialized_ || cameras.empty())
            return -1;

        const glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);
        prepareInstances(cameras, scale, last_train_color_, last_eval_color_, {},
                         true, view_position, scene_transform, scene_transforms);
        if (cached_instances_.empty())
            return -1;

        const int vp_width = static_cast<int>(viewport_size.x);
        const int vp_height = static_cast<int>(viewport_size.y);

        if (vp_width != picking_fbo_width_ || vp_height != picking_fbo_height_) {
            picking_fbo_width_ = vp_width;
            picking_fbo_height_ = vp_height;

            glBindTexture(GL_TEXTURE_2D, picking_color_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, picking_fbo_width_, picking_fbo_height_, 0, GL_RGB, GL_FLOAT, nullptr);

            glBindTexture(GL_TEXTURE_2D, picking_depth_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, picking_fbo_width_, picking_fbo_height_, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        }

        const int pixel_x = std::clamp(static_cast<int>(mouse_pos.x - viewport_pos.x), 0, picking_fbo_width_ - 1);
        const int pixel_y = std::clamp(static_cast<int>(viewport_size.y - (mouse_pos.y - viewport_pos.y)), 0, picking_fbo_height_ - 1);
        const int read_x = std::max(0, pixel_x - 1);
        const int read_y = std::max(0, pixel_y - 1);
        const int read_width = std::min(PICKING_SAMPLE_SIZE, picking_fbo_width_ - read_x);
        const int read_height = std::min(PICKING_SAMPLE_SIZE, picking_fbo_height_ - read_y);

        GLint current_fbo;
        GLint current_viewport[4];
        GLboolean scissor_was_enabled;
        GLint prev_scissor[4];
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
        glGetIntegerv(GL_VIEWPORT, current_viewport);
        scissor_was_enabled = glIsEnabled(GL_SCISSOR_TEST);
        glGetIntegerv(GL_SCISSOR_BOX, prev_scissor);

        glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
        glViewport(0, 0, picking_fbo_width_, picking_fbo_height_);

        glEnable(GL_SCISSOR_TEST);
        glScissor(read_x, read_y, read_width, read_height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Separate frustum and sphere instances for picking
        std::vector<InstanceData> frustum_pick;
        std::vector<InstanceData> sphere_pick;
        for (const auto& inst : cached_instances_) {
            if (inst.is_equirectangular) {
                sphere_pick.push_back(inst);
            } else {
                frustum_pick.push_back(inst);
            }
        }

        const auto setupPickingAttributes = [this](std::vector<InstanceData>& instances) {
            BufferBinder<GL_ARRAY_BUFFER> instance_bind(instance_vbo_);
            upload_buffer(GL_ARRAY_BUFFER, std::span(instances), GL_DYNAMIC_DRAW);

            for (int i = 0; i < 4; ++i) {
                glEnableVertexAttribArray(2 + i);
                glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                      reinterpret_cast<void*>(sizeof(glm::vec4) * i));
                glVertexAttribDivisor(2 + i, 1);
            }

            glEnableVertexAttribArray(6);
            glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                  reinterpret_cast<void*>(offsetof(InstanceData, color)));
            glVertexAttribDivisor(6, 1);

            glEnableVertexAttribArray(7);
            glVertexAttribIPointer(7, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, texture_id)));
            glVertexAttribDivisor(7, 1);

            glEnableVertexAttribArray(8);
            glVertexAttribIPointer(8, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_validation)));
            glVertexAttribDivisor(8, 1);

            glEnableVertexAttribArray(9);
            glVertexAttribIPointer(9, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_equirectangular)));
            glVertexAttribDivisor(9, 1);

            glEnableVertexAttribArray(10);
            glVertexAttribIPointer(10, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_training_disabled)));
            glVertexAttribDivisor(10, 1);

            glEnableVertexAttribArray(11);
            glVertexAttribIPointer(11, 1, GL_UNSIGNED_INT, sizeof(InstanceData),
                                   reinterpret_cast<void*>(offsetof(InstanceData, is_emphasized)));
            glVertexAttribDivisor(11, 1);
        };

        const auto cleanupPickingAttributes = []() {
            for (int i = 2; i <= 11; ++i) {
                glDisableVertexAttribArray(i);
                glVertexAttribDivisor(i, 0);
            }
        };

        {
            ShaderScope shader(shader_);
            if (!shader.isBound()) {
                glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
                glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);
                return std::unexpected("Failed to bind picking shader");
            }

            const glm::mat4 view_proj = projection * view;
            const glm::vec3 view_pos = glm::vec3(glm::inverse(view)[3]);

            shader->set("viewProj", view_proj);
            shader->set("viewPos", view_pos);
            shader->set("pickingMode", true);
            shader->set("minimumPickDistance", scale * 2.0f);

            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);

            // Draw frustums for picking
            if (!frustum_pick.empty()) {
                VAOBinder vao_bind(vao_);
                setupPickingAttributes(frustum_pick);
                {
                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                    glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(num_face_indices_),
                                            GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(frustum_pick.size()));
                }
                cleanupPickingAttributes();
            }

            // Draw spheres for picking
            if (!sphere_pick.empty()) {
                VAOBinder vao_bind(sphere_vao_);
                setupPickingAttributes(sphere_pick);
                {
                    BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(sphere_face_ebo_);
                    glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(num_sphere_face_indices_),
                                            GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(sphere_pick.size()));
                }
                cleanupPickingAttributes();
            }
        }

        const int write_pbo = pbo_index_;
        const int read_pbo = 1 - pbo_index_;

        glBindBuffer(GL_PIXEL_PACK_BUFFER, picking_pbos_[write_pbo]);
        const int sample_bytes = read_width * read_height * 3 * static_cast<int>(sizeof(float));
        if (read_width != pbo_sample_w_ || read_height != pbo_sample_h_) {
            glBufferData(GL_PIXEL_PACK_BUFFER, sample_bytes, nullptr, GL_STREAM_READ);
        }
        glReadPixels(read_x, read_y, read_width, read_height, GL_RGB, GL_FLOAT, nullptr);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

        int id = -1;
        if (pbo_has_data_) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, picking_pbos_[read_pbo]);
            const auto* pixels = static_cast<const float*>(
                glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));
            if (pixels) {
                id = decodePickId(pixels, pbo_sample_w_, pbo_sample_h_);
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        }

        pbo_sample_w_ = read_width;
        pbo_sample_h_ = read_height;
        pbo_index_ = read_pbo;
        pbo_has_data_ = true;

        if (scissor_was_enabled) {
            glScissor(prev_scissor[0], prev_scissor[1], prev_scissor[2], prev_scissor[3]);
        } else {
            glDisable(GL_SCISSOR_TEST);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
        glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);

        if (id >= 0 && id < static_cast<int>(camera_ids_.size())) {
            return camera_ids_[id];
        }
        return -1;
    }

    int CameraFrustumRenderer::decodePickId(const float* pixels, const int width, const int height) const {
        int center_idx = 0;
        if (width == 3 && height == 3) {
            center_idx = 4 * 3;
        } else if (width >= 2 && height >= 2) {
            center_idx = ((height / 2) * width + (width / 2)) * 3;
        }
        return (static_cast<int>(pixels[center_idx] * 255.0f + 0.5f) << 16 |
                static_cast<int>(pixels[center_idx + 1] * 255.0f + 0.5f) << 8 |
                static_cast<int>(pixels[center_idx + 2] * 255.0f + 0.5f)) -
               1;
    }

    GLuint CameraFrustumRenderer::getOrLoadThumbnail(const lfs::core::Camera& camera) {
        const int uid = camera.uid();

        // Return layer index + 1 (0 means no texture)
        if (const auto it = uid_to_layer_.find(uid); it != uid_to_layer_.end()) {
            return static_cast<GLuint>(it->second + 1);
        }

        const auto& image_path = camera.image_path();
        if (image_path.empty() || !std::filesystem::exists(image_path)) {
            return 0;
        }

        queueThumbnailLoad(camera);
        return 0;
    }

    void CameraFrustumRenderer::startThumbnailLoader() {
        if (thumbnail_loader_running_)
            return;

        thumbnail_loader_running_ = true;
        thumbnail_loader_thread_ = std::thread(&CameraFrustumRenderer::thumbnailLoaderWorker, this);
    }

    void CameraFrustumRenderer::stopThumbnailLoader() {
        if (!thumbnail_loader_running_)
            return;

        thumbnail_loader_running_ = false;
        load_queue_cv_.notify_all();

        if (thumbnail_loader_thread_.joinable()) {
            thumbnail_loader_thread_.join();
        }
    }

    void CameraFrustumRenderer::setImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader,
                                               const bool allow_fallback) {
        std::lock_guard lock(shared_loader_mutex_);
        shared_loader_ = std::move(loader);
        allow_fallback_loader_ = allow_fallback;
        if (shared_loader_ || !allow_fallback_loader_) {
            fallback_loader_.reset();
        }
    }

    void CameraFrustumRenderer::thumbnailLoaderWorker() {
        const auto get_loader = [&]() -> std::shared_ptr<lfs::io::PipelinedImageLoader> {
            std::lock_guard lock(shared_loader_mutex_);
            if (shared_loader_) {
                return shared_loader_;
            }
            if (!allow_fallback_loader_) {
                return {};
            }
            if (!fallback_loader_) {
                lfs::io::PipelinedLoaderConfig config;
                config.io_threads = 0;
                config.cold_process_threads = 0;
                config.max_cache_bytes = 64ULL * 1024 * 1024;
                fallback_loader_ = std::make_shared<lfs::io::PipelinedImageLoader>(config);
            }
            return fallback_loader_;
        };

        while (thumbnail_loader_running_) {
            ThumbnailRequest request;

            {
                std::unique_lock lock(load_queue_mutex_);
                load_queue_cv_.wait(lock, [this] {
                    return !thumbnail_load_queue_.empty() || !thumbnail_loader_running_;
                });

                if (!thumbnail_loader_running_)
                    break;

                if (thumbnail_load_queue_.empty())
                    continue;

                request = std::move(thumbnail_load_queue_.front());
                thumbnail_load_queue_.pop();
            }

            const uint64_t current_generation = thumbnail_generation_.load(std::memory_order_acquire);
            if (request.generation != current_generation) {
                continue;
            }

            LoadedThumbnail loaded;
            loaded.camera_uid = request.camera_uid;

            try {
                auto loader = get_loader();
                if (!loader) {
                    if (request.generation == thumbnail_generation_.load(std::memory_order_acquire)) {
                        std::lock_guard lock(pending_mutex_);
                        thumbnail_pending_.erase(request.camera_uid);
                    }
                    continue;
                }

                lfs::io::LoadParams params;
                params.max_width = THUMBNAIL_SIZE;
                params.output_uint8 = true;

                auto tensor = loader->load_image_immediate(request.image_path, params);
                if (request.generation != thumbnail_generation_.load(std::memory_order_acquire)) {
                    continue;
                }
                assert(tensor.ndim() == 3);

                // [C,H,W] on GPU -> uint8 [H,W,C] on CPU with Y-flip.
                auto hwc = tensor.permute({1, 2, 0}).contiguous();
                if (hwc.dtype() != lfs::core::DataType::UInt8) {
                    hwc = (hwc.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8).contiguous();
                }
                hwc = hwc.cpu().contiguous();

                const int h = static_cast<int>(hwc.shape()[0]);
                const int w = static_cast<int>(hwc.shape()[1]);
                const int ch = static_cast<int>(hwc.shape()[2]);
                assert(ch == 3);

                loaded.width = w;
                loaded.height = h;
                loaded.pixel_data.resize(static_cast<size_t>(w) * h * 3);

                const auto* src = hwc.ptr<uint8_t>();
                const size_t row_bytes = static_cast<size_t>(w) * 3;
                for (int y = 0; y < h; ++y) {
                    std::memcpy(loaded.pixel_data.data() + y * row_bytes,
                                src + (h - 1 - y) * row_bytes, row_bytes);
                }

                {
                    std::lock_guard lock(ready_queue_mutex_);
                    thumbnail_ready_queue_.push(std::move(loaded));
                }

            } catch (const std::exception& e) {
                LOG_WARN("Thumbnail load failed for camera {}: {}", request.camera_uid, e.what());
                if (request.generation == thumbnail_generation_.load(std::memory_order_acquire)) {
                    std::lock_guard lock(pending_mutex_);
                    thumbnail_pending_.erase(request.camera_uid);
                }
            }
        }
    }

    void CameraFrustumRenderer::queueThumbnailLoad(const lfs::core::Camera& camera) {
        const int uid = camera.uid();

        {
            std::lock_guard lock(pending_mutex_);
            if (thumbnail_pending_.contains(uid) || uid_to_layer_.contains(uid)) {
                return;
            }
            thumbnail_pending_.insert(uid);
        }

        ThumbnailRequest request{
            .camera_uid = uid,
            .image_path = camera.image_path(),
            .image_width = camera.image_width(),
            .image_height = camera.image_height(),
            .generation = thumbnail_generation_.load(std::memory_order_acquire)};

        {
            std::lock_guard lock(load_queue_mutex_);
            thumbnail_load_queue_.push(std::move(request));
        }
        load_queue_cv_.notify_one();
    }

    void CameraFrustumRenderer::uploadReadyThumbnails() {
        while (true) {
            LoadedThumbnail loaded;

            {
                std::lock_guard lock(ready_queue_mutex_);
                if (thumbnail_ready_queue_.empty())
                    break;
                loaded = std::move(thumbnail_ready_queue_.front());
                thumbnail_ready_queue_.pop();
            }

            // Initialize texture array if needed
            if (thumbnail_array_capacity_ == 0) {
                GLuint tex_id;
                glGenTextures(1, &tex_id);
                thumbnail_array_ = Texture(tex_id);

                glBindTexture(GL_TEXTURE_2D_ARRAY, thumbnail_array_);
                glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8,
                             THUMBNAIL_SIZE, THUMBNAIL_SIZE, INITIAL_TEXTURE_ARRAY_CAPACITY,
                             0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

                thumbnail_array_capacity_ = INITIAL_TEXTURE_ARRAY_CAPACITY;
                LOG_DEBUG("Created texture array with capacity {}", thumbnail_array_capacity_);
            }

            // Grow texture array if needed
            if (thumbnail_array_count_ >= thumbnail_array_capacity_) {
                const int new_capacity = thumbnail_array_capacity_ * 2;

                GLuint new_tex;
                glGenTextures(1, &new_tex);
                glBindTexture(GL_TEXTURE_2D_ARRAY, new_tex);
                glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB8,
                             THUMBNAIL_SIZE, THUMBNAIL_SIZE, new_capacity,
                             0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

                // Copy existing layers
                glCopyImageSubData(thumbnail_array_, GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                                   new_tex, GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0,
                                   THUMBNAIL_SIZE, THUMBNAIL_SIZE, thumbnail_array_count_);
                glFinish();

                thumbnail_array_ = Texture(new_tex);
                thumbnail_array_capacity_ = new_capacity;
                LOG_DEBUG("Grew texture array to capacity {}", new_capacity);
            }

            // Upload to next available layer (resize to THUMBNAIL_SIZE if needed)
            const int layer = thumbnail_array_count_++;
            glBindTexture(GL_TEXTURE_2D_ARRAY, thumbnail_array_);

            // Resize if dimensions don't match
            if (loaded.width != THUMBNAIL_SIZE || loaded.height != THUMBNAIL_SIZE) {
                std::vector<uint8_t> resized(THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3);
                const float scale_x = static_cast<float>(loaded.width) / THUMBNAIL_SIZE;
                const float scale_y = static_cast<float>(loaded.height) / THUMBNAIL_SIZE;
                for (int y = 0; y < THUMBNAIL_SIZE; ++y) {
                    for (int x = 0; x < THUMBNAIL_SIZE; ++x) {
                        const int src_x = std::min(static_cast<int>(x * scale_x), loaded.width - 1);
                        const int src_y = std::min(static_cast<int>(y * scale_y), loaded.height - 1);
                        const int dst_idx = (y * THUMBNAIL_SIZE + x) * 3;
                        const int src_idx = (src_y * loaded.width + src_x) * 3;
                        resized[dst_idx] = loaded.pixel_data[src_idx];
                        resized[dst_idx + 1] = loaded.pixel_data[src_idx + 1];
                        resized[dst_idx + 2] = loaded.pixel_data[src_idx + 2];
                    }
                }
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer,
                                THUMBNAIL_SIZE, THUMBNAIL_SIZE, 1, GL_RGB, GL_UNSIGNED_BYTE, resized.data());
            } else {
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, layer,
                                THUMBNAIL_SIZE, THUMBNAIL_SIZE, 1, GL_RGB, GL_UNSIGNED_BYTE, loaded.pixel_data.data());
            }
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

            uid_to_layer_[loaded.camera_uid] = layer;

            {
                std::lock_guard lock(pending_mutex_);
                thumbnail_pending_.erase(loaded.camera_uid);
            }
        }
    }

} // namespace lfs::rendering
