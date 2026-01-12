/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_frustum_renderer.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "io/nvcodec_image_loader.hpp"
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
        constexpr int NVCODEC_DECODER_POOL_SIZE = 4;
        constexpr int INITIAL_TEXTURE_ARRAY_CAPACITY = 256;
        constexpr float EQUIRECTANGULAR_DISPLAY_FOV = 1.0472f; // 60 degrees

        const glm::mat4 GL_TO_COLMAP = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
    } // namespace

    CameraFrustumRenderer::~CameraFrustumRenderer() {
        stopThumbnailLoader();
    }

    void CameraFrustumRenderer::clearThumbnailCache() {
        const std::scoped_lock lock(pending_mutex_, load_queue_mutex_, ready_queue_mutex_);

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
                const glm::vec2 uv(static_cast<float>(lon) / LON_SEGS, static_cast<float>(lat) / LAT_SEGS);
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
        return {};
    }

    void CameraFrustumRenderer::prepareInstances(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        const bool for_picking,
        const glm::vec3& view_position,
        const glm::mat4& scene_transform) {

        const bool needs_regeneration =
            cached_instances_.size() != cameras.size() ||
            last_scale_ != scale ||
            last_train_color_ != train_color ||
            last_eval_color_ != eval_color ||
            last_scene_transform_ != scene_transform;

        if (!needs_regeneration && !cached_instances_.empty()) {
            updateInstanceVisibility(view_position);
            return;
        }

        cached_instances_.clear();
        cached_instances_.reserve(cameras.size());
        camera_ids_.clear();
        camera_ids_.reserve(cameras.size());
        camera_positions_.clear();
        camera_positions_.reserve(cameras.size());

        for (const auto& cam : cameras) {
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

            const glm::mat4 transformed_c2w = scene_transform * glm::inverse(w2c);
            const glm::vec3 cam_pos = glm::vec3(transformed_c2w[3]);
            camera_positions_.push_back(cam_pos);

            const float aspect = static_cast<float>(cam->image_width()) / static_cast<float>(cam->image_height());
            const bool is_equirect = cam->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
            const float fov_y = is_equirect ? EQUIRECTANGULAR_DISPLAY_FOV
                                            : lfs::core::focal2fov(cam->focal_y(), cam->image_height());
            const float half_height = std::tan(fov_y * 0.5f);
            const float half_width = half_height * aspect;

            const glm::mat4 fov_scale = glm::scale(glm::mat4(1.0f), glm::vec3(half_width * 2.0f * scale, half_height * 2.0f * scale, scale));
            const glm::mat4 model = transformed_c2w * GL_TO_COLMAP * fov_scale;

            const bool is_validation = cam->image_name().find("test") != std::string::npos;
            const glm::vec3 color = is_validation ? eval_color : train_color;

            float alpha = 1.0f;
            if (!for_picking) {
                const float distance = glm::length(cam_pos - view_position);
                const float fade_start = FADE_START_MULTIPLIER * scale;
                const float fade_end = FADE_END_MULTIPLIER * scale;
                const float min_visible = MIN_VISIBLE_MULTIPLIER * scale;

                if (distance < min_visible) {
                    alpha = 0.0f;
                } else if (distance < fade_end) {
                    alpha = MIN_VISIBLE_ALPHA;
                } else if (distance < fade_start) {
                    const float t = (distance - fade_end) / (fade_start - fade_end);
                    alpha = MIN_VISIBLE_ALPHA + (1.0f - MIN_VISIBLE_ALPHA) * (t * t * (3.0f - 2.0f * t));
                }
            }

            cached_instances_.push_back({model, color, alpha, 0, is_validation ? 1u : 0u, is_equirect ? 1u : 0u, 0});
            camera_ids_.push_back(cam->uid());
        }

        last_scale_ = scale;
        last_train_color_ = train_color;
        last_eval_color_ = eval_color;
        last_view_position_ = view_position;
        last_scene_transform_ = scene_transform;
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
            cached_instances_[i].alpha = alpha;
        }
        last_view_position_ = view_position;
    }

    Result<void> CameraFrustumRenderer::render(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        const glm::mat4& view,
        const glm::mat4& projection,
        const float scale,
        const glm::vec3& train_color,
        const glm::vec3& eval_color,
        const glm::mat4& scene_transform,
        const bool equirectangular_view) {

        if (!initialized_ || cameras.empty())
            return {};

        uploadReadyThumbnails();

        const glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);
        prepareInstances(cameras, scale, train_color, eval_color, false, view_position, scene_transform);

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
        };

        const auto cleanupInstanceAttributes = []() {
            for (int i = 2; i <= 9; ++i) {
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

        const auto findHighlightIndex = [this](const std::vector<int>& indices) -> int {
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] == highlighted_camera_)
                    return static_cast<int>(i);
            }
            return -1;
        };

        const int frustum_highlight = findHighlightIndex(frustum_indices);
        const int sphere_highlight = findHighlightIndex(sphere_indices);

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
                shader->set("highlightIndex", frustum_highlight);
                VAOBinder vao_bind(vao_);
                setupInstanceAttributes(frustum_instances);
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> face_bind(face_ebo_);
                glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr,
                                        static_cast<GLsizei>(frustum_instances.size()));
                cleanupInstanceAttributes();
            }

            if (!sphere_instances.empty()) {
                shader->set("highlightIndex", sphere_highlight);
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
                shader->set("highlightIndex", frustum_highlight);
                VAOBinder vao_bind(vao_);
                setupInstanceAttributes(frustum_instances);
                BufferBinder<GL_ELEMENT_ARRAY_BUFFER> edge_bind(edge_ebo_);
                glDrawElementsInstanced(GL_LINES, static_cast<GLsizei>(num_edge_indices_),
                                        GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(frustum_instances.size()));
                cleanupInstanceAttributes();
            }

            if (!sphere_instances.empty()) {
                shader->set("highlightIndex", sphere_highlight);
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
        const glm::mat4& scene_transform) {

        if (!initialized_ || cameras.empty())
            return -1;

        if (cached_instances_.empty() || camera_ids_.size() != cameras.size()) {
            const glm::vec3 view_position = glm::vec3(glm::inverse(view)[3]);
            prepareInstances(cameras, scale, last_train_color_, last_eval_color_, false, view_position, scene_transform);
            if (cached_instances_.empty())
                return -1;
        }

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

        GLint current_fbo;
        GLint current_viewport[4];
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
        glGetIntegerv(GL_VIEWPORT, current_viewport);

        glBindFramebuffer(GL_FRAMEBUFFER, picking_fbo_);
        glViewport(0, 0, picking_fbo_width_, picking_fbo_height_);
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
        };

        const auto cleanupPickingAttributes = []() {
            for (int i = 2; i <= 9; ++i) {
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

        glFinish();

        const int pixel_x = std::clamp(static_cast<int>(mouse_pos.x - viewport_pos.x), 0, picking_fbo_width_ - 1);
        const int pixel_y = std::clamp(static_cast<int>(viewport_size.y - (mouse_pos.y - viewport_pos.y)), 0, picking_fbo_height_ - 1);

        const int read_x = std::max(0, pixel_x - 1);
        const int read_y = std::max(0, pixel_y - 1);
        const int read_width = std::min(PICKING_SAMPLE_SIZE, picking_fbo_width_ - read_x);
        const int read_height = std::min(PICKING_SAMPLE_SIZE, picking_fbo_height_ - read_y);

        std::vector<float> pixels(read_width * read_height * 3);
        glReadPixels(read_x, read_y, read_width, read_height, GL_RGB, GL_FLOAT, pixels.data());

        int center_idx = 0;
        if (read_width == 3 && read_height == 3) {
            center_idx = 4 * 3;
        } else if (read_width >= 2 && read_height >= 2) {
            center_idx = ((read_height / 2) * read_width + (read_width / 2)) * 3;
        }

        const int id = (static_cast<int>(pixels[center_idx] * 255.0f + 0.5f) << 16 |
                        static_cast<int>(pixels[center_idx + 1] * 255.0f + 0.5f) << 8 |
                        static_cast<int>(pixels[center_idx + 2] * 255.0f + 0.5f)) -
                       1;

        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
        glViewport(current_viewport[0], current_viewport[1], current_viewport[2], current_viewport[3]);

        if (id >= 0 && id < static_cast<int>(camera_ids_.size())) {
            return camera_ids_[id];
        }
        return -1;
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

    void CameraFrustumRenderer::thumbnailLoaderWorker() {
        constexpr auto IDLE_TIMEOUT = std::chrono::seconds(5);
        constexpr auto POLL_INTERVAL = std::chrono::milliseconds(500);

        std::unique_ptr<lfs::io::NvCodecImageLoader> nvcodec;
        const bool nvcodec_supported = lfs::io::NvCodecImageLoader::is_available();
        auto last_activity = std::chrono::steady_clock::now();

        const auto create_nvcodec = [&]() -> bool {
            if (nvcodec || !nvcodec_supported)
                return nvcodec != nullptr;
            try {
                lfs::io::NvCodecImageLoader::Options opts;
                opts.device_id = 0;
                opts.decoder_pool_size = NVCODEC_DECODER_POOL_SIZE;
                nvcodec = std::make_unique<lfs::io::NvCodecImageLoader>(opts);
                return true;
            } catch (const std::exception& e) {
                LOG_WARN("nvImageCodec init failed: {}", e.what());
                return false;
            }
        };

        while (thumbnail_loader_running_) {
            ThumbnailRequest request;
            bool has_work = false;

            {
                std::unique_lock lock(load_queue_mutex_);
                load_queue_cv_.wait_for(lock, POLL_INTERVAL, [this] {
                    return !thumbnail_load_queue_.empty() || !thumbnail_loader_running_;
                });

                if (!thumbnail_loader_running_)
                    break;

                if (!thumbnail_load_queue_.empty()) {
                    request = std::move(thumbnail_load_queue_.front());
                    thumbnail_load_queue_.pop();
                    has_work = true;
                    last_activity = std::chrono::steady_clock::now();
                }
            }

            // Release nvcodec after idle timeout to free CUDA resources for training
            if (!has_work) {
                if (nvcodec && (std::chrono::steady_clock::now() - last_activity) > IDLE_TIMEOUT) {
                    LOG_DEBUG("Releasing idle thumbnail nvcodec");
                    nvcodec.reset();
                }
                continue;
            }

            LoadedThumbnail loaded;
            loaded.camera_uid = request.camera_uid;

            try {
                std::string ext = request.image_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                bool loaded_with_nvcodec = false;
                const bool is_jpeg = (ext == ".jpg" || ext == ".jpeg");

                if (is_jpeg && create_nvcodec()) {
                    try {
                        const int max_dim = std::max(request.image_width, request.image_height);
                        int pot_resize = 1;
                        while (pot_resize * 2 <= max_dim / THUMBNAIL_SIZE) {
                            pot_resize *= 2;
                        }

                        auto tensor = nvcodec->load_image_gpu(request.image_path, pot_resize, THUMBNAIL_SIZE);
                        auto cpu_tensor = tensor.cpu().contiguous();
                        const auto shape = cpu_tensor.shape();
                        const int c = shape[0], h = shape[1], w = shape[2];

                        if (c != 3)
                            throw std::runtime_error("Expected 3 channels");

                        loaded.width = w;
                        loaded.height = h;
                        loaded.pixel_data.resize(w * h * 3);

                        // CHW (channel-first) to HWC (interleaved) with Y-flip
                        const float* src = cpu_tensor.ptr<float>();
                        const int plane_size = h * w;
                        const float* r_plane = src;
                        const float* g_plane = src + plane_size;
                        const float* b_plane = src + 2 * plane_size;

                        for (int y = 0; y < h; ++y) {
                            const int src_y = h - 1 - y;
                            uint8_t* dst_row = loaded.pixel_data.data() + y * w * 3;
                            const float* r_row = r_plane + src_y * w;
                            const float* g_row = g_plane + src_y * w;
                            const float* b_row = b_plane + src_y * w;

                            for (int x = 0; x < w; ++x) {
                                dst_row[x * 3] = static_cast<uint8_t>(std::clamp(r_row[x] * 255.0f, 0.0f, 255.0f));
                                dst_row[x * 3 + 1] = static_cast<uint8_t>(std::clamp(g_row[x] * 255.0f, 0.0f, 255.0f));
                                dst_row[x * 3 + 2] = static_cast<uint8_t>(std::clamp(b_row[x] * 255.0f, 0.0f, 255.0f));
                            }
                        }
                        loaded_with_nvcodec = true;
                    } catch (...) {}
                }

                if (!loaded_with_nvcodec) {
                    auto [data, width, height, channels] = lfs::core::load_image(request.image_path, -1, THUMBNAIL_SIZE);
                    if (!data)
                        continue;

                    loaded.width = width;
                    loaded.height = height;
                    loaded.pixel_data.resize(width * height * 3);

                    for (int y = 0; y < height; ++y) {
                        std::memcpy(loaded.pixel_data.data() + y * width * 3,
                                    data + (height - 1 - y) * width * 3,
                                    width * 3);
                    }
                    lfs::core::free_image(data);
                }

                {
                    std::lock_guard lock(ready_queue_mutex_);
                    thumbnail_ready_queue_.push(std::move(loaded));
                }

            } catch (const std::exception& e) {
                LOG_WARN("Thumbnail load failed for camera {}: {}", request.camera_uid, e.what());
                std::lock_guard lock(pending_mutex_);
                thumbnail_pending_.erase(request.camera_uid);
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
            .image_height = camera.image_height()};

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