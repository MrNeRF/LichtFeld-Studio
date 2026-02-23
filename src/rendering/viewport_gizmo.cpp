/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_gizmo.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "rendering/render_constants.hpp"
#include "shader_paths.hpp"
#include "text_renderer.hpp"
#include <SDL3/SDL.h>
#include <algorithm>
#include <format>
#include <numbers>
#include <ranges>
#include <vector>

namespace lfs::rendering {

    namespace {
        constexpr float HOVER_SCALE = 1.2f;
        constexpr float HOVER_BRIGHTNESS = 1.3f;
        constexpr float HIT_RADIUS_SCALE = 2.5f;
    } // namespace

    constexpr glm::vec3 ViewportGizmo::AXIS_COLORS[];

    ViewportGizmo::ViewportGizmo() = default;
    ViewportGizmo::~ViewportGizmo() = default;

    Result<void> ViewportGizmo::initialize() {
        if (initialized_)
            return {};

        LOG_TIMER("ViewportGizmo::initialize");

        if (auto result = createShaders(); !result) {
            return result;
        }

        if (auto result = generateGeometry(); !result) {
            return result;
        }

        // Initialize text renderer using actual window size (will be updated in render)
        int width = 1280, height = 720;
        if (SDL_Window* window = SDL_GL_GetCurrentWindow()) {
            SDL_GetWindowSizeInPixels(window, &width, &height);
        }
        text_renderer_ = std::make_unique<TextRenderer>(width, height);

        // Load font from our assets
        auto font_path = lfs::core::getFontsDir() / "JetBrainsMono-Regular.ttf";
        if (auto result = text_renderer_->LoadFont(font_path, 48); !result) {
            LOG_WARN("ViewportGizmo: Failed to load font: {}", result.error());
            text_renderer_.reset();
        }

        initialized_ = true;
        LOG_INFO("ViewportGizmo initialized successfully");
        return {};
    }

    void ViewportGizmo::shutdown() {
        LOG_DEBUG("Shutting down ViewportGizmo");
        vao_ = VAO();
        vbo_ = VBO();
        shader_ = ManagedShader();
        text_renderer_.reset();
        initialized_ = false;
    }

    Result<void> ViewportGizmo::createShaders() {
        LOG_TIMER_TRACE("ViewportGizmo::createShaders");

        auto result = load_shader("viewport_gizmo", "viewport_gizmo.vert", "viewport_gizmo.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load viewport gizmo shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);
        LOG_DEBUG("ViewportGizmo shaders created successfully");
        return {};
    }

    Result<void> ViewportGizmo::generateGeometry() {
        LOG_TIMER_TRACE("ViewportGizmo::generateGeometry");

        auto vao_result = create_vao();
        if (!vao_result) {
            LOG_ERROR("Failed to create VAO: {}", vao_result.error());
            return std::unexpected(vao_result.error());
        }
        vao_ = std::move(*vao_result);

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            LOG_ERROR("Failed to create VBO: {}", vbo_result.error());
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        VAOBinder vao_bind(vao_);
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);

        std::vector<float> vertices;
        vertices.reserve(10000);

        auto addVertex = [&](float x, float y, float z, float nx, float ny, float nz) {
            vertices.insert(vertices.end(), {x, y, z, nx, ny, nz});
        };

        // Generate cylinder (for axes)
        constexpr int segments = 16;
        constexpr size_t kVertexBufferReserve = segments * 6; // Number of vertices for cylinder
        vertices.reserve(kVertexBufferReserve);

        // Generate cylinder (for axes)
        constexpr float two_pi = 2 * std::numbers::pi_v<float>;

        for (const auto i : std::views::iota(0, segments)) {
            float a1 = static_cast<float>(i) / segments * two_pi;
            float a2 = static_cast<float>(i + 1) / segments * two_pi;
            float c1 = cos(a1), s1 = sin(a1);
            float c2 = cos(a2), s2 = sin(a2);

            addVertex(c1, s1, 0, c1, s1, 0);
            addVertex(c2, s2, 0, c2, s2, 0);
            addVertex(c1, s1, 1, c1, s1, 0);

            addVertex(c2, s2, 0, c2, s2, 0);
            addVertex(c2, s2, 1, c2, s2, 0);
            addVertex(c1, s1, 1, c1, s1, 0);
        }
        cylinder_vertex_count_ = segments * 6;

        // Generate sphere
        sphere_vertex_start_ = vertices.size() / 6;
        constexpr int latBands = 16, longBands = 16;

        for (const auto lat : std::views::iota(0, latBands)) {
            float theta1 = static_cast<float>(lat) * std::numbers::pi_v<float> / latBands;
            float theta2 = static_cast<float>(lat + 1) * std::numbers::pi_v<float> / latBands;

            float sinTheta1 = sin(theta1), cosTheta1 = cos(theta1);
            float sinTheta2 = sin(theta2), cosTheta2 = cos(theta2);

            for (const auto lon : std::views::iota(0, longBands)) {
                float phi1 = static_cast<float>(lon) * two_pi / longBands;
                float phi2 = static_cast<float>(lon + 1) * two_pi / longBands;

                float sinPhi1 = sin(phi1), cosPhi1 = cos(phi1);
                float sinPhi2 = sin(phi2), cosPhi2 = cos(phi2);

                float x1 = sinTheta1 * cosPhi1, y1 = cosTheta1, z1 = sinTheta1 * sinPhi1;
                float x2 = sinTheta1 * cosPhi2, y2 = cosTheta1, z2 = sinTheta1 * sinPhi2;
                float x3 = sinTheta2 * cosPhi2, y3 = cosTheta2, z3 = sinTheta2 * sinPhi2;
                float x4 = sinTheta2 * cosPhi1, y4 = cosTheta2, z4 = sinTheta2 * sinPhi1;

                addVertex(x1, y1, z1, x1, y1, z1);
                addVertex(x2, y2, z2, x2, y2, z2);
                addVertex(x3, y3, z3, x3, y3, z3);

                addVertex(x1, y1, z1, x1, y1, z1);
                addVertex(x3, y3, z3, x3, y3, z3);
                addVertex(x4, y4, z4, x4, y4, z4);
            }
        }
        sphere_vertex_count_ = (vertices.size() / 6) - sphere_vertex_start_;

        // Generate ring (flat washer for negative axis indicators)
        ring_vertex_start_ = vertices.size() / 6;
        constexpr int RING_SEGMENTS = 32;
        constexpr float OUTER_RADIUS = 1.0f;
        constexpr float INNER_RADIUS = 0.55f;

        for (const auto i : std::views::iota(0, RING_SEGMENTS)) {
            const float theta1 = static_cast<float>(i) * two_pi / RING_SEGMENTS;
            const float theta2 = static_cast<float>(i + 1) * two_pi / RING_SEGMENTS;
            const float cos_t1 = cos(theta1), sin_t1 = sin(theta1);
            const float cos_t2 = cos(theta2), sin_t2 = sin(theta2);

            const glm::vec3 outer1(OUTER_RADIUS * cos_t1, OUTER_RADIUS * sin_t1, 0.0f);
            const glm::vec3 outer2(OUTER_RADIUS * cos_t2, OUTER_RADIUS * sin_t2, 0.0f);
            const glm::vec3 inner1(INNER_RADIUS * cos_t1, INNER_RADIUS * sin_t1, 0.0f);
            const glm::vec3 inner2(INNER_RADIUS * cos_t2, INNER_RADIUS * sin_t2, 0.0f);

            // Front face
            addVertex(outer1.x, outer1.y, outer1.z, 0, 0, 1);
            addVertex(outer2.x, outer2.y, outer2.z, 0, 0, 1);
            addVertex(inner1.x, inner1.y, inner1.z, 0, 0, 1);
            addVertex(inner1.x, inner1.y, inner1.z, 0, 0, 1);
            addVertex(outer2.x, outer2.y, outer2.z, 0, 0, 1);
            addVertex(inner2.x, inner2.y, inner2.z, 0, 0, 1);

            // Back face
            addVertex(outer1.x, outer1.y, outer1.z, 0, 0, -1);
            addVertex(inner1.x, inner1.y, inner1.z, 0, 0, -1);
            addVertex(outer2.x, outer2.y, outer2.z, 0, 0, -1);
            addVertex(inner1.x, inner1.y, inner1.z, 0, 0, -1);
            addVertex(inner2.x, inner2.y, inner2.z, 0, 0, -1);
            addVertex(outer2.x, outer2.y, outer2.z, 0, 0, -1);
        }
        ring_vertex_count_ = (vertices.size() / 6) - ring_vertex_start_;

        upload_buffer(GL_ARRAY_BUFFER, std::span(vertices), GL_STATIC_DRAW);

        // Position attribute
        VertexAttribute position_attr{
            .index = 0,
            .size = 3,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = 6 * sizeof(float),
            .offset = nullptr};
        position_attr.apply();

        // Normal attribute
        VertexAttribute normal_attr{
            .index = 1,
            .size = 3,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = 6 * sizeof(float),
            .offset = (void*)(3 * sizeof(float))};
        normal_attr.apply();

        LOG_DEBUG("Generated {} cylinder, {} sphere, {} ring vertices",
                  cylinder_vertex_count_, sphere_vertex_count_, ring_vertex_count_);
        return {};
    }

    Result<void> ViewportGizmo::render(const glm::mat3& camera_rotation,
                                       const glm::vec2& viewport_pos,
                                       const glm::vec2& viewport_size) {
        if (!initialized_) {
            LOG_ERROR("Viewport gizmo not initialized");
            return std::unexpected("Viewport gizmo not initialized");
        }

        LOG_TIMER_TRACE("ViewportGizmo::render");

        GLStateGuard state_guard;

        // Get framebuffer size for ImGui-to-GL coordinate conversion
        int fb_height = 0;
        if (SDL_Window* const window = SDL_GL_GetCurrentWindow()) {
            int fb_width = 0;
            SDL_GetWindowSizeInPixels(window, &fb_width, &fb_height);
        }

        // Position gizmo at upper-right of viewport (convert ImGui Y to GL Y)
        const int gizmo_x = static_cast<int>(viewport_pos.x + viewport_size.x - size_ - margin_x_);
        const int gizmo_y = fb_height - static_cast<int>(viewport_pos.y) - margin_y_ - size_;

        // Set gizmo viewport
        glViewport(gizmo_x, gizmo_y, size_, size_);

        // Clear depth for gizmo
        glEnable(GL_SCISSOR_TEST);
        glScissor(gizmo_x, gizmo_y, size_, size_);
        glClear(GL_DEPTH_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);

        // Enable depth testing for gizmo
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);

        // Disable face culling for gizmo
        glDisable(GL_CULL_FACE);

        // Build view matrix matching main renderer's coordinate system
        constexpr float GIZMO_DISTANCE = 2.8f;
        constexpr float GIZMO_FOV = 38.0f;
        const glm::mat3 view_rotation = computeViewRotation(camera_rotation);
        glm::mat4 view(1.0f);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                view[i][j] = view_rotation[i][j];
        view[3][2] = -GIZMO_DISTANCE;
        const glm::mat4 proj = glm::perspective(glm::radians(GIZMO_FOV), 1.0f, 0.1f, 10.0f);

        const glm::vec3 originCamSpace = glm::vec3(view * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        const float refDist = glm::length(originCamSpace);

        // Use shader
        ShaderScope s(shader_);

        // Draw axes
        const float sphereRadius = 0.198f;
        const float axisLen = 0.63f - sphereRadius;
        const float axisRad = 0.0225f;
        const float labelDistance = 0.63f;

        const glm::mat4 rotations[3] = {
            glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 1, 0)),  // X
            glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(1, 0, 0)), // Y
            glm::mat4(1)                                                         // Z
        };

        VAOBinder vao_bind(vao_);

        // Draw axis cylinders
        for (int i = 0; i < 3; i++) {
            glm::mat4 model = rotations[i] * glm::scale(glm::mat4(1), glm::vec3(axisRad, axisRad, axisLen));
            glm::mat4 mvp = proj * view * model;
            if (auto result = s->set("uMVP", mvp); !result)
                return result;
            if (auto result = s->set("uModel", model); !result)
                return result;
            if (auto result = s->set("uColor", AXIS_COLORS[i]); !result)
                return result;
            if (auto result = s->set("uAlpha", 1.0f); !result)
                return result;
            if (auto result = s->set("uUseLighting", 1); !result)
                return result;
            glDrawArrays(GL_TRIANGLES, 0, cylinder_vertex_count_);
        }

        // Enable blending for spheres
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);

        // Sphere info for text rendering
        struct SphereInfo {
            glm::vec3 screenPos;
            float depth;
            int index;
            bool visible;
        };
        SphereInfo sphereInfo[3];

        // Draw spheres and calculate positions
        for (int i = 0; i < 3; i++) {
            glm::vec3 labelPos = glm::vec3(0);
            labelPos[i] = labelDistance;

            glm::vec3 camSpacePos = glm::vec3(view * glm::vec4(labelPos, 1.0f));
            float dist = glm::length(camSpacePos);
            float scaleFactor = dist / refDist;

            const bool is_hovered = hovered_axis_.has_value() &&
                                    static_cast<int>(*hovered_axis_) == i && !hovered_negative_;
            const float scale = is_hovered ? HOVER_SCALE : 1.0f;
            const float brightness = is_hovered ? HOVER_BRIGHTNESS : 1.0f;

            const glm::mat4 model = glm::translate(glm::mat4(1), labelPos) *
                                    glm::scale(glm::mat4(1), glm::vec3(sphereRadius * scaleFactor * scale));
            const glm::mat4 mvp = proj * view * model;
            if (auto r = s->set("uMVP", mvp); !r)
                return r;
            if (auto r = s->set("uModel", glm::mat4(1.0f)); !r)
                return r;
            const glm::vec3 color = glm::min(AXIS_COLORS[i] * brightness, glm::vec3(1.0f));
            if (auto r = s->set("uColor", color); !r)
                return r;
            if (auto r = s->set("uAlpha", 1.0f); !r)
                return r;
            if (auto r = s->set("uUseLighting", 0); !r)
                return r;
            glDrawArrays(GL_TRIANGLES, sphere_vertex_start_, sphere_vertex_count_);

            const glm::vec4 clipPos = proj * view * glm::vec4(labelPos, 1.0f);
            if (clipPos.w > 0) {
                const glm::vec3 ndcPos = glm::vec3(clipPos) / clipPos.w;
                const float gizmoX = (ndcPos.x * 0.5f + 0.5f) * size_;
                const float gizmoY = (ndcPos.y * 0.5f + 0.5f) * size_;

                sphereInfo[i].screenPos.x = gizmoX + gizmo_x;
                sphereInfo[i].screenPos.y = gizmo_y + gizmoY;
                sphereInfo[i].depth = clipPos.z / clipPos.w;
                sphereInfo[i].index = i;
                sphereInfo[i].visible = true;

                sphere_hits_[i].screen_pos = glm::vec2(
                    sphereInfo[i].screenPos.x, fb_height - sphereInfo[i].screenPos.y);
                sphere_hits_[i].radius = sphereRadius * scaleFactor * size_ * 0.5f;
                sphere_hits_[i].visible = true;
            } else {
                sphereInfo[i].visible = false;
                sphere_hits_[i].visible = false;
            }
        }

        // Draw rings (negative axis indicators)
        constexpr glm::vec3 RING_NORMAL(0.0f, 0.0f, 1.0f);
        const glm::vec3 camWorldPos = -glm::transpose(view_rotation) * glm::vec3(view[3]);

        for (int i = 0; i < 3; i++) {
            glm::vec3 pos(0.0f);
            pos[i] = -labelDistance;

            const glm::vec3 camSpacePos = glm::vec3(view * glm::vec4(pos, 1.0f));
            const float scaleFactor = glm::length(camSpacePos) / refDist;
            const glm::vec3 toCam = glm::normalize(camWorldPos - pos);

            const bool is_hovered = hovered_axis_.has_value() &&
                                    static_cast<int>(*hovered_axis_) == i && hovered_negative_;
            const float scale = is_hovered ? HOVER_SCALE : 1.0f;
            const float brightness = is_hovered ? HOVER_BRIGHTNESS : 1.0f;

            glm::mat4 faceRotation(1.0f);
            const float dot = glm::dot(RING_NORMAL, toCam);
            if (dot < 0.999f && dot > -0.999f) {
                const glm::vec3 axis = glm::normalize(glm::cross(RING_NORMAL, toCam));
                faceRotation = glm::rotate(glm::mat4(1.0f), std::acosf(dot), axis);
            } else if (dot <= -0.999f) {
                faceRotation = glm::rotate(glm::mat4(1.0f), std::numbers::pi_v<float>, glm::vec3(0, 1, 0));
            }

            const glm::mat4 model = glm::translate(glm::mat4(1), pos) *
                                    faceRotation *
                                    glm::scale(glm::mat4(1), glm::vec3(sphereRadius * scaleFactor * scale));
            const glm::mat4 mvp = proj * view * model;

            if (auto r = s->set("uMVP", mvp); !r)
                return r;
            if (auto r = s->set("uModel", model); !r)
                return r;
            const glm::vec3 color = glm::min(AXIS_COLORS[i] * brightness, glm::vec3(1.0f));
            if (auto r = s->set("uColor", color); !r)
                return r;
            if (auto r = s->set("uAlpha", 1.0f); !r)
                return r;
            if (auto r = s->set("uUseLighting", 0); !r)
                return r;

            glDrawArrays(GL_TRIANGLES, ring_vertex_start_, ring_vertex_count_);

            const glm::vec4 clipPos = proj * view * glm::vec4(pos, 1.0f);
            if (clipPos.w > 0) {
                const glm::vec3 ndcPos = glm::vec3(clipPos) / clipPos.w;
                const float ringX = (ndcPos.x * 0.5f + 0.5f) * size_;
                const float ringY = (ndcPos.y * 0.5f + 0.5f) * size_;

                ring_hits_[i].screen_pos = glm::vec2(ringX + gizmo_x, fb_height - ringY - gizmo_y);
                ring_hits_[i].radius = sphereRadius * scaleFactor * size_ * 0.5f;
                ring_hits_[i].visible = true;
            } else {
                ring_hits_[i].visible = false;
            }
        }

        // Sort spheres by depth
        std::sort(sphereInfo, sphereInfo + 3, [](const SphereInfo& a, const SphereInfo& b) {
            return a.depth > b.depth;
        });

        // Get current viewport for relative positioning
        auto vp = state_guard.savedState().viewport;

        // Make screen positions relative to the current viewport origin
        for (int i = 0; i < 3; ++i) {
            sphereInfo[i].screenPos.x -= vp[0];
            sphereInfo[i].screenPos.y -= vp[1];
        }

        // Draw text labels with occlusion
        if (text_renderer_) {
            const char* axisLabels[3] = {"X", "Y", "Z"};

            // Update text renderer size if needed
            int window_width = vp[2];
            int window_height = vp[3];
            text_renderer_->updateScreenSize(window_width, window_height);

            // Ensure proper color mask for text rendering
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

            glEnable(GL_STENCIL_TEST);

            for (int i = 0; i < 3; i++) {
                if (sphereInfo[i].visible) {
                    int idx = sphereInfo[i].index;

                    glClearStencil(0);
                    glClear(GL_STENCIL_BUFFER_BIT);

                    // Mark occluding spheres in stencil
                    glStencilFunc(GL_ALWAYS, 1, 0xFF);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
                    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                    glDepthMask(GL_FALSE);
                    glDepthFunc(GL_LEQUAL);

                    // Redraw closer spheres
                    for (int j = 0; j < 3; j++) {
                        if (sphereInfo[j].visible && sphereInfo[j].depth < sphereInfo[i].depth) {
                            int jdx = sphereInfo[j].index;
                            glm::vec3 jLabelPos = glm::vec3(0);
                            jLabelPos[jdx] = labelDistance;

                            glm::vec3 jCamSpacePos = glm::vec3(view * glm::vec4(jLabelPos, 1.0f));
                            float jDist = glm::length(jCamSpacePos);
                            float jScaleFactor = jDist / refDist;

                            glm::mat4 jModel = glm::translate(glm::mat4(1), jLabelPos) *
                                               glm::scale(glm::mat4(1), glm::vec3(sphereRadius * jScaleFactor));
                            glm::mat4 jMvp = proj * view * jModel;
                            if (auto result = s->set("uMVP", jMvp); !result) {
                                LOG_WARN("Failed to set uMVP: {}", result.error());
                            }

                            if (auto result = s->set("uModel", glm::mat4(1.0f)); !result) {
                                LOG_WARN("Failed to set uModel: {}", result.error());
                            }

                            if (auto result = s->set("uColor", AXIS_COLORS[jdx]); !result) {
                                LOG_WARN("Failed to set uColor: {}", result.error());
                            }

                            if (auto result = s->set("uAlpha", 1.0f); !result) {
                                LOG_WARN("Failed to set uAlpha: {}", result.error());
                            }

                            if (auto result = s->set("uUseLighting", 0); !result) {
                                LOG_WARN("Failed to set uUseLighting: {}", result.error());
                            }
                            glDrawArrays(GL_TRIANGLES, sphere_vertex_start_, sphere_vertex_count_);
                        }
                    }

                    glDepthFunc(GL_LESS);
                    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                    glDepthMask(GL_TRUE);
                    glStencilFunc(GL_EQUAL, 0, 0xFF);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

                    // Render axis label centered in sphere
                    glViewport(vp[0], vp[1], vp[2], vp[3]);

                    constexpr float TEXT_SCALE = 0.28f;
                    const char label = axisLabels[idx][0];
                    const glm::vec2 size = text_renderer_->getCharacterSize(label, TEXT_SCALE);
                    const glm::vec2 bearing = text_renderer_->getCharacterBearing(label, TEXT_SCALE);
                    const float x = sphereInfo[i].screenPos.x - bearing.x - size.x * 0.5f;
                    const float y = sphereInfo[i].screenPos.y - bearing.y + size.y * 0.5f;

                    glDisable(GL_DEPTH_TEST);
                    glDepthMask(GL_FALSE);
                    glActiveTexture(GL_TEXTURE0);

                    if (auto result = text_renderer_->RenderText(
                            axisLabels[idx], x, y, TEXT_SCALE, glm::vec3(1.0f));
                        !result) {
                        LOG_WARN("Failed to render text: {}", result.error());
                    }

                    glEnable(GL_DEPTH_TEST);
                    glDepthMask(GL_TRUE);
                    glViewport(gizmo_x, gizmo_y, size_, size_);
                }
            }

            glDisable(GL_STENCIL_TEST);
        }

        // State automatically restored by GLStateGuard destructor
        return {};
    }

    std::optional<GizmoHitResult> ViewportGizmo::hitTest(const glm::vec2& click_pos,
                                                         const glm::vec2& viewport_pos,
                                                         const glm::vec2& viewport_size) const {
        if (!initialized_)
            return std::nullopt;

        // Gizmo bounds (upper-right corner in ImGui coords)
        const float gizmo_x = viewport_pos.x + viewport_size.x - size_ - margin_x_;
        const float gizmo_y = viewport_pos.y + margin_y_;

        // Early out if outside gizmo bounds
        if (click_pos.x < gizmo_x || click_pos.x > gizmo_x + size_ ||
            click_pos.y < gizmo_y || click_pos.y > gizmo_y + size_) {
            return std::nullopt;
        }

        // Check spheres (positive axes)
        for (int i = 0; i < 3; ++i) {
            if (!sphere_hits_[i].visible)
                continue;
            const float dx = click_pos.x - sphere_hits_[i].screen_pos.x;
            const float dy = click_pos.y - sphere_hits_[i].screen_pos.y;
            const float r = sphere_hits_[i].radius * HIT_RADIUS_SCALE;
            if (dx * dx + dy * dy <= r * r) {
                return GizmoHitResult{static_cast<GizmoAxis>(i), false};
            }
        }

        // Check rings (negative axes)
        for (int i = 0; i < 3; ++i) {
            if (!ring_hits_[i].visible)
                continue;
            const float dx = click_pos.x - ring_hits_[i].screen_pos.x;
            const float dy = click_pos.y - ring_hits_[i].screen_pos.y;
            const float r = ring_hits_[i].radius * HIT_RADIUS_SCALE;
            if (dx * dx + dy * dy <= r * r) {
                return GizmoHitResult{static_cast<GizmoAxis>(i), true};
            }
        }

        return std::nullopt;
    }

    glm::mat3 ViewportGizmo::getAxisViewRotation(const GizmoAxis axis, const bool negative) {
        // Returns camera-to-world rotation for looking along an axis
        const float sign = negative ? -1.0f : 1.0f;

        glm::vec3 forward, up, right;
        switch (axis) {
        case GizmoAxis::X:
            forward = glm::vec3(sign, 0.0f, 0.0f);
            up = glm::vec3(0.0f, 1.0f, 0.0f);
            break;
        case GizmoAxis::Y:
            forward = glm::vec3(0.0f, -sign, 0.0f);
            up = glm::vec3(0.0f, 0.0f, sign);
            break;
        case GizmoAxis::Z:
            forward = glm::vec3(0.0f, 0.0f, sign);
            up = glm::vec3(0.0f, 1.0f, 0.0f);
            break;
        default:
            return glm::mat3(1.0f);
        }

        right = glm::normalize(glm::cross(up, forward));
        up = glm::normalize(glm::cross(forward, right));
        return glm::mat3(right, up, forward); // Columns: [right, up, forward]
    }

} // namespace lfs::rendering