/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen_overlay_renderer.hpp"

#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "text_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace lfs::rendering {

    namespace {
        constexpr unsigned int kTextAtlasSize = 32;
        constexpr int kMinCircleSegments = 12;
        constexpr int kMaxCircleSegments = 64;
    } // namespace

    ScreenOverlayRenderer::ScreenOverlayRenderer() = default;
    ScreenOverlayRenderer::~ScreenOverlayRenderer() = default;

    Result<void> ScreenOverlayRenderer::initialize() {
        LOG_TIMER("ScreenOverlayRenderer::initialize");
        if (initialized_) {
            return {};
        }

        auto shader_result = load_shader("screen_overlay", "screen_overlay.vert", "screen_overlay.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load screen_overlay shader: {}", shader_result.error().what());
            return std::unexpected(shader_result.error().what());
        }
        shader_ = std::move(*shader_result);

        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }
        vao_ = std::move(*vao_result);

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        VAOBinder vao_bind(vao_);
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);

        constexpr std::size_t kInitialBytes = sizeof(Vertex) * 4096;
        glBufferData(GL_ARRAY_BUFFER, kInitialBytes, nullptr, GL_DYNAMIC_DRAW);
        vbo_capacity_bytes_ = kInitialBytes;

        VertexAttribute pos_attr{
            .index = 0,
            .size = 2,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = sizeof(Vertex),
            .offset = reinterpret_cast<const void*>(offsetof(Vertex, pos_px))};
        pos_attr.apply();

        VertexAttribute color_attr{
            .index = 1,
            .size = 4,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = sizeof(Vertex),
            .offset = reinterpret_cast<const void*>(offsetof(Vertex, color_premul))};
        color_attr.apply();

        text_atlas_size_ = kTextAtlasSize;
        initialized_ = true;
        LOG_INFO("ScreenOverlayRenderer initialized");
        return {};
    }

    void ScreenOverlayRenderer::shutdown() {
        text_renderer_.reset();
        text_renderer_failed_ = false;
        shader_ = ManagedShader();
        vbo_ = VBO();
        vao_ = VAO();
        vbo_capacity_bytes_ = 0;
        verts_.clear();
        clip_stack_.clear();
        initialized_ = false;
    }

    Result<void> ScreenOverlayRenderer::ensureFontLoaded() {
        if (text_renderer_) {
            return {};
        }
        if (text_renderer_failed_) {
            return std::unexpected("text renderer previously failed to init");
        }

        assert(window_w_ > 0 && window_h_ > 0);
        text_renderer_ = std::make_unique<TextRenderer>(static_cast<unsigned>(window_w_),
                                                        static_cast<unsigned>(window_h_));

        const auto font_path = lfs::core::getFontsDir() / "Inter-Regular.ttf";
        if (auto r = text_renderer_->LoadAsciiFont(font_path, text_atlas_size_); !r) {
            LOG_WARN("ScreenOverlayRenderer: failed to load font: {}", r.error());
            text_renderer_.reset();
            text_renderer_failed_ = true;
            return std::unexpected(r.error());
        }
        return {};
    }

    void ScreenOverlayRenderer::beginFrame(int window_pixel_w, int window_pixel_h) {
        assert(initialized_);
        assert(!frame_active_);
        window_w_ = window_pixel_w;
        window_h_ = window_pixel_h;
        if (text_renderer_) {
            text_renderer_->updateScreenSize(static_cast<unsigned>(std::max(1, window_w_)),
                                             static_cast<unsigned>(std::max(1, window_h_)));
        }
        verts_.clear();
        clip_stack_.clear();
        frame_active_ = true;
    }

    void ScreenOverlayRenderer::endFrame() {
        if (!frame_active_) {
            return;
        }
        flush();
        assert(clip_stack_.empty() && "ScopedClipRect leaked across endFrame");
        clip_stack_.clear();
        frame_active_ = false;
    }

    void ScreenOverlayRenderer::pushClipRect(glm::vec2 min, glm::vec2 max, bool intersect_with_current) {
        flush();
        ClipRect r{min, max};
        if (intersect_with_current && !clip_stack_.empty()) {
            const auto& cur = clip_stack_.back();
            r.min.x = std::max(r.min.x, cur.min.x);
            r.min.y = std::max(r.min.y, cur.min.y);
            r.max.x = std::min(r.max.x, cur.max.x);
            r.max.y = std::min(r.max.y, cur.max.y);
        }
        if (r.max.x < r.min.x)
            r.max.x = r.min.x;
        if (r.max.y < r.min.y)
            r.max.y = r.min.y;
        clip_stack_.push_back(r);
    }

    void ScreenOverlayRenderer::popClipRect() {
        flush();
        if (!clip_stack_.empty()) {
            clip_stack_.pop_back();
        }
    }

    glm::vec4 ScreenOverlayRenderer::toPremul(OverlayColor c) const {
        return {c.r * c.a, c.g * c.a, c.b * c.a, c.a};
    }

    int ScreenOverlayRenderer::adaptiveSegments(float radius) const {
        const int n = static_cast<int>(std::ceil(radius * 0.6f));
        return std::clamp(n, kMinCircleSegments, kMaxCircleSegments);
    }

    void ScreenOverlayRenderer::applyClipScissor() const {
        if (clip_stack_.empty()) {
            glDisable(GL_SCISSOR_TEST);
            return;
        }
        const auto& r = clip_stack_.back();
        const float fbh = static_cast<float>(window_h_);
        const int sx = static_cast<int>(std::floor(r.min.x));
        const int sy = static_cast<int>(std::floor(fbh - r.max.y));
        const int sw = std::max(0, static_cast<int>(std::ceil(r.max.x - r.min.x)));
        const int sh = std::max(0, static_cast<int>(std::ceil(r.max.y - r.min.y)));
        glEnable(GL_SCISSOR_TEST);
        glScissor(sx, sy, sw, sh);
    }

    std::vector<glm::vec2> ScreenOverlayRenderer::circlePerimeter(glm::vec2 center, float radius,
                                                                  int segments) const {
        constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;
        std::vector<glm::vec2> pts;
        pts.reserve(segments);
        for (int i = 0; i < segments; ++i) {
            const float a = (static_cast<float>(i) / static_cast<float>(segments)) * two_pi;
            pts.emplace_back(center.x + std::cos(a) * radius, center.y + std::sin(a) * radius);
        }
        return pts;
    }

    void ScreenOverlayRenderer::emitTriangle(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec4 c) {
        verts_.push_back({p0, c});
        verts_.push_back({p1, c});
        verts_.push_back({p2, c});
    }

    void ScreenOverlayRenderer::emitLineSegment(glm::vec2 a, glm::vec2 b, glm::vec4 c, float thickness) {
        glm::vec2 d = b - a;
        const float len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len <= 1e-6f) {
            return;
        }
        const float half = thickness * 0.5f;
        const glm::vec2 n = glm::vec2(-d.y, d.x) / len * half;
        const glm::vec2 v0 = a + n;
        const glm::vec2 v1 = a - n;
        const glm::vec2 v2 = b + n;
        const glm::vec2 v3 = b - n;
        emitTriangle(v0, v1, v3, c);
        emitTriangle(v0, v3, v2, c);
    }

    void ScreenOverlayRenderer::addLine(glm::vec2 a, glm::vec2 b, OverlayColor c, float thickness) {
        assert(frame_active_);
        emitLineSegment(a, b, toPremul(c), thickness);
    }

    void ScreenOverlayRenderer::addPolyline(std::span<const glm::vec2> pts, OverlayColor c,
                                            bool closed, float thickness) {
        assert(frame_active_);
        if (pts.size() < 2)
            return;
        const glm::vec4 col = toPremul(c);
        for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
            emitLineSegment(pts[i], pts[i + 1], col, thickness);
        }
        if (closed && pts.size() >= 3) {
            emitLineSegment(pts.back(), pts.front(), col, thickness);
        }
    }

    void ScreenOverlayRenderer::addConvexPolyFilled(std::span<const glm::vec2> pts, OverlayColor c) {
        assert(frame_active_);
        if (pts.size() < 3)
            return;
        const glm::vec4 col = toPremul(c);
        for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
            emitTriangle(pts[0], pts[i], pts[i + 1], col);
        }
    }

    void ScreenOverlayRenderer::addCircle(glm::vec2 center, float radius, OverlayColor c,
                                          int segments, float thickness) {
        assert(frame_active_);
        if (radius <= 0.0f)
            return;
        const int n = (segments > 0) ? std::clamp(segments, 3, 256) : adaptiveSegments(radius);
        const auto pts = circlePerimeter(center, radius, n);
        addPolyline(pts, c, true, thickness);
    }

    void ScreenOverlayRenderer::addCircleFilled(glm::vec2 center, float radius, OverlayColor c,
                                                int segments) {
        assert(frame_active_);
        if (radius <= 0.0f)
            return;
        const int n = (segments > 0) ? std::clamp(segments, 3, 256) : adaptiveSegments(radius);
        const auto pts = circlePerimeter(center, radius, n);
        const glm::vec4 col = toPremul(c);
        for (int i = 0; i < n; ++i) {
            emitTriangle(center, pts[i], pts[(i + 1) % n], col);
        }
    }

    void ScreenOverlayRenderer::addRect(glm::vec2 a, glm::vec2 b, OverlayColor c, float thickness) {
        assert(frame_active_);
        const float x0 = std::min(a.x, b.x);
        const float y0 = std::min(a.y, b.y);
        const float x1 = std::max(a.x, b.x);
        const float y1 = std::max(a.y, b.y);
        const glm::vec4 col = toPremul(c);
        emitLineSegment({x0, y0}, {x1, y0}, col, thickness);
        emitLineSegment({x1, y0}, {x1, y1}, col, thickness);
        emitLineSegment({x1, y1}, {x0, y1}, col, thickness);
        emitLineSegment({x0, y1}, {x0, y0}, col, thickness);
    }

    void ScreenOverlayRenderer::addRectFilled(glm::vec2 a, glm::vec2 b, OverlayColor c) {
        assert(frame_active_);
        const float x0 = std::min(a.x, b.x);
        const float y0 = std::min(a.y, b.y);
        const float x1 = std::max(a.x, b.x);
        const float y1 = std::max(a.y, b.y);
        const glm::vec4 col = toPremul(c);
        emitTriangle({x0, y0}, {x1, y0}, {x1, y1}, col);
        emitTriangle({x0, y0}, {x1, y1}, {x0, y1}, col);
    }

    void ScreenOverlayRenderer::addTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2,
                                                  OverlayColor c) {
        assert(frame_active_);
        emitTriangle(p0, p1, p2, toPremul(c));
    }

    glm::vec2 ScreenOverlayRenderer::measureText(std::string_view text, float size_px) const {
        if (!text_renderer_) {
            return {0.0f, size_px};
        }
        const float scale = size_px / static_cast<float>(text_atlas_size_);
        return text_renderer_->measureString(text, scale);
    }

    void ScreenOverlayRenderer::addText(glm::vec2 top_left_px, std::string_view text,
                                        OverlayColor c, float size_px) {
        assert(frame_active_);
        if (text.empty())
            return;
        if (auto r = ensureFontLoaded(); !r) {
            return;
        }
        flush();

        const float scale = size_px / static_cast<float>(text_atlas_size_);
        const float ascent = text_renderer_->getAscender(scale);
        const float gl_y = static_cast<float>(window_h_) - (top_left_px.y + ascent);

        GLStateGuard text_state_guard;
        glViewport(0, 0, window_w_, window_h_);
        applyClipScissor();

        std::string s(text);
        if (auto r = text_renderer_->RenderText(s, top_left_px.x, gl_y, scale,
                                                glm::vec3(c.r, c.g, c.b), c.a); !r)
            LOG_WARN("RenderText failed: {}", r.error());
    }

    void ScreenOverlayRenderer::addTextWithShadow(glm::vec2 top_left_px, std::string_view text,
                                                  OverlayColor c, OverlayColor shadow_c,
                                                  float size_px, glm::vec2 shadow_offset) {
        addText(top_left_px + shadow_offset, text, shadow_c, size_px);
        addText(top_left_px, text, c, size_px);
    }

    void ScreenOverlayRenderer::flush() {
        if (verts_.empty())
            return;
        uploadAndDraw();
        verts_.clear();
    }

    void ScreenOverlayRenderer::uploadAndDraw() {
        assert(initialized_);
        if (window_w_ <= 0 || window_h_ <= 0) {
            return;
        }

        GLStateGuard state_guard;

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);
        glDisable(GL_STENCIL_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glViewport(0, 0, window_w_, window_h_);
        applyClipScissor();

        ShaderScope s(shader_);
        if (auto r = s->set("uViewportSizePx", glm::vec2(static_cast<float>(window_w_),
                                                         static_cast<float>(window_h_)));
            !r) {
            LOG_ERROR("screen_overlay: failed to set uViewportSizePx: {}", r.error());
            return;
        }

        VAOBinder vao_bind(vao_);
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);

        const std::size_t needed_bytes = verts_.size() * sizeof(Vertex);
        if (needed_bytes > vbo_capacity_bytes_) {
            std::size_t new_cap = vbo_capacity_bytes_;
            while (new_cap < needed_bytes) {
                new_cap *= 2;
            }
            glBufferData(GL_ARRAY_BUFFER, new_cap, nullptr, GL_DYNAMIC_DRAW);
            vbo_capacity_bytes_ = new_cap;
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, needed_bytes, verts_.data());
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts_.size()));
    }

} // namespace lfs::rendering
