/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/video_preview_element.hpp"

#include <RmlUi/Core/RenderManager.h>
#include <glad/glad.h>
#include <algorithm>
#include <cstring>

namespace lfs::vis::gui {

    VideoPreviewElement::VideoPreviewElement(const Rml::String& tag) : Rml::Element(tag) {}

    VideoPreviewElement::~VideoPreviewElement() {
        if (gl_texture_ != 0)
            glDeleteTextures(1, &gl_texture_);
    }

    bool VideoPreviewElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {1.f, 200.f};
        ratio = 0.f;
        return true;
    }

    void VideoPreviewElement::OnResize() { geom_dirty_ = true; }

    void VideoPreviewElement::setFrameData(const uint8_t* rgb_data, int w, int h) {
        if (!rgb_data || w <= 0 || h <= 0)
            return;

        const size_t byte_count = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
        std::lock_guard lock(mutex_);
        pending_data_.resize(byte_count);
        std::memcpy(pending_data_.data(), rgb_data, byte_count);
        pending_width_ = w;
        pending_height_ = h;
        has_pending_ = true;
    }

    void VideoPreviewElement::uploadTexture() {
        std::vector<uint8_t> data;
        int w = 0;
        int h = 0;

        {
            std::lock_guard lock(mutex_);
            if (!has_pending_)
                return;
            data = std::move(pending_data_);
            w = pending_width_;
            h = pending_height_;
            has_pending_ = false;
        }

        if (data.empty())
            return;

        if (gl_texture_ == 0) {
            glGenTextures(1, &gl_texture_);
            glBindTexture(GL_TEXTURE_2D, gl_texture_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        glBindTexture(GL_TEXTURE_2D, gl_texture_);

        const bool size_changed = (texture_width_ != w || texture_height_ != h);
        if (size_changed) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
            texture_width_ = w;
            texture_height_ = h;
            geom_dirty_ = true;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, data.data());
        }

        glBindTexture(GL_TEXTURE_2D, 0);

        // Only rebuild the callback texture source when texture or size changes
        if (size_changed) {
            auto* rm = GetRenderManager();
            if (rm) {
                const uint32_t tex_id = gl_texture_;
                const int tw = texture_width_;
                const int th = texture_height_;
                texture_source_ = Rml::CallbackTextureSource(
                    [tex_id, tw, th](const Rml::CallbackTextureInterface& iface) -> bool {
                        iface.SetTextureHandle(static_cast<Rml::TextureHandle>(tex_id),
                                               Rml::Vector2i(tw, th));
                        return true;
                    });
            }
        }
    }

    void VideoPreviewElement::rebuildGeometry() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float box_w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float box_h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (box_w <= 0.f || box_h <= 0.f)
            return;

        // Background quad (always drawn)
        {
            Rml::Mesh mesh;
            Rml::ColourbPremultiplied bg(20, 20, 20, 255);
            mesh.vertices.push_back({{0, 0}, bg, {0, 0}});
            mesh.vertices.push_back({{box_w, 0}, bg, {0, 0}});
            mesh.vertices.push_back({{box_w, box_h}, bg, {0, 0}});
            mesh.vertices.push_back({{0, box_h}, bg, {0, 0}});
            mesh.indices = {0, 1, 2, 0, 2, 3};
            bg_geom_ = rm->MakeGeometry(std::move(mesh));
        }

        // Video quad (only when we have a texture)
        if (gl_texture_ != 0 && texture_width_ > 0 && texture_height_ > 0) {
            const float video_aspect =
                static_cast<float>(texture_width_) / static_cast<float>(texture_height_);
            const float box_aspect = box_w / box_h;

            float display_w, display_h;
            if (video_aspect > box_aspect) {
                display_w = box_w;
                display_h = box_w / video_aspect;
            } else {
                display_h = box_h;
                display_w = box_h * video_aspect;
            }

            const float offset_x = (box_w - display_w) * 0.5f;
            const float offset_y = (box_h - display_h) * 0.5f;

            Rml::Mesh mesh;
            Rml::ColourbPremultiplied white(255, 255, 255, 255);
            mesh.vertices.push_back({{offset_x, offset_y}, white, {0.f, 0.f}});
            mesh.vertices.push_back({{offset_x + display_w, offset_y}, white, {1.f, 0.f}});
            mesh.vertices.push_back(
                {{offset_x + display_w, offset_y + display_h}, white, {1.f, 1.f}});
            mesh.vertices.push_back({{offset_x, offset_y + display_h}, white, {0.f, 1.f}});
            mesh.indices = {0, 1, 2, 0, 2, 3};
            video_geom_ = rm->MakeGeometry(std::move(mesh));
        } else {
            video_geom_ = {};
        }

        geom_dirty_ = false;
    }

    void VideoPreviewElement::OnRender() {
        uploadTexture();

        if (geom_dirty_)
            rebuildGeometry();

        const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);

        // Draw dark background
        bg_geom_.Render(offset);

        // Draw video frame with texture
        auto* rm = GetRenderManager();
        if (rm && gl_texture_ != 0 && texture_width_ > 0) {
            Rml::Texture tex = texture_source_.GetTexture(*rm);
            video_geom_.Render(offset, tex);
        }
    }

} // namespace lfs::vis::gui
