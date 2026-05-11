/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/elements/video_preview_element.hpp"

#include <RmlUi/Core/RenderManager.h>
#include <algorithm>
#include <cstring>

namespace lfs::vis::gui {

    VideoPreviewElement::VideoPreviewElement(const Rml::String& tag) : Rml::Element(tag) {}

    VideoPreviewElement::~VideoPreviewElement() = default;

    bool VideoPreviewElement::GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) {
        dimensions = {1.f, 200.f};
        ratio = 0.f;
        return true;
    }

    void VideoPreviewElement::OnResize() { geom_dirty_ = true; }

    void VideoPreviewElement::setVideoSource(VideoSource source) {
        video_source_ = std::move(source);
    }

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

    void VideoPreviewElement::pullSourceFrame() {
        if (!video_source_.is_open || !video_source_.frame_data || !video_source_.width || !video_source_.height)
            return;
        if (!video_source_.is_open()) {
            last_frame_ptr_ = nullptr;
            return;
        }

        if (playing_ && video_source_.update) {
            video_source_.update();
        }

        const uint8_t* frame_ptr = video_source_.frame_data();
        const int w = video_source_.width();
        const int h = video_source_.height();
        if (!frame_ptr || w <= 0 || h <= 0)
            return;

        if (frame_ptr == last_frame_ptr_ && texture_width_ == w && texture_height_ == h)
            return;

        last_frame_ptr_ = frame_ptr;
        setFrameData(frame_ptr, w, h);
    }

    void VideoPreviewElement::uploadTexture() {
        std::vector<uint8_t> rgb;
        int w = 0;
        int h = 0;

        {
            std::lock_guard lock(mutex_);
            if (!has_pending_)
                return;
            rgb = std::move(pending_data_);
            w = pending_width_;
            h = pending_height_;
            has_pending_ = false;
        }

        if (rgb.empty() || w <= 0 || h <= 0)
            return;

        const size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
        std::vector<Rml::byte> rgba(pixel_count * 4);
        for (size_t i = 0; i < pixel_count; ++i) {
            rgba[i * 4 + 0] = static_cast<Rml::byte>(rgb[i * 3 + 0]);
            rgba[i * 4 + 1] = static_cast<Rml::byte>(rgb[i * 3 + 1]);
            rgba[i * 4 + 2] = static_cast<Rml::byte>(rgb[i * 3 + 2]);
            rgba[i * 4 + 3] = static_cast<Rml::byte>(255);
        }

        if (texture_width_ != w || texture_height_ != h) {
            texture_width_ = w;
            texture_height_ = h;
            geom_dirty_ = true;
        }

        const Rml::Vector2i dims(w, h);
        texture_source_ = Rml::CallbackTextureSource(
            [data = std::move(rgba), dims](const Rml::CallbackTextureInterface& iface) -> bool {
                return iface.GenerateTexture(
                    Rml::Span<const Rml::byte>(data.data(), data.size()), dims);
            });
        has_texture_ = true;
    }

    void VideoPreviewElement::rebuildGeometry() {
        auto* rm = GetRenderManager();
        if (!rm)
            return;

        const float box_w = GetBox().GetSize(Rml::BoxArea::Content).x;
        const float box_h = GetBox().GetSize(Rml::BoxArea::Content).y;
        if (box_w <= 0.f || box_h <= 0.f)
            return;

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

        if (has_texture_ && texture_width_ > 0 && texture_height_ > 0) {
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
        pullSourceFrame();
        uploadTexture();

        if (geom_dirty_)
            rebuildGeometry();

        const auto offset = GetAbsoluteOffset(Rml::BoxArea::Content);
        bg_geom_.Render(offset);

        auto* rm = GetRenderManager();
        if (rm && has_texture_ && texture_width_ > 0) {
            Rml::Texture tex = texture_source_.GetTexture(*rm);
            video_geom_.Render(offset, tex);
        }
    }

} // namespace lfs::vis::gui
