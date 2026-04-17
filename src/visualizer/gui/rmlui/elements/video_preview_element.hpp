/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/CallbackTexture.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/Geometry.h>
#include <core/export.hpp>

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace lfs::vis::gui {

    class VideoPreviewElement : public Rml::Element {
    public:
        explicit VideoPreviewElement(const Rml::String& tag);
        ~VideoPreviewElement() override;

        LFS_VIS_API void setFrameData(const uint8_t* rgb_data, int w, int h);

        // Direct C++ video source — bypasses Python for frame updates
        struct VideoSource {
            std::function<bool()> update;
            std::function<const uint8_t*()> frame_data;
            std::function<int()> width;
            std::function<int()> height;
            std::function<bool()> is_open;
        };
        LFS_VIS_API void setVideoSource(VideoSource source);
        LFS_VIS_API void setPlaying(bool playing) { playing_ = playing; }

    protected:
        void OnRender() override;
        void OnResize() override;
        bool GetIntrinsicDimensions(Rml::Vector2f& dimensions, float& ratio) override;

    private:
        void pullSourceFrame();
        void uploadTexture();
        void rebuildGeometry();

        std::mutex mutex_;
        std::vector<uint8_t> pending_data_;
        int pending_width_ = 0;
        int pending_height_ = 0;
        bool has_pending_ = false;

        uint32_t gl_texture_ = 0;
        int texture_width_ = 0;
        int texture_height_ = 0;

        Rml::Geometry bg_geom_;
        Rml::Geometry video_geom_;
        Rml::CallbackTextureSource texture_source_;
        bool geom_dirty_ = true;

        VideoSource video_source_;
        bool playing_ = false;
        const uint8_t* last_frame_ptr_ = nullptr;
    };

} // namespace lfs::vis::gui
