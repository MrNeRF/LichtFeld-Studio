/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/video/video_encoder_interface.hpp"

#include <filesystem>
#include <functional>
#include <memory>

namespace lfs::vis::gui {
    struct PanelDrawContext;
    struct PanelInputState;
} // namespace lfs::vis::gui

namespace lfs::gui {

    class IVideoExtractorWidget {
    public:
        virtual ~IVideoExtractorWidget() = default;
        [[nodiscard]] virtual bool isVideoPlaying() const = 0;
        [[nodiscard]] virtual bool openVideoPath(const std::filesystem::path& path) {
            (void)path;
            return false;
        }
        virtual void shutdown() = 0;

        [[nodiscard]] virtual bool supportsDirectDraw() const { return false; }
        virtual void preloadDirect(float w, float h,
                                   const lfs::vis::gui::PanelDrawContext& ctx,
                                   float clip_y_min, float clip_y_max,
                                   const lfs::vis::gui::PanelInputState* input) {
            (void)w;
            (void)h;
            (void)ctx;
            (void)clip_y_min;
            (void)clip_y_max;
            (void)input;
        }
        virtual void drawDirect(float x, float y, float w, float h,
                                const lfs::vis::gui::PanelDrawContext& ctx) {
            (void)x;
            (void)y;
            (void)w;
            (void)h;
            (void)ctx;
        }
        virtual bool drawDirectCached(float x, float y, float w, float h,
                                      const lfs::vis::gui::PanelDrawContext& ctx) {
            (void)x;
            (void)y;
            (void)w;
            (void)h;
            (void)ctx;
            return false;
        }
        [[nodiscard]] virtual float getDirectDrawHeight() const { return 0.0f; }
        virtual void setInputClipY(float y_min, float y_max) {
            (void)y_min;
            (void)y_max;
        }
        virtual void setInput(const lfs::vis::gui::PanelInputState* input) { (void)input; }
        virtual void setForcedHeight(float h) { (void)h; }
        virtual void setFloating(bool floating) { (void)floating; }
        [[nodiscard]] virtual bool wantsKeyboard() const { return false; }
        [[nodiscard]] virtual bool needsAnimationFrame() const { return isVideoPlaying(); }
        virtual void reloadRmlResources() {}
    };

    using VideoWidgetFactory = std::function<std::unique_ptr<IVideoExtractorWidget>()>;
    using VideoEncoderFactory = std::function<std::unique_ptr<io::video::IVideoEncoder>()>;

    LFS_VIS_API void setVideoWidgetFactory(VideoWidgetFactory factory);
    LFS_VIS_API std::unique_ptr<IVideoExtractorWidget> createVideoWidget();

    LFS_VIS_API void setVideoEncoderFactory(VideoEncoderFactory factory);
    LFS_VIS_API std::unique_ptr<io::video::IVideoEncoder> createVideoEncoder();

} // namespace lfs::gui
