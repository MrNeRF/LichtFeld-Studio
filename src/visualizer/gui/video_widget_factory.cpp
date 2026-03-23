/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/gui/video_widget_interface.hpp"

namespace lfs::gui {

    static VideoWidgetFactory g_video_widget_factory;
    static VideoEncoderFactory g_video_encoder_factory;

    void setVideoWidgetFactory(VideoWidgetFactory factory) {
        g_video_widget_factory = std::move(factory);
    }

    std::unique_ptr<IVideoExtractorWidget> createVideoWidget() {
        return g_video_widget_factory ? g_video_widget_factory() : nullptr;
    }

    void setVideoEncoderFactory(VideoEncoderFactory factory) {
        g_video_encoder_factory = std::move(factory);
    }

    std::unique_ptr<io::video::IVideoEncoder> createVideoEncoder() {
        return g_video_encoder_factory ? g_video_encoder_factory() : nullptr;
    }

} // namespace lfs::gui
