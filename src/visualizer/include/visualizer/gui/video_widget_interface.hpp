/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "io/video/video_encoder_interface.hpp"

#include <functional>
#include <memory>

namespace lfs::gui {

    class IVideoExtractorWidget {
    public:
        virtual ~IVideoExtractorWidget() = default;
        virtual bool render() = 0;
        [[nodiscard]] virtual bool isVideoPlaying() const = 0;
        virtual void shutdown() = 0;
    };

    using VideoWidgetFactory = std::function<std::unique_ptr<IVideoExtractorWidget>()>;
    using VideoEncoderFactory = std::function<std::unique_ptr<io::video::IVideoEncoder>()>;

    LFS_VIS_API void setVideoWidgetFactory(VideoWidgetFactory factory);
    LFS_VIS_API std::unique_ptr<IVideoExtractorWidget> createVideoWidget();

    LFS_VIS_API void setVideoEncoderFactory(VideoEncoderFactory factory);
    LFS_VIS_API std::unique_ptr<io::video::IVideoEncoder> createVideoEncoder();

} // namespace lfs::gui
