/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Builds without CUDA: SAM2 video masking has no port to the other tensor backends.

#include "preprocessing/video_masking.hpp"
#include "core/source_site.hpp"

namespace lfs::preprocessing {

    namespace {
        Error unsupported() {
            return make_error({.code = ErrorCode::Unsupported,
                               .domain = ErrorDomain::Core,
                               .user_message = "Video masking requires the CUDA tensor backend",
                               .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    } // namespace

    std::expected<VideoMaskResourcePlan, Error> plan_video_mask_resources(
        std::span<const VideoMaskFrame>, const VideoMaskOptions&, size_t) {
        return std::unexpected(unsupported());
    }

    std::expected<VideoMaskReport, Error> mask_video_sequence(
        std::span<const VideoMaskFrame>, const ImageCircle&, const VideoMaskOptions&, cudaStream_t) {
        return std::unexpected(unsupported());
    }

} // namespace lfs::preprocessing
