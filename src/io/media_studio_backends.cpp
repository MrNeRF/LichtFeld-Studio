// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media_studio_backends.hpp"
#include "hdr_studio_backend.hpp"
#include "media/media_backends.hpp"
#include <mutex>
namespace lfs::io {
#if LFS_HAS_CUDA
    std::unique_ptr<media::detail::GpuJpegEncoder> createStudioJpegEncoder(const media::detail::JpegSettings&);
#endif
    void registerStudioMediaBackends() {
        static std::once_flag once;
        std::call_once(once, [] {
            media::detail::registerHdrFactory([]() -> std::unique_ptr<HdrRenderer> {
                return std::make_unique<HdrStudioRenderer>();
            });
#if LFS_HAS_CUDA
            media::detail::registerGpuJpegFactory(createStudioJpegEncoder);
#endif
        });
    }
} // namespace lfs::io
