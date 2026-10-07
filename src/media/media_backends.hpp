// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/export.hpp"
#include "media/cuda_frame.hpp"
#include "media/hdr_renderer.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace lfs::media::detail {
    // Studio registers the backends it builds before starting media work.
    // The leaf library never imports tensor, CUDA or Vulkan runtime symbols.
    struct JpegSettings {
        int source_width, source_height, output_width, output_height;
        int estimated_frames, rotation;
        bool hardware_decode, needs_scale, hdr_to_sdr;
    };
    class LFS_MEDIA_API GpuJpegEncoder {
    public:
        virtual ~GpuJpegEncoder() = default;
        virtual std::size_t capacity() const noexcept = 0;
        virtual bool canConvertHardware() const noexcept = 0;
        virtual void convertHardware(const CudaVideoFrame&, std::uint8_t* optional_readback) = 0;
        virtual void finishHardware() = 0;
        virtual void* queueHardware(std::size_t index) = 0;
        virtual void* queueHost(std::size_t index, const std::uint8_t*) = 0;
        virtual std::vector<std::vector<std::uint8_t>> encode(const std::vector<void*>&, int width, int height, int quality) = 0;
    };
    using GpuJpegFactory = std::unique_ptr<GpuJpegEncoder> (*)(const JpegSettings&);
    using HdrFactory = std::unique_ptr<io::HdrRenderer> (*)();
    LFS_MEDIA_API void registerGpuJpegFactory(GpuJpegFactory) noexcept;
    LFS_MEDIA_API void registerHdrFactory(HdrFactory) noexcept;
    // Registered by GPU hosts independently of the optional JPEG encoder.
    LFS_MEDIA_API void registerCudaVideoDecodeBackend() noexcept;
    LFS_MEDIA_API bool hasCudaVideoDecodeBackend() noexcept;
    LFS_MEDIA_API bool hasGpuJpegBackend() noexcept;
    LFS_MEDIA_API bool hasHdrBackend() noexcept;
    LFS_MEDIA_API std::unique_ptr<GpuJpegEncoder> createGpuJpegEncoder(const JpegSettings&);
    LFS_MEDIA_API std::unique_ptr<io::HdrRenderer> createHdrRenderer();
} // namespace lfs::media::detail
