// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media_backends.hpp"
#include <atomic>
namespace lfs::media::detail {
    namespace {
        std::atomic<bool> cuda_video_decode{false};
        std::atomic<GpuJpegFactory> jpeg_factory{nullptr};
        std::atomic<LinearVideoFactory> linear_factory{nullptr};
        std::atomic<HdrFactory> hdr_factory{nullptr};
    } // namespace
    void registerGpuJpegFactory(GpuJpegFactory value) noexcept { jpeg_factory.store(value, std::memory_order_release); }
    void registerLinearVideoFactory(LinearVideoFactory value) noexcept { linear_factory.store(value, std::memory_order_release); }
    std::unique_ptr<LinearVideoRenderer> createLinearVideoRenderer() {
        const auto factory = linear_factory.load(std::memory_order_acquire);
        return factory ? factory() : nullptr;
    }
    void registerHdrFactory(HdrFactory value) noexcept { hdr_factory.store(value, std::memory_order_release); }
    void registerCudaVideoDecodeBackend() noexcept { cuda_video_decode.store(true, std::memory_order_release); }
    bool hasCudaVideoDecodeBackend() noexcept { return cuda_video_decode.load(std::memory_order_acquire); }
    bool hasGpuJpegBackend() noexcept { return jpeg_factory.load(std::memory_order_acquire) != nullptr; }
    bool hasHdrBackend() noexcept { return hdr_factory.load(std::memory_order_acquire) != nullptr; }
    std::unique_ptr<GpuJpegEncoder> createGpuJpegEncoder(const JpegSettings& settings) {
        const auto factory = jpeg_factory.load(std::memory_order_acquire);
        return factory ? factory(settings) : nullptr;
    }
    std::unique_ptr<io::HdrRenderer> createHdrRenderer() {
        const auto factory = hdr_factory.load(std::memory_order_acquire);
        return factory ? factory() : nullptr;
    }
} // namespace lfs::media::detail
