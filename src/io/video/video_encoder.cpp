/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "video_encoder.hpp"
#include "core/error.hpp"
#include "core/error_reporter.hpp"
#include "core/logger.hpp"
#include "core/provenance.hpp"
#include "core/tensor_backend.hpp"
#include "media/video_encode_session.hpp"
#include <array>
#include <exception>
#include <format>
#if LFS_HAS_CUDA
#include "color_convert.cuh"
#include <cuda_runtime.h>
#endif
namespace lfs::io::video {
    namespace {
        struct YuvPlanes {
            core::Tensor y;
            core::Tensor u;
            core::Tensor v;
        };

        YuvPlanes rgbToYuv420p(const core::Tensor& rgb) {
            const int height = static_cast<int>(rgb.size(0));
            const int width = static_cast<int>(rgb.size(1));
#if LFS_HAS_CUDA
            if (core::gpu_backend_of(rgb) == core::GpuBackend::CUDA) {
                const auto* pixels = rgb.ptr<float>();
                const auto stream = rgb.stream();
                auto y = core::Tensor::empty_like(rgb, {static_cast<size_t>(height), static_cast<size_t>(width)}, core::DataType::UInt8);
                auto u = core::Tensor::empty_like(rgb, {static_cast<size_t>(height / 2), static_cast<size_t>(width / 2)}, core::DataType::UInt8);
                auto v = core::Tensor::empty_like(rgb, {static_cast<size_t>(height / 2), static_cast<size_t>(width / 2)}, core::DataType::UInt8);
                y.set_stream(stream);
                u.set_stream(stream);
                v.set_stream(stream);
                rgbToYuv420pCuda(pixels, y.ptr<uint8_t>(), u.ptr<uint8_t>(), v.ptr<uint8_t>(),
                                 width, height, stream);
                return {std::move(y), std::move(u), std::move(v)};
            }
#endif
            const auto bytes = (rgb.clamp(0.0f, 1.0f) * 255.0f + 0.5f).floor();
            const auto channel = [](const core::Tensor& image, const size_t c) {
                return image.slice(2, c, c + 1).reshape({static_cast<int>(image.size(0)), static_cast<int>(image.size(1))});
            };
            const auto y = ((channel(bytes, 0) * 66.0f +
                             channel(bytes, 1) * 129.0f +
                             channel(bytes, 2) * 25.0f + 128.0f) /
                            256.0f)
                               .floor()
                               .add(16.0f)
                               .to(core::DataType::UInt8);
            const auto chroma = (bytes.reshape({height / 2, 2, width / 2, 2, 3})
                                     .sum({1, 3}) /
                                 4.0f)
                                    .floor();
            const auto u = ((channel(chroma, 0) * -38.0f +
                             channel(chroma, 1) * -74.0f +
                             channel(chroma, 2) * 112.0f + 128.0f) /
                            256.0f)
                               .floor()
                               .add(128.0f)
                               .clamp(0.0f, 255.0f)
                               .to(core::DataType::UInt8);
            const auto v = ((channel(chroma, 0) * 112.0f +
                             channel(chroma, 1) * -94.0f +
                             channel(chroma, 2) * -18.0f + 128.0f) /
                            256.0f)
                               .floor()
                               .add(128.0f)
                               .clamp(0.0f, 255.0f)
                               .to(core::DataType::UInt8);
            return {y, u, v};
        }

        lfs::Result<void> transferError(std::string text) {
            return lfs::Result<void>::failure(lfs::make_error({.code = lfs::ErrorCode::Unavailable,
                                                               .domain = lfs::ErrorDomain::IO,
                                                               .detail = std::move(text),
                                                               .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        struct PlaneWriter final : media::VideoEncodeWriter {
            const YuvPlanes& planes;
            int width, height;
            std::exception_ptr producer_error;
            PlaneWriter(const YuvPlanes& value, int w, int h) : planes(value), width(w), height(h) {}
            lfs::Result<void> write(const media::VideoEncodeTarget& target) override {
                try {
                    return writePlanes(target);
                } catch (const lfs::Exception&) {
                    // Keep Studio's owner report and legacy message at its boundary.
                    producer_error = std::current_exception();
                    throw;
                }
            }
            lfs::Result<void> writePlanes(const media::VideoEncodeTarget& target) {
                if (target.backend == media::VideoEncodeBackend::Cuda) {
#if LFS_HAS_CUDA
                    if (core::gpu_backend_of(planes.y) != core::GpuBackend::CUDA)
                        return transferError("NVENC requires a CUDA tensor frame");
                    const auto stream = planes.y.stream();
                    auto y = core::Tensor::from_blob(target.planes[0].data,
                                                     {static_cast<size_t>(height), target.planes[0].row_stride},
                                                     core::Device::GPU, core::DataType::UInt8, stream);
                    auto uv = core::Tensor::from_blob(target.planes[1].data,
                                                      {static_cast<size_t>(height / 2), target.planes[1].row_stride},
                                                      core::Device::GPU, core::DataType::UInt8, stream);
                    y.slice(1, 0, width).copy_from(planes.y);
                    uv.slice(1, 0, width).copy_from(core::Tensor::stack({planes.u, planes.v}, 2).reshape({height / 2, width}));
                    const auto status = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
                    if (status != cudaSuccess)
                        return transferError(std::format("NVENC frame synchronization failed: {} ({})",
                                                         cudaGetErrorString(status), cudaGetErrorName(status)));
                    return {};
#else
                    return transferError("CUDA video frames are not supported by this producer");
#endif
                }
                const auto source = std::array{planes.y.to_pageable_host(), planes.u.to_pageable_host(), planes.v.to_pageable_host()};
                for (int plane = 0; plane < 3; ++plane) {
                    const auto& output = target.planes[plane];
                    auto pixels = core::Tensor::from_blob(output.data,
                                                          {static_cast<size_t>(output.height), output.row_stride},
                                                          core::Device::CPU, core::DataType::UInt8);
                    pixels.slice(1, 0, output.width).copy_from(source[plane]);
                }
                return {};
            }
        };
        std::expected<void, std::string> legacyResult(const lfs::Result<void>& result) {
            return result ? std::expected<void, std::string>{} : std::unexpected(std::string(result.error().detail()));
        }
    } // namespace
    class VideoEncoderImpl {
        media::VideoEncodeSession session_;
        int width_ = 0, height_ = 0;
        int64_t frame_count_ = 0;

    public:
        std::expected<void, std::string> open(const std::filesystem::path& path, const VideoExportOptions& options) {
            if (session_.isOpen())
                return std::unexpected("Encoder is already open");
            if (auto result = validateVideoEncodingOptions(options); !result)
                return result;
            auto stamp = options.provenance ? *options.provenance : core::make_minimal_provenance_stamp();
            media::VideoEncodeOptions request{.width = options.width, .height = options.height, .framerate = options.framerate, .crf = options.crf, .comment = core::provenance_to_json(stamp)};
#if LFS_HAS_CUDA
            if (core::default_gpu_backend() == core::GpuBackend::CUDA)
                request.preferred_backend = media::VideoEncodeBackend::Cuda;
#elif defined(__APPLE__)
            request.preferred_backend = media::VideoEncodeBackend::VideoToolbox;
#endif
            if (auto result = session_.open(path, request); !result)
                return legacyResult(result);
            width_ = options.width;
            height_ = options.height;
            frame_count_ = 0;
            return {};
        }
        std::expected<void, std::string> writeFrame(const core::Tensor& rgb_hwc) {
            if (!session_.isOpen())
                return std::unexpected("Encoder not open");
            if (!rgb_hwc.is_valid() || rgb_hwc.dtype() != core::DataType::Float32 ||
                rgb_hwc.ndim() != 3 || rgb_hwc.size(2) != 3)
                return std::unexpected("Video frame must be an HWC float32 RGB tensor");
            if (rgb_hwc.size(1) != static_cast<size_t>(width_) || rgb_hwc.size(0) != static_cast<size_t>(height_))
                return std::unexpected("Frame size mismatch");
            try {
                const auto frame = session_.backend() == media::VideoEncodeBackend::Cuda && rgb_hwc.device() == core::Device::CPU
                                       ? rgb_hwc.gpu()
                                       : rgb_hwc;
                const auto planes = rgbToYuv420p(frame.contiguous());
                PlaneWriter writer(planes, width_, height_);
                auto result = session_.writeFrame(writer);
                if (writer.producer_error)
                    std::rethrow_exception(writer.producer_error);
                if (!result)
                    return legacyResult(result);
                ++frame_count_;
                return {};
            } catch (const lfs::Exception& e) {
                auto error = lfs::Error(e.error()).with_context("write video frame", LFS_SOURCE_SITE_CURRENT(),
                                                                lfs::SmallFields{}.add("width", static_cast<int64_t>(width_)).add("height", static_cast<int64_t>(height_)).add("frame", frame_count_));
                lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
                return std::unexpected(std::string(e.what()));
            } catch (const std::exception& e) {
                return std::unexpected(std::string(e.what()));
            }
        }
        std::expected<void, std::string> close() { return legacyResult(session_.close()); }
        bool isOpen() const { return session_.isOpen(); }
        media::VideoEncodeBackend backend() const { return session_.backend(); }
    };
    VideoEncoder::VideoEncoder() : impl_(std::make_unique<VideoEncoderImpl>()) {}
    VideoEncoder::~VideoEncoder() = default;
    VideoEncoder::VideoEncoder(VideoEncoder&&) noexcept = default;
    VideoEncoder& VideoEncoder::operator=(VideoEncoder&&) noexcept = default;

    std::expected<void, std::string> VideoEncoder::open(
        const std::filesystem::path& path, const VideoExportOptions& opts) {
        return impl_->open(path, opts);
    }

    std::expected<void, std::string> VideoEncoder::writeFrame(
        std::span<const uint8_t> rgba_data, const int width, const int height) {
        if (width <= 0 || height <= 0 ||
            rgba_data.size() != static_cast<size_t>(width) * height * 4) {
            return std::unexpected("CPU RGBA frame size mismatch");
        }
        auto rgba = core::Tensor::from_blob(const_cast<uint8_t*>(rgba_data.data()),
                                            {static_cast<size_t>(height), static_cast<size_t>(width), 4},
                                            core::Device::CPU, core::DataType::UInt8);
        auto rgb = rgba.slice(2, 0, 3).to(core::DataType::Float32).div(255.0f);
        return impl_->writeFrame(rgb);
    }

    std::expected<void, std::string> VideoEncoder::writeFrame(const core::Tensor& rgb_hwc) {
        return impl_->writeFrame(rgb_hwc);
    }

    std::expected<void, std::string> VideoEncoder::close() {
        return impl_->close();
    }

    bool VideoEncoder::isOpen() const {
        return impl_->isOpen();
    }

    media::VideoEncodeBackend VideoEncoder::backend() const {
        return impl_->backend();
    }

} // namespace lfs::io::video
