// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "io/image_output.hpp"
#include <exception>
#include <format>
#include <limits>
#include <new>

namespace lfs::io {
    namespace {
        media::SinkResult imageError(ErrorCode code, std::string detail) {
            return media::SinkResult::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace
    media::SinkResult writeExrImage(const std::filesystem::path& path, const core::Tensor& image,
                                    media::FrameColor color, const media::ExrOutputOptions& options,
                                    const media::FrameInfo& info) {
        try {
            if (!image.is_valid() || image.ndim() != 3 || !image.size(0) || !image.size(1) ||
                image.size(0) > size_t(std::numeric_limits<int>::max()) || image.size(1) > size_t(std::numeric_limits<int>::max()) ||
                (image.size(2) != 3 && image.size(2) != 4) ||
                (image.dtype() != core::DataType::Float16 && image.dtype() != core::DataType::Float32) ||
                image.numel() > std::numeric_limits<size_t>::max() / sizeof(float))
                return imageError(ErrorCode::InvalidArgument, std::format("EXR image requires positive HWC Float16/Float32 RGB/RGBA (valid={}, shape={}, dtype={})", image.is_valid(), image.shape().str(), int(image.dtype())));
            if (options.cancelled && options.cancelled())
                return imageError(ErrorCode::Cancelled, "EXR tensor output cancelled before readback");
            // Cast where the tensor lives, then use its existing synchronized
            // CPU transfer. No FrameSurface copy or intermediate byte image.
            const auto pixels = image.to(core::DataType::Float32).cpu().contiguous();
            const auto channels = pixels.size(2);
            const media::FrameView frame{
                {int(pixels.size(1)), int(pixels.size(0)), pixels.size(1) * channels * sizeof(float),
                 channels == 4 ? media::FramePixelFormat::RGBAFloat32 : media::FramePixelFormat::RGBFloat32, color},
                info,
                {reinterpret_cast<const uint8_t*>(pixels.ptr<float>()), pixels.numel() * sizeof(float)}};
            return media::ImageOutput::writeExr(path, frame, options);
        } catch (const Exception& error) {
            return media::SinkResult::failure(error.error());
        } catch (const std::bad_alloc&) {
            return imageError(ErrorCode::ResourceExhausted, "EXR tensor output allocation failed");
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): preserve exception detail in a typed error; the shared writer owns target cleanup.
            return imageError(ErrorCode::Internal, std::format("EXR tensor output failed (exception={})", error.what()));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): callback/transfer failures become structured errors, never a successful write.
            return imageError(ErrorCode::Internal, "EXR tensor output failed (nonstandard_exception=true)");
        }
    }
} // namespace lfs::io
