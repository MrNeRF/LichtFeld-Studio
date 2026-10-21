// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "io/image_sequence.hpp"
#include "core/color_transfer.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

namespace lfs::io {
    namespace {
        media::SinkResult fail(ErrorCode code, std::string detail) {
            return media::SinkResult::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace
    media::SinkResult TensorImageSequence::open(const TensorImageSequenceOptions& options) {
        if (sequence_.isOpen())
            return fail(ErrorCode::FailedPrecondition, "Tensor image sequence already open");
        const auto& layout = options.output.layout;
        const auto& color = options.input_color;
        if ((color.transfer != media::ColorTransfer::Linear && color.transfer != media::ColorTransfer::Srgb) ||
            color.primaries != layout.color.primaries || color.alpha != layout.color.alpha ||
            color.units != layout.color.units ||
            (color.transfer == media::ColorTransfer::Srgb && (color.primaries != media::ColorPrimaries::Bt709 ||
                                                              color.units != media::ColorUnits::Relative || color.alpha == media::AlphaMode::Premultiplied)))
            return fail(ErrorCode::Unsupported, "Tensor sequence requires explicit compatible linear/sRGB color, units and alpha");
        if (layout.width <= 0 || layout.height <= 0)
            return fail(ErrorCode::InvalidArgument, "Tensor sequence dimensions must be positive");
        const auto channels = layout.format == media::FramePixelFormat::RGBAFloat32 ? 4ULL : 3ULL;
        const auto components = static_cast<uint64_t>(layout.width) * static_cast<uint64_t>(layout.height) * channels;
        if (components > options.readback_budget / sizeof(float) || components > std::numeric_limits<std::size_t>::max() / sizeof(float))
            return fail(ErrorCode::ResourceExhausted, "Tensor sequence exceeds the per-frame readback budget");
        try {
            options_ = options;
            linear_.clear();
            bytes_.clear();
            return sequence_.open(options.output);
        } catch (const std::bad_alloc&) {
            return fail(ErrorCode::ResourceExhausted, "Tensor sequence setup allocation failed");
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): Setup failures retain their detail at the typed host boundary.
            return fail(ErrorCode::Internal, std::string("Tensor sequence setup failed: ") + error.what());
        }
    }
    media::SinkResult TensorImageSequence::write(const core::Tensor& image) {
        const auto write = [&]() -> media::SinkResult {
            if (!sequence_.isOpen())
                return fail(ErrorCode::FailedPrecondition, "Tensor image sequence is not open");
            const auto& output = options_.output;
            const auto& layout = output.layout;
            const size_t channels = layout.format == media::FramePixelFormat::RGBAFloat32 ? 4 : 3;
            if (!image.is_valid() || image.ndim() != 3 ||
                (image.dtype() != core::DataType::Float32 && image.dtype() != core::DataType::Float16) ||
                image.size(0) != static_cast<size_t>(layout.height) || image.size(1) != static_cast<size_t>(layout.width) || image.size(2) != channels)
                return fail(ErrorCode::InvalidArgument, "Tensor sequence frame requires matching HWC Float16/Float32 RGB/RGBA");
            if (output.cancelled && output.cancelled())
                return fail(ErrorCode::Cancelled, "Tensor image sequence cancelled before readback");
            const auto pixels = image.to(core::DataType::Float32).cpu().contiguous();
            const float* samples = pixels.ptr<float>();
            const auto count = pixels.numel();
            const bool input_linear = options_.input_color.transfer == media::ColorTransfer::Linear;
            const bool rgb8 = layout.format == media::FramePixelFormat::RGB8;
            if (rgb8)
                bytes_.resize(count);
            else if (!input_linear)
                linear_.resize(count);
            for (size_t i = 0; i < count; ++i) {
                const auto value = samples[i];
                if (!std::isfinite(value))
                    return fail(ErrorCode::InvalidArgument, "Tensor sequence contains a nonfinite sample");
                if (rgb8) {
                    const auto encoded = input_linear ? core::color::linearToSrgb(value) : value;
                    bytes_[i] = static_cast<uint8_t>(std::lround(std::clamp(encoded, 0.f, 1.f) * 255.f));
                } else if (!input_linear) {
                    linear_[i] = channels == 4 && i % 4 == 3 ? value : core::color::srgbToLinear(value);
                    if (!std::isfinite(linear_[i]))
                        return fail(ErrorCode::InvalidArgument, "Tensor sequence transfer conversion overflowed");
                }
            }
            auto packed_layout = layout;
            packed_layout.row_stride = static_cast<size_t>(layout.width) * media::pixelBytes(layout.format);
            const std::span<const uint8_t> bytes = rgb8
                                                       ? std::span<const uint8_t>(bytes_)
                                                       : std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(input_linear ? samples : linear_.data()), count * sizeof(float));
            return sequence_.write({packed_layout, {}, bytes});
        };
        auto result = [&]() -> media::SinkResult {
            try {
                return write();
            } catch (const Exception& error) {
                return media::SinkResult::failure(error.error());
            } catch (const std::bad_alloc&) {
                return fail(ErrorCode::ResourceExhausted, "Tensor sequence readback allocation failed");
            } catch (const std::exception& error) {
                // LFS-CENSUS-OK(empty-catch): Readback failures retain their detail and abort the output below.
                return fail(ErrorCode::Internal, std::string("Tensor sequence readback failed: ") + error.what());
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): A nonstandard callback failure is an error, never a completed frame.
                return fail(ErrorCode::Internal, "Tensor sequence readback failed with a nonstandard exception");
            }
        }();
        if (!result)
            sequence_.abort();
        return result;
    }
} // namespace lfs::io
