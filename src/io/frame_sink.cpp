// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "io/media/frame_sink.hpp"
#include <cstring>
#include <limits>
#include <utility>

namespace lfs::media {
    namespace {
        SinkResult sinkError(ErrorCode code, std::string message) {
            return SinkResult::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace

    Result<std::size_t> FrameView::requiredBytes() const {
        if (layout.format != FramePixelFormat::RGB8 || layout.width <= 0 || layout.height <= 0)
            return Result<std::size_t>(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Frame requires positive RGB8 dimensions", .detection = LFS_SOURCE_SITE_CURRENT()}));
        const auto width = static_cast<std::size_t>(layout.width);
        if (width > std::numeric_limits<std::size_t>::max() / 3)
            return Result<std::size_t>(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Frame row size overflows", .detection = LFS_SOURCE_SITE_CURRENT()}));
        const auto row_bytes = width * 3;
        const auto rows = static_cast<std::size_t>(layout.height - 1);
        if (layout.row_stride < row_bytes ||
            (rows && layout.row_stride > (std::numeric_limits<std::size_t>::max() - row_bytes) / rows))
            return Result<std::size_t>(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Frame stride is invalid or overflows", .detection = LFS_SOURCE_SITE_CURRENT()}));
        const auto required = rows * layout.row_stride + row_bytes;
        if (pixels.size() < required)
            return Result<std::size_t>(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Frame pixel buffer is shorter than its layout", .detection = LFS_SOURCE_SITE_CURRENT()}));
        return required;
    }
    Result<FrameSurface> FrameSurface::copyOf(const FrameView& source) {
        const auto size = source.requiredBytes();
        if (!size)
            return Result<FrameSurface>(size.error());
        auto pixels = std::make_shared<std::vector<std::uint8_t>>(*size, 0);
        const auto row_bytes = static_cast<std::size_t>(source.layout.width) * 3;
        for (int row = 0; row < source.layout.height; ++row) {
            const auto offset = static_cast<std::size_t>(row) * source.layout.row_stride;
            std::memcpy(pixels->data() + offset, source.pixels.data() + offset, row_bytes);
        }
        FrameSurface result;
        result.layout_ = source.layout;
        result.info_ = source.info;
        result.pixels_ = std::move(pixels);
        return result;
    }
    FrameView FrameSurface::view() const {
        return {layout_, info_, pixels_ ? std::span<const std::uint8_t>(*pixels_) : std::span<const std::uint8_t>{}};
    }
    MemoryFrameSink::MemoryFrameSink(std::size_t payload_budget, std::size_t frame_limit)
        : payload_budget_(payload_budget), frame_limit_(frame_limit) {}
    SinkResult MemoryFrameSink::begin(const SinkSession&) {
        if (active_)
            return sinkError(ErrorCode::FailedPrecondition, "Memory sink already active");
        frames_.clear();
        payload_bytes_ = 0;
        outcome_.reset();
        active_ = true;
        return {};
    }
    SinkResult MemoryFrameSink::write(const FrameView& frame) {
        if (!active_)
            return sinkError(ErrorCode::FailedPrecondition, "Memory sink is not active");
        const auto required = frame.requiredBytes();
        if (!required)
            return SinkResult::failure(required.error());
        if (frames_.size() >= frame_limit_ || *required > payload_budget_ - payload_bytes_)
            return sinkError(ErrorCode::ResourceExhausted, "Memory sink payload or frame limit exceeded");
        auto snapshot = FrameSurface::copyOf(frame);
        if (!snapshot)
            return SinkResult::failure(snapshot.error());
        frames_.push_back(std::move(*snapshot));
        payload_bytes_ += *required;
        return {};
    }
    SinkResult MemoryFrameSink::complete(const SinkSummary&) {
        if (!active_)
            return sinkError(ErrorCode::FailedPrecondition, "Memory sink is not active");
        active_ = false;
        outcome_ = SinkOutcome::Completed;
        return {};
    }
    void MemoryFrameSink::abort(const SinkSummary& summary) noexcept {
        active_ = false;
        outcome_ = summary.outcome;
    }
} // namespace lfs::media
