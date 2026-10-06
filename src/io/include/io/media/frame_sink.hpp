// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/export.hpp"
#include "io/media/media_probe.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace lfs::media {
    enum class FramePixelFormat { RGB8 };
    enum class TimestampOrigin { Missing,
                                 BestEffort,
                                 Presentation };
    struct FrameLayout {
        int width = 0;
        int height = 0;
        std::size_t row_stride = 0;
        FramePixelFormat format = FramePixelFormat::RGB8;
    };
    struct FrameInfo {
        std::optional<Timestamp> source_timestamp;
        TimestampOrigin timestamp_origin = TimestampOrigin::Missing;
        // Zero-based ordinal within this decode run, not a global source index after seeking.
        std::uint64_t decode_index = 0;
        std::uint64_t delivery_index = 0;
        double relative_seconds = 0;
        int legacy_source_frame = 0;
        double sharpness_score = 0;
    };
    // Read-only borrowed pixels, valid only during the synchronous sink callback.
    struct LFS_MEDIA_API FrameView {
        FrameLayout layout;
        FrameInfo info;
        std::span<const std::uint8_t> pixels;
        [[nodiscard]] Result<std::size_t> requiredBytes() const;
    };
    // Immutable owning snapshot; copies share ownership, never decoder storage.
    class LFS_MEDIA_API FrameSurface {
    public:
        [[nodiscard]] static Result<FrameSurface> copyOf(const FrameView& source);
        [[nodiscard]] FrameView view() const;

    private:
        FrameLayout layout_;
        FrameInfo info_;
        std::shared_ptr<const std::vector<std::uint8_t>> pixels_;
    };
    using SinkResult = Result<void>;
    enum class SinkOutcome { Completed,
                             Cancelled,
                             Failed };
    struct SinkSession {
        MediaDescription source;
        FrameLayout output;
        int applied_rotation = 0;
    };
    struct SinkSummary {
        SinkOutcome outcome = SinkOutcome::Failed;
        std::size_t frames_accepted = 0;
        std::string error;
    };
    // One synchronous begin, ordered writes, then complete or abort. To retain
    // pixels beyond write(), take FrameSurface::copyOf(view). No callbacks run concurrently.
    // Failed/cancelled begin or complete attempts also receive abort. abort must not throw.
    class LFS_MEDIA_API FrameSink {
    public:
        virtual ~FrameSink() = default;
        virtual SinkResult begin(const SinkSession&) { return {}; }
        virtual SinkResult write(const FrameView&) = 0;
        virtual SinkResult complete(const SinkSummary&) { return {}; }
        virtual void abort(const SinkSummary&) noexcept {}
    };
    class LFS_MEDIA_API MemoryFrameSink final : public FrameSink {
    public:
        explicit MemoryFrameSink(std::size_t payload_budget = 256ULL * 1024 * 1024,
                                 std::size_t frame_limit = 100000);
        SinkResult begin(const SinkSession&) override;
        SinkResult write(const FrameView&) override;
        SinkResult complete(const SinkSummary&) override;
        void abort(const SinkSummary&) noexcept override;
        [[nodiscard]] const std::vector<FrameSurface>& frames() const { return frames_; }
        [[nodiscard]] std::size_t payloadBytes() const { return payload_bytes_; }
        [[nodiscard]] std::optional<SinkOutcome> outcome() const { return outcome_; }

    private:
        std::size_t payload_budget_;
        std::size_t frame_limit_;
        std::size_t payload_bytes_ = 0;
        bool active_ = false;
        std::optional<SinkOutcome> outcome_;
        std::vector<FrameSurface> frames_;
    };
} // namespace lfs::media
