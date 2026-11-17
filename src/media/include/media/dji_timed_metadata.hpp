// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/media_probe.hpp"
#include <memory>
#include <span>
#include <utility>

namespace lfs::media {
    struct DjiSample {
        std::uint64_t sequence = 0;
        std::uint64_t camera_timestamp = 0; // Microseconds since camera power-up.
        std::optional<float> iso;
        std::optional<std::array<std::uint64_t, 2>> exposure_time_ratio;
        std::optional<std::uint64_t> white_balance_cct;
        std::optional<std::array<float, 4>> quaternion_wxyz;
        std::optional<std::array<float, 3>> accelerometer_xyz;
        std::optional<std::uint64_t> gps_status;
    };
    // Interpret collected djmd packets declaring dvtm_ac203/204.proto 2.0.1.
    // One sample per packet; optional values track the containing message.
    [[nodiscard]] LFS_MEDIA_API Result<std::vector<DjiSample>> interpretDjiMetadata(
        const TimedMetadataObservation& observation);
    struct DjiFrameSample {
        Timestamp pts;
        DjiSample metadata;
        std::optional<std::int64_t> source_offset;
        std::vector<std::uint8_t> bytes;
    };
    struct DjiFrameStorage;
    // Bind a collected track to exact presentation timestamps from the same container.
    // Copies share immutable samples; missing or non-increasing PTS are refused.
    class LFS_MEDIA_API DjiFrameMetadata {
    public:
        [[nodiscard]] static Result<DjiFrameMetadata> create(const TimedMetadataObservation&);
        [[nodiscard]] int streamIndex() const noexcept;
        [[nodiscard]] std::span<const DjiFrameSample> samples() const noexcept;
        [[nodiscard]] Result<std::optional<std::size_t>> lookup(Timestamp) const;

    private:
        explicit DjiFrameMetadata(std::shared_ptr<const DjiFrameStorage> storage) : storage_(std::move(storage)) {}
        std::shared_ptr<const DjiFrameStorage> storage_;
    };
} // namespace lfs::media
