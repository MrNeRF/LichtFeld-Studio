// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/media_probe.hpp"

namespace lfs::media {
    // Presence follows the wire representation, including omitted proto3 zeros.
    // Values remain in device coordinates/units; no UTC, GPS fix or video-clock
    // association is inferred. packet_index refers to the original observation.
    struct DjiAc204Sample {
        std::size_t packet_index = 0;
        std::optional<std::uint64_t> sequence;
        std::optional<std::uint64_t> camera_timestamp;
        std::optional<float> iso;
        std::optional<std::array<std::uint64_t, 2>> exposure_time_ratio;
        std::optional<std::uint64_t> white_balance_cct;
        std::optional<std::array<std::optional<float>, 4>> quaternion_wxyz;
        std::optional<std::array<std::optional<float>, 3>> accelerometer_xyz;
        std::optional<std::uint64_t> gps_status;
    };
    struct DjiAc204Metadata {
        std::vector<DjiAc204Sample> samples;
    };
    // Explicit, in-memory interpretation of djmd/dvtm_ac204.proto 2.0.1 only.
    // Requires the declaration in the first packet. Rejects malformed/ambiguous
    // supported fields and observations exceeding payload/packet maxima. Unknown
    // fields remain untouched in the caller-owned packets, never re-read from disk.
    [[nodiscard]] LFS_MEDIA_API Result<DjiAc204Metadata> interpretDjiAc204Metadata(
        const TimedMetadataObservation& observation);
} // namespace lfs::media
