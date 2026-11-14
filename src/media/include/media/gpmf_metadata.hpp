// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/media_probe.hpp"

namespace lfs::media {
    struct GpmfStream {
        std::optional<std::uint64_t> device_id;
        std::string device_name;
        std::string key;
        std::string name;
        // One fixed-width byte string for all components, or one per component.
        std::vector<std::string> units;
        std::string orientation_in;
        std::string orientation_out;
        std::vector<double> calibration_matrix;
        std::optional<std::uint64_t> total_samples;
        std::optional<std::uint64_t> sensor_timestamp;
        std::optional<std::uint64_t> gps_fix;
        std::optional<std::uint64_t> gps_precision;
        std::string gps_utc;
        std::uint32_t components = 0;
        std::vector<double> values;
    };
    struct GpmfPacket {
        std::optional<Timestamp> pts;
        std::optional<Timestamp> duration;
        std::vector<GpmfStream> streams;
    };
    inline constexpr std::size_t gpmf_default_decoded_bytes = 128u * 1024 * 1024;
    // Decode ACCL, GYRO and GPS5 streams from collected gpmd packets.
    // SCAL is applied; sample axes and calibration descriptors remain as recorded.
    // Packet clocks and sensor counters remain distinct from individual sample times.
    [[nodiscard]] LFS_MEDIA_API Result<std::vector<GpmfPacket>> interpretGpmfMetadata(const TimedMetadataObservation&, std::size_t decoded_byte_limit = gpmf_default_decoded_bytes);
} // namespace lfs::media
