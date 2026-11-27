// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_probe_ffmpeg.hpp"
#include "media/protobuf_wire.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/intreadwrite.h>
}
#include <algorithm>
#include <array>
#include <cerrno>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::media::detail {
    namespace {
        constexpr std::string_view magic = "8db42d694ccc418790edff439fe026bf";
        constexpr int footer_size = 72;
        constexpr int header_size = 6;
        constexpr int compact_size = 20;
        constexpr int index_entry_size = 10;
        constexpr int index_slot_count = 256;
        constexpr std::size_t trailer_record_limit = 4096;
        constexpr std::size_t layout_metadata_limit = 1024 * 1024;
        // Allow a 90-minute segment at 1 kHz, plus room for rate variation.
        constexpr std::size_t sample_limit = 6'000'000;

        Error trailerError(ErrorCode code, std::string message) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        int readAt(AVIOContext* input, std::int64_t offset, std::uint8_t* data, int size) {
            const auto position = avio_seek(input, offset, SEEK_SET);
            if (position != offset)
                return position < 0 ? static_cast<int>(position) : AVERROR(EIO);
            const int count = avio_read(input, data, size);
            if (count == size)
                return 0;
            if (count < 0 && count != AVERROR_EOF)
                return count;
            return input->error < 0 ? input->error : AVERROR_INVALIDDATA;
        }
    } // namespace

    Result<std::vector<Insta360ImuSample>> collectInsta360Imu(AVFormatContext* context) {
        auto* input = context->pb;
        if (!input || !(input->seekable & AVIO_SEEKABLE_NORMAL))
            return trailerError(ErrorCode::Unsupported, "Insta360 IMU requires a seekable input");
        const auto interrupted = [&] {
            const auto& interrupt = context->interrupt_callback;
            return interrupt.callback && interrupt.callback(interrupt.opaque);
        };
        const auto size = avio_size(input);
        if (size < 0)
            return probeFailure(static_cast<int>(size), "Determine Insta360 input size", interrupted());
        if (size < footer_size)
            return trailerError(ErrorCode::Unsupported, "Insta360 trailer is absent");

        std::array<std::uint8_t, footer_size> footer;
        int status = readAt(input, size - footer_size, footer.data(), footer_size);
        if (status < 0)
            return probeFailure(status, "Read Insta360 trailer footer", interrupted());
        if (!std::equal(magic.begin(), magic.end(), footer.begin() + 40))
            return trailerError(ErrorCode::Unsupported, "Insta360 trailer is absent");
        const auto version = AV_RL32(footer.data() + 36);
        if (version != 3)
            return trailerError(ErrorCode::Unsupported, "Unsupported Insta360 trailer version (version=" + std::to_string(version) + ", supported=3)");
        const auto length = AV_RL32(footer.data() + 32);
        if (length < footer_size || length > size)
            return trailerError(ErrorCode::DataLoss, "Invalid Insta360 trailer length (declared_bytes=" + std::to_string(length) + ", file_bytes=" + std::to_string(size) + ", footer_bytes=" + std::to_string(footer_size) + ")");

        const auto start = size - length;
        auto end = size - footer_size;
        std::int64_t imu_offset = -1;
        std::uint32_t imu_length = 0;
        std::int64_t metadata_offset = -1;
        std::uint32_t metadata_length = 0;
        const auto remember = [&](std::uint16_t tag, std::int64_t offset, std::uint32_t length) -> std::optional<Error> {
            if (tag == 0x0300 || tag == 0x0003) {
                if (imu_offset >= 0)
                    return trailerError(ErrorCode::DataLoss, "Multiple Insta360 IMU records");
                imu_offset = offset;
                imu_length = length;
            } else if (tag == 0x0101) {
                if (metadata_offset >= 0)
                    return trailerError(ErrorCode::DataLoss, "Multiple Insta360 metadata records");
                metadata_offset = offset;
                metadata_length = length;
            }
            return {};
        };

        std::size_t records = 0;
        while (end != start) {
            if (interrupted())
                return probeFailure(AVERROR_EXIT, "Scan Insta360 trailer", true);
            if (++records > trailer_record_limit)
                return trailerError(ErrorCode::ResourceExhausted, "Insta360 trailer record budget exceeded (records=" + std::to_string(records) + ", limit=" + std::to_string(trailer_record_limit) + ")");
            if (end - start < header_size)
                return trailerError(ErrorCode::DataLoss, "Incomplete Insta360 record header (remaining_bytes=" + std::to_string(end - start) + ", header_bytes=" + std::to_string(header_size) + ")");
            end -= header_size;
            std::array<std::uint8_t, header_size> header;
            status = readAt(input, end, header.data(), header_size);
            if (status < 0)
                return probeFailure(status, "Read Insta360 record header", interrupted());
            const auto payload_length = AV_RL32(header.data() + 2);
            if (payload_length > end - start)
                return trailerError(ErrorCode::DataLoss, "Insta360 record extends outside the trailer (declared_bytes=" + std::to_string(payload_length) + ", remaining_bytes=" + std::to_string(end - start) + ")");
            const auto tag = AV_RL16(header.data());
            end -= payload_length;
            if (records == 1 && tag == 0) {
                if (payload_length % index_entry_size != 0)
                    return trailerError(ErrorCode::DataLoss, "Invalid Insta360 trailer directory length");
                if (payload_length / index_entry_size > index_slot_count)
                    return trailerError(ErrorCode::Unsupported, "Unsupported Insta360 trailer directory layout");

                // A1 directory offsets are relative to the start of the trailer.
                // Slot payloads need not have a sequential record footer.
                std::array<std::uint8_t, index_slot_count * index_entry_size> index;
                status = readAt(input, end, index.data(), static_cast<int>(payload_length));
                if (status < 0)
                    return probeFailure(status, "Read Insta360 trailer directory", interrupted());
                const auto data_length = end - start;
                std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
                for (std::uint32_t slot = 0; slot < payload_length / index_entry_size; ++slot) {
                    const auto* entry = index.data() + slot * index_entry_size;
                    const auto entry_tag = AV_RL16(entry);
                    const auto length = AV_RL32(entry + 2);
                    const auto offset = AV_RL32(entry + 6);
                    if (entry_tag == 0 && length == 0 && offset == 0)
                        continue;
                    if (slot == 0 || (entry_tag & 0xff) != slot)
                        return trailerError(ErrorCode::Unsupported, "Unsupported Insta360 trailer directory layout");
                    if (length == 0 || offset > data_length || length > data_length - offset)
                        return trailerError(ErrorCode::DataLoss, "Insta360 directory record extends outside the trailer payload");
                    ranges.emplace_back(offset, offset + length);
                    if (entry_tag == 0x0003 || entry_tag == 0x0101) {
                        if (auto error = remember(entry_tag, start + offset, length))
                            return std::move(*error);
                    }
                }
                std::ranges::sort(ranges);
                for (std::size_t i = 1; i < ranges.size(); ++i) {
                    if (ranges[i].first < ranges[i - 1].second)
                        return trailerError(ErrorCode::DataLoss, "Overlapping Insta360 directory records");
                }
                break;
            }
            if (tag == 0x0300 || tag == 0x0101) {
                if (auto error = remember(tag, end, payload_length))
                    return std::move(*error);
            }
        }
        if (imu_offset < 0)
            return trailerError(ErrorCode::Unsupported, "Insta360 trailer has no IMU record");
        if (metadata_offset < 0 || metadata_length == 0)
            return trailerError(ErrorCode::Unsupported, "Insta360 compact IMU declaration is absent");
        if (metadata_length > layout_metadata_limit)
            return trailerError(ErrorCode::ResourceExhausted, "Insta360 metadata record budget exceeded (bytes=" + std::to_string(metadata_length) + ", limit=" + std::to_string(layout_metadata_limit) + ")");
        {
            std::vector<std::uint8_t> metadata(metadata_length);
            status = readAt(input, metadata_offset, metadata.data(), static_cast<int>(metadata_length));
            if (status < 0)
                return probeFailure(status, "Read Insta360 IMU layout", interrupted());
            constexpr protobuf::Reader wire;
            constexpr std::uint32_t raw_gyro_field = 62;
            try {
                const auto raw = wire.last(metadata, raw_gyro_field);
                if (!raw || !raw->uint())
                    return trailerError(ErrorCode::Unsupported, "Insta360 metadata does not declare compact IMU samples");
            } catch (const protobuf::WireFailure& error) {
                return trailerError(error.code, "Insta360: " + std::string(error.detail));
            }
        }
        if (imu_length % compact_size != 0)
            return trailerError(ErrorCode::DataLoss, "Incomplete Insta360 compact IMU sample (record_bytes=" + std::to_string(imu_length) + ", sample_bytes=" + std::to_string(compact_size) + ")");
        const auto count = imu_length / compact_size;
        if (count > sample_limit)
            return trailerError(ErrorCode::ResourceExhausted, "Insta360 IMU sample budget exceeded (samples=" + std::to_string(count) + ", limit=" + std::to_string(sample_limit) + ")");

        std::vector<Insta360ImuSample> samples;
        samples.reserve(count);
        std::array<std::uint8_t, compact_size * 256> buffer;
        while (imu_length != 0) {
            if (interrupted())
                return probeFailure(AVERROR_EXIT, "Read Insta360 IMU samples", true);
            const auto batch = std::min(imu_length, static_cast<std::uint32_t>(buffer.size()));
            status = readAt(input, imu_offset, buffer.data(), static_cast<int>(batch));
            if (status < 0)
                return probeFailure(status, "Read Insta360 IMU samples", interrupted());
            for (std::uint32_t offset = 0; offset < batch; offset += compact_size) {
                const auto* data = buffer.data() + offset;
                Insta360ImuSample sample{.device_tick = AV_RL64(data)};
                for (std::size_t axis = 0; axis < sample.acceleration_counts.size(); ++axis) {
                    sample.acceleration_counts[axis] = static_cast<int>(AV_RL16(data + 8 + axis * 2)) - 0x8000;
                    sample.angular_velocity_counts[axis] = static_cast<int>(AV_RL16(data + 14 + axis * 2)) - 0x8000;
                }
                samples.push_back(sample);
            }
            imu_offset += batch;
            imu_length -= batch;
        }
        return samples;
    }
} // namespace lfs::media::detail
