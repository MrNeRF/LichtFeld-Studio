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
#include <string_view>

namespace lfs::media::detail {
    namespace {
        constexpr std::string_view magic = "8db42d694ccc418790edff439fe026bf";
        constexpr int footer_size = 72;
        constexpr int header_size = 6;
        constexpr int compact_size = 20;

        Error trailerError(ErrorCode code, const char* message) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = message, .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        int readAt(AVIOContext* input, std::int64_t offset, std::uint8_t* data, int size) {
            const auto position = avio_seek(input, offset, SEEK_SET);
            if (position != offset)
                return position < 0 ? static_cast<int>(position) : AVERROR(EIO);
            const int count = avio_read(input, data, size);
            return count == size ? 0 : count < 0 ? count
                                                 : AVERROR_INVALIDDATA;
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
        if (AV_RL32(footer.data() + 36) != 3)
            return trailerError(ErrorCode::Unsupported, "Unsupported Insta360 trailer version");
        const auto length = AV_RL32(footer.data() + 32);
        if (length < footer_size || length > size)
            return trailerError(ErrorCode::DataLoss, "Invalid Insta360 trailer length");

        const auto start = size - length;
        auto end = size - footer_size;
        std::int64_t imu_offset = -1;
        std::uint32_t imu_length = 0;
        std::int64_t metadata_offset = -1;
        std::uint32_t metadata_length = 0;
        std::size_t records = 0;
        while (end != start) {
            if (interrupted())
                return probeFailure(AVERROR_EXIT, "Scan Insta360 trailer", true);
            if (++records > timed_metadata_maxima.records)
                return trailerError(ErrorCode::ResourceExhausted, "Insta360 trailer record budget exceeded");
            if (end - start < header_size)
                return trailerError(ErrorCode::DataLoss, "Incomplete Insta360 record header");
            end -= header_size;
            std::array<std::uint8_t, header_size> header;
            status = readAt(input, end, header.data(), header_size);
            if (status < 0)
                return probeFailure(status, "Read Insta360 record header", interrupted());
            const auto payload_length = AV_RL32(header.data() + 2);
            if (payload_length > end - start)
                return trailerError(ErrorCode::DataLoss, "Insta360 record extends outside the trailer");
            const auto tag = AV_RL16(header.data());
            if (records == 1 && tag == 0) {
                if (payload_length % 10 != 0)
                    return trailerError(ErrorCode::DataLoss, "Invalid Insta360 trailer directory length");
                return trailerError(ErrorCode::Unsupported, "Indexed Insta360 trailers are unsupported");
            }
            end -= payload_length;
            if (tag == 0x0300) {
                if (imu_offset >= 0)
                    return trailerError(ErrorCode::DataLoss, "Multiple Insta360 IMU records");
                imu_offset = end;
                imu_length = payload_length;
            } else if (tag == 0x0101) {
                if (metadata_offset >= 0)
                    return trailerError(ErrorCode::DataLoss, "Multiple Insta360 metadata records");
                metadata_offset = end;
                metadata_length = payload_length;
            }
        }
        if (imu_offset < 0)
            return trailerError(ErrorCode::Unsupported, "Insta360 trailer has no IMU record");
        if (metadata_offset < 0 || metadata_length == 0)
            return trailerError(ErrorCode::Unsupported, "Insta360 compact IMU declaration is absent");
        if (metadata_length > timed_metadata_maxima.bytes)
            return trailerError(ErrorCode::ResourceExhausted, "Insta360 metadata record budget exceeded");
        {
            std::vector<std::uint8_t> metadata(metadata_length);
            status = readAt(input, metadata_offset, metadata.data(), static_cast<int>(metadata_length));
            if (status < 0)
                return probeFailure(status, "Read Insta360 IMU layout", interrupted());
            constexpr protobuf::Reader wire{"Insta360"};
            constexpr std::uint32_t raw_gyro_field = 62;
            try {
                const auto raw = wire.last(metadata, raw_gyro_field);
                if (!raw || !raw->uint())
                    return trailerError(ErrorCode::Unsupported, "Insta360 metadata does not declare compact IMU samples");
            } catch (const protobuf::Failure& error) {
                return trailerError(error.code, error.detail.c_str());
            }
        }
        if (imu_length % compact_size != 0)
            return trailerError(ErrorCode::DataLoss, "Incomplete Insta360 compact IMU sample");
        const auto count = imu_length / compact_size;
        if (count > timed_metadata_maxima.bytes / sizeof(Insta360ImuSample))
            return trailerError(ErrorCode::ResourceExhausted, "Insta360 IMU sample budget exceeded");

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
