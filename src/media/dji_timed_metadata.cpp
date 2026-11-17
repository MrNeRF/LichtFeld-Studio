// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/dji_timed_metadata.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <initializer_list>
#include <span>
#include <string_view>
#include <utility>
extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/mathematics.h>
}

namespace lfs::media {
    namespace {
        using Bytes = std::span<const std::uint8_t>;
        struct Failure {
            ErrorCode code;
            std::string detail;
        };
        [[noreturn]] void malformed(std::string_view detail) {
            throw Failure{ErrorCode::DataLoss, std::string(detail)};
        }
        std::uint64_t varint(Bytes& bytes) {
            std::uint64_t value = 0;
            for (unsigned i = 0; i < 10; ++i) {
                if (bytes.empty())
                    malformed("Truncated DJI protobuf varint");
                const auto byte = bytes.front();
                bytes = bytes.subspan(1);
                if (i == 9 && byte > 1)
                    malformed("Overflowing DJI protobuf varint");
                value |= std::uint64_t(byte & 127) << (7 * i);
                if (!(byte & 128))
                    return value;
            }
            malformed("Unterminated DJI protobuf varint");
        }
        struct Field {
            std::uint32_t number;
            unsigned wire;
            std::uint64_t integer = 0;
            Bytes bytes;
            void expect(unsigned type) const {
                if (wire != type)
                    malformed("Wrong wire type for supported DJI field");
            }
            std::uint64_t uint() const {
                expect(0);
                return integer;
            }
            float real() const {
                expect(5);
                const auto bits = std::uint32_t(bytes[0]) | std::uint32_t(bytes[1]) << 8 |
                                  std::uint32_t(bytes[2]) << 16 | std::uint32_t(bytes[3]) << 24;
                const auto value = std::bit_cast<float>(bits);
                if (!std::isfinite(value))
                    malformed("Nonfinite DJI scalar");
                return value;
            }
            bool text(std::string_view expected) const {
                expect(2);
                return bytes.size() == expected.size() &&
                       std::equal(bytes.begin(), bytes.end(), expected.begin());
            }
        };
        Field next(Bytes& bytes) {
            const auto tag = varint(bytes);
            if (!(tag >> 3) || (tag >> 3) > 0x1fffffff)
                malformed("Invalid DJI protobuf field number");
            Field field{static_cast<std::uint32_t>(tag >> 3), static_cast<unsigned>(tag & 7)};
            if (field.wire == 0) {
                field.integer = varint(bytes);
                return field;
            }
            std::uint64_t size = 0;
            switch (field.wire) {
            case 1: size = 8; break;
            case 2: size = varint(bytes); break;
            case 5: size = 4; break;
            case 3:
            case 4: throw Failure{ErrorCode::Unsupported, "DJI protobuf groups are unsupported"};
            default: malformed("Invalid DJI protobuf wire type");
            }
            if (size > bytes.size())
                malformed("Truncated DJI protobuf field");
            field.bytes = bytes.first(static_cast<std::size_t>(size));
            bytes = bytes.subspan(static_cast<std::size_t>(size));
            return field;
        }
        std::optional<Field> last(Bytes bytes, std::uint32_t number) {
            std::optional<Field> result;
            while (!bytes.empty()) {
                auto field = next(bytes);
                if (field.number == number)
                    result = field;
            }
            return result;
        }
        std::optional<Bytes> sub(Bytes bytes, std::initializer_list<std::uint32_t> path) {
            for (const auto number : path) {
                const auto field = last(bytes, number);
                if (!field)
                    return std::nullopt;
                field->expect(2);
                bytes = field->bytes;
            }
            return bytes;
        }
        std::uint64_t uint(Bytes bytes, std::uint32_t number) {
            const auto field = last(bytes, number);
            return field ? field->uint() : 0;
        }
        float real(Bytes bytes, std::uint32_t number) {
            const auto field = last(bytes, number);
            return field ? field->real() : 0.0f;
        }
        std::array<std::uint64_t, 2> exposure(Bytes bytes) {
            std::array<std::uint64_t, 2> ratio{};
            std::size_t count = 0;
            const auto append = [&](std::uint64_t value) {
                if (count == ratio.size())
                    malformed("Too many DJI exposure operands");
                ratio[count++] = value;
            };
            while (!bytes.empty()) {
                const auto field = next(bytes);
                if (field.number != 1)
                    continue;
                if (field.wire == 0)
                    append(field.integer);
                else {
                    field.expect(2);
                    auto packed = field.bytes;
                    while (!packed.empty())
                        append(varint(packed));
                }
            }
            if (count != 2 || !ratio[1])
                malformed("DJI exposure requires two operands and a nonzero denominator");
            return ratio;
        }
    } // namespace
    Result<std::vector<DjiSample>> interpretDjiMetadata(const TimedMetadataObservation& observation) {
        try {
            constexpr std::uint32_t djmd = 'd' | std::uint32_t('j') << 8 | std::uint32_t('m') << 16 | std::uint32_t('d') << 24;
            if (observation.codec_tag != djmd || observation.packets.empty())
                throw Failure{ErrorCode::Unsupported, "DJI interpretation requires a nonempty djmd observation"};
            const auto header = sub(observation.packets.front().bytes, {1, 1});
            const auto name = header ? last(*header, 1) : std::nullopt;
            const auto version = header ? last(*header, 3) : std::nullopt;
            if (!name || (!name->text("dvtm_ac203.proto") && !name->text("dvtm_ac204.proto")) || !version || !version->text("2.0.1"))
                throw Failure{ErrorCode::Unsupported, "DJI interpretation requires dvtm_ac203.proto or dvtm_ac204.proto version 2.0.1 in the first packet"};
            std::vector<DjiSample> result;
            result.reserve(observation.packets.size());
            for (const auto& packet : observation.packets) {
                if (packet.bytes.empty())
                    malformed("Empty DJI packet");
                DjiSample sample;
                if (const auto value = sub(packet.bytes, {3, 1})) {
                    sample.sequence = uint(*value, 1);
                    sample.camera_timestamp = uint(*value, 2);
                }
                if (const auto value = sub(packet.bytes, {3, 2, 3}))
                    sample.iso = real(*value, 1);
                if (const auto value = sub(packet.bytes, {3, 2, 4}))
                    sample.exposure_time_ratio = exposure(*value);
                if (const auto value = sub(packet.bytes, {3, 2, 6}))
                    sample.white_balance_cct = uint(*value, 1);
                if (const auto value = sub(packet.bytes, {3, 2, 9}))
                    sample.quaternion_wxyz = std::array{real(*value, 1), real(*value, 2), real(*value, 3), real(*value, 4)};
                if (const auto value = sub(packet.bytes, {3, 2, 10}))
                    sample.accelerometer_xyz = std::array{real(*value, 2), real(*value, 3), real(*value, 4)};
                if (const auto value = sub(packet.bytes, {3, 4, 2}))
                    sample.gps_status = uint(*value, 3);
                result.push_back(std::move(sample));
            }
            return result;
        } catch (Failure& error) {
            return make_error({.code = error.code, .domain = ErrorDomain::IO, .detail = std::move(error.detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        } catch (const std::bad_alloc&) {
            // Do not allocate a diagnostic while handling an allocation failure.
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
    struct DjiFrameStorage {
        int stream_index;
        std::vector<DjiFrameSample> samples;
    };
    Result<DjiFrameMetadata> DjiFrameMetadata::create(const TimedMetadataObservation& observation) {
        try {
            auto parsed = interpretDjiMetadata(observation);
            if (!parsed)
                return parsed.error();
            // Validate clocks before copying payloads decoded by the shared interpreter.
            for (std::size_t i = 0; i < observation.packets.size(); ++i) {
                const auto& pts = observation.packets[i].pts;
                if (!pts)
                    throw Failure{ErrorCode::InvalidArgument, "DJI frame metadata requires valid packet PTS (packet=" + std::to_string(i) + ")"};
                if (i && observation.packets[i - 1].pts->ticks >= pts->ticks)
                    throw Failure{ErrorCode::FailedPrecondition, "DJI frame metadata requires strictly increasing packet PTS (packet=" + std::to_string(i) + ")"};
            }
            auto storage = std::make_shared<DjiFrameStorage>();
            storage->stream_index = observation.stream_index;
            storage->samples.reserve(parsed->size());
            for (std::size_t i = 0; i < parsed->size(); ++i) {
                const auto& packet = observation.packets[i];
                storage->samples.push_back({*packet.pts, std::move((*parsed)[i]), packet.source_offset, packet.bytes});
            }
            return DjiFrameMetadata(std::move(storage));
        } catch (Failure& error) {
            return make_error({.code = error.code, .domain = ErrorDomain::IO, .detail = std::move(error.detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
    int DjiFrameMetadata::streamIndex() const noexcept {
        return storage_ ? storage_->stream_index : -1;
    }
    std::span<const DjiFrameSample> DjiFrameMetadata::samples() const noexcept {
        return storage_ ? std::span<const DjiFrameSample>(storage_->samples) : std::span<const DjiFrameSample>{};
    }
    Result<std::optional<std::size_t>> DjiFrameMetadata::lookup(Timestamp query) const {
        if (!storage_ || query.ticks == AV_NOPTS_VALUE || query.time_base.numerator <= 0 || query.time_base.denominator <= 0)
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "DJI frame lookup requires a live binding and a valid timestamp with positive time base", .detection = LFS_SOURCE_SITE_CURRENT()});
        const auto& entries = storage_->samples;
        const auto clock = entries.front().pts.time_base;
        const AVRational source{query.time_base.numerator, query.time_base.denominator}, target{clock.numerator, clock.denominator};
        auto ticks = query.ticks;
        if (av_cmp_q(source, target) != 0) {
            const auto down = av_rescale_q_rnd(ticks, source, target, AV_ROUND_DOWN);
            const auto up = av_rescale_q_rnd(ticks, source, target, AV_ROUND_UP);
            // FFmpeg reports overflow as INT64_MIN, also its missing-PTS sentinel.
            // Neither sentinel nor rounded fractional ticks can match a sample.
            if (down == AV_NOPTS_VALUE || up == AV_NOPTS_VALUE)
                return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "DJI frame lookup timestamp cannot be represented in the stream time base (ticks=" + std::to_string(ticks) + ", time_base=" + std::to_string(source.num) + "/" + std::to_string(source.den) + ")", .detection = LFS_SOURCE_SITE_CURRENT()});
            if (down != up)
                return std::optional<std::size_t>{};
            ticks = down;
        }
        const auto found = std::lower_bound(entries.begin(), entries.end(), ticks, [](const auto& sample, std::int64_t value) { return sample.pts.ticks < value; });
        if (found == entries.end() || found->pts.ticks != ticks)
            return std::optional<std::size_t>{};
        return std::optional<std::size_t>(static_cast<std::size_t>(found - entries.begin()));
    }
} // namespace lfs::media
