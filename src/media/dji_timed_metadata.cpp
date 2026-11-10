// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/dji_timed_metadata.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <span>
#include <string_view>
#include <utility>

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
        enum class Message { Product,
                             Clip,
                             Header,
                             Frame,
                             FrameHeader,
                             Camera,
                             Iso,
                             Exposure,
                             Cct,
                             Quaternion,
                             Accelerometer,
                             Gimbal,
                             Gps };
        struct PacketParser {
            DjiAc204Sample sample;
            bool protocol = false, version = false, declaration = false;
            // The graph below has fixed depth, follows only qualified message
            // fields and skips unknown payloads without recursively guessing them.
            void message(Bytes bytes, Message type) {
                if (type == Message::Header)
                    declaration = true;
                std::uint64_t seen = 0;
                std::array<std::uint64_t, 2> ratio{};
                std::size_t ratio_size = 0;
                while (!bytes.empty()) {
                    const auto field = next(bytes);
                    const auto n = field.number;
                    bool known = true;
                    const auto child = [&](Message nested) {
                        field.expect(2);
                        message(field.bytes, nested);
                    };
                    // Supported singular fields are never silently merged or
                    // overwritten. Repeated exposure operands are handled below.
                    if (n < 64 && (seen & (std::uint64_t{1} << n)))
                        malformed("Duplicate supported DJI field");
                    switch (type) {
                    case Message::Product:
                        if (n == 1)
                            child(Message::Clip);
                        else if (n == 3)
                            child(Message::Frame);
                        else
                            known = false;
                        break;
                    case Message::Clip:
                        if (n == 1)
                            child(Message::Header);
                        else
                            known = false;
                        break;
                    case Message::Header:
                        if (n == 1)
                            protocol = field.text("dvtm_ac204.proto");
                        else if (n == 3)
                            version = field.text("2.0.1");
                        else
                            known = false;
                        break;
                    case Message::Frame:
                        if (n == 1)
                            child(Message::FrameHeader);
                        else if (n == 2)
                            child(Message::Camera);
                        else if (n == 4)
                            child(Message::Gimbal);
                        else
                            known = false;
                        break;
                    case Message::FrameHeader:
                        if (n == 1)
                            sample.sequence = field.uint();
                        else if (n == 2)
                            sample.camera_timestamp = field.uint();
                        else
                            known = false;
                        break;
                    case Message::Camera:
                        if (n == 3)
                            child(Message::Iso);
                        else if (n == 4)
                            child(Message::Exposure);
                        else if (n == 6)
                            child(Message::Cct);
                        else if (n == 9) {
                            sample.quaternion_wxyz.emplace();
                            child(Message::Quaternion);
                        } else if (n == 10) {
                            sample.accelerometer_xyz.emplace();
                            child(Message::Accelerometer);
                        } else
                            known = false;
                        break;
                    case Message::Iso:
                        if (n == 1)
                            sample.iso = field.real();
                        else
                            known = false;
                        break;
                    case Message::Cct:
                        if (n == 1)
                            sample.white_balance_cct = field.uint();
                        else
                            known = false;
                        break;
                    case Message::Exposure:
                        if (n == 1) {
                            const auto append = [&](std::uint64_t value) {
                                if (ratio_size == ratio.size())
                                    malformed("Too many DJI exposure operands");
                                ratio[ratio_size++] = value;
                            };
                            if (field.wire == 0)
                                append(field.integer);
                            else {
                                field.expect(2);
                                auto packed = field.bytes;
                                while (!packed.empty())
                                    append(varint(packed));
                            }
                            known = false; // repeated packed/unpacked uint64
                        } else
                            known = false;
                        break;
                    case Message::Quaternion:
                        if (n >= 1 && n <= 4)
                            (*sample.quaternion_wxyz)[n - 1] = field.real();
                        else
                            known = false;
                        break;
                    case Message::Accelerometer:
                        if (n >= 2 && n <= 4)
                            (*sample.accelerometer_xyz)[n - 2] = field.real();
                        else
                            known = false;
                        break;
                    case Message::Gimbal:
                        if (n == 2)
                            child(Message::Gps);
                        else
                            known = false;
                        break;
                    case Message::Gps:
                        if (n == 3)
                            sample.gps_status = field.uint();
                        else
                            known = false;
                        break;
                    }
                    if (known)
                        seen |= std::uint64_t{1} << n;
                }
                if (type == Message::Exposure) {
                    if (ratio_size != 2 || !ratio[1])
                        malformed("DJI exposure requires two operands and a nonzero denominator");
                    sample.exposure_time_ratio = ratio;
                }
            }
        };
    } // namespace
    Result<DjiAc204Metadata> interpretDjiAc204Metadata(const TimedMetadataObservation& observation) {
        try {
            constexpr std::uint32_t djmd = 'd' | std::uint32_t('j') << 8 | std::uint32_t('m') << 16 | std::uint32_t('d') << 24;
            if (observation.codec_tag != djmd || observation.packets.empty())
                throw Failure{ErrorCode::Unsupported, "DJI AC204 interpretation requires a nonempty djmd observation"};
            if (observation.packets.size() > timed_metadata_maxima.records)
                throw Failure{ErrorCode::ResourceExhausted, "DJI packet count exceeds collection maximum (packets=" + std::to_string(observation.packets.size()) + "/" + std::to_string(timed_metadata_maxima.records) + ")"};
            std::size_t bytes = 0;
            DjiAc204Metadata result;
            result.samples.reserve(observation.packets.size());
            for (std::size_t i = 0; i < observation.packets.size(); ++i) {
                const auto& packet = observation.packets[i];
                if (packet.bytes.empty())
                    malformed("Empty DJI packet");
                if (packet.bytes.size() > timed_metadata_maxima.bytes - bytes)
                    throw Failure{ErrorCode::ResourceExhausted, "DJI payload bytes exceed collection maximum (retained=" + std::to_string(bytes) + ", packet=" + std::to_string(packet.bytes.size()) + ", maximum=" + std::to_string(timed_metadata_maxima.bytes) + ")"};
                bytes += packet.bytes.size();
                PacketParser parser;
                parser.sample.packet_index = i;
                parser.message(packet.bytes, Message::Product);
                if ((i == 0 || parser.declaration) && (!parser.protocol || !parser.version))
                    throw Failure{ErrorCode::Unsupported, "DJI profile requires dvtm_ac204.proto version 2.0.1 in the first packet and consistent subsequent declarations"};
                result.samples.push_back(std::move(parser.sample));
            }
            return result;
        } catch (Failure& error) {
            return make_error({.code = error.code, .domain = ErrorDomain::IO, .detail = std::move(error.detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        } catch (const std::bad_alloc&) {
            // Do not allocate a diagnostic while handling an allocation failure.
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::media
