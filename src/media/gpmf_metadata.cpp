// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/gpmf_metadata.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <span>
#include <string_view>

namespace lfs::media {
    namespace {
        using Bytes = std::span<const std::uint8_t>;
        struct Failure {
            ErrorCode code;
            std::string detail;
        };
        [[noreturn]] void malformed(std::string_view message) { throw Failure{ErrorCode::DataLoss, std::string(message)}; }
        std::uint64_t integer(Bytes bytes) {
            std::uint64_t value = 0;
            for (auto byte : bytes)
                value = (value << 8) | byte;
            return value;
        }
        struct Record {
            std::string_view key;
            std::uint8_t type, width;
            std::uint16_t count;
            Bytes data;
        };
        std::optional<Record> next(Bytes& bytes) {
            while (bytes.size() >= 4 && integer(bytes.first(4)) == 0)
                bytes = bytes.subspan(4);
            if (bytes.empty())
                return {};
            if (bytes.size() < 8)
                malformed("Truncated GPMF record header");
            Record record{std::string_view(reinterpret_cast<const char*>(bytes.data()), 4), bytes[4], bytes[5], static_cast<std::uint16_t>(integer(bytes.subspan(6, 2))), {}};
            if (!record.width)
                malformed("GPMF record has zero structure size");
            const auto length = std::size_t(record.width) * record.count;
            const auto padded = (length + 3) & ~std::size_t(3);
            if (padded > bytes.size() - 8)
                malformed("Truncated GPMF record payload");
            record.data = bytes.subspan(8, length);
            bytes = bytes.subspan(8 + padded);
            return record;
        }
        void validateNested(const Record& record, unsigned depth) {
            if (record.type != 0)
                return;
            if (depth >= 16)
                throw Failure{ErrorCode::ResourceExhausted, "GPMF nesting exceeds the parser limit"};
            auto bytes = record.data;
            while (auto child = next(bytes))
                validateNested(*child, depth + 1);
        }
        unsigned numericWidth(std::uint8_t type) {
            switch (type) {
            case 'b':
            case 'B': return 1;
            case 's':
            case 'S': return 2;
            case 'l':
            case 'L':
            case 'f':
            case 'q': return 4;
            case 'j':
            case 'J':
            case 'd':
            case 'Q': return 8;
            default: throw Failure{ErrorCode::Unsupported, "Unsupported GPMF numeric sample type"};
            }
        }
        double number(std::uint8_t type, Bytes bytes) {
            const auto bits = integer(bytes);
            double value;
            switch (type) {
            case 'b': value = std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(bits)); break;
            case 's': value = std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(bits)); break;
            case 'l': value = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(bits)); break;
            case 'j': value = static_cast<double>(std::bit_cast<std::int64_t>(bits)); break;
            case 'f': value = std::bit_cast<float>(static_cast<std::uint32_t>(bits)); break;
            case 'd': value = std::bit_cast<double>(bits); break;
            case 'q': value = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(bits)) / 65536.0; break;
            case 'Q': value = static_cast<double>(std::bit_cast<std::int64_t>(bits)) / 4294967296.0; break;
            default: value = static_cast<double>(bits); break;
            }
            if (!std::isfinite(value))
                malformed("Nonfinite GPMF numeric value");
            return value;
        }
        std::size_t numericCount(const Record& record) {
            const auto width = numericWidth(record.type);
            if (record.width % width)
                malformed("GPMF structure does not contain complete numeric components");
            return record.data.size() / width;
        }
        std::string_view text(const std::optional<Record>& field) {
            if (!field)
                return {};
            if (field->type != 'c' && field->type != 'u' && field->type != 'U')
                malformed("GPMF text descriptor has a non-text type");
            return {reinterpret_cast<const char*>(field->data.data()), field->data.size()};
        }
        std::optional<std::uint64_t> counter(const std::optional<Record>& field) {
            if (!field)
                return {};
            if (field->type != 'B' && field->type != 'S' && field->type != 'L' && field->type != 'J')
                malformed("GPMF counter requires an unsigned integer");
            if (field->count != 1 || field->width != numericWidth(field->type))
                malformed("GPMF counter requires one complete value");
            return integer(field->data);
        }
        void retain(std::size_t bytes, std::size_t& remaining) {
            if (bytes > remaining)
                throw Failure{ErrorCode::ResourceExhausted, "GPMF decoded metadata exceeds the byte budget"};
            remaining -= bytes;
        }
        struct Context {
            std::optional<Record> device_id, device_name, name, si_units, units, scale, matrix;
            std::optional<Record> orientation_in, orientation_out, total_samples, sensor_timestamp, gps_fix, gps_precision, gps_utc;
        };
        GpmfStream samples(const Record& record, const Context& context, std::size_t& remaining) {
            const auto width = numericWidth(record.type);
            const unsigned components = record.key == "GPS5" ? 5 : 3;
            if (record.width != components * width)
                malformed("Unexpected GPMF sample dimensions");
            const auto name = text(context.name), device_name = text(context.device_name);
            const auto si_units = text(context.si_units);
            const auto units = si_units.empty() ? text(context.units) : si_units;
            const auto orientation_in = text(context.orientation_in), orientation_out = text(context.orientation_out);
            const auto gps_utc = record.key == "GPS5" ? text(context.gps_utc) : std::string_view{};
            const auto matrix_count = context.matrix ? numericCount(*context.matrix) : 0;
            if (context.matrix && matrix_count != components * components)
                malformed("GPMF calibration matrix dimensions do not match samples");
            const auto count = numericCount(record);
            retain(sizeof(GpmfStream) + device_name.size() + name.size() + units.size() + orientation_in.size() + orientation_out.size() + gps_utc.size() + (count + matrix_count) * sizeof(double), remaining);
            GpmfStream result;
            result.device_id = counter(context.device_id);
            result.device_name = device_name;
            result.key = record.key;
            result.name = name;
            result.units = units;
            result.orientation_in = orientation_in;
            result.orientation_out = orientation_out;
            result.total_samples = counter(context.total_samples);
            result.sensor_timestamp = counter(context.sensor_timestamp);
            result.components = components;
            if (record.key == "GPS5") {
                result.gps_fix = counter(context.gps_fix);
                result.gps_precision = counter(context.gps_precision);
                result.gps_utc = gps_utc;
            }
            std::array<double, 5> scales{1, 1, 1, 1, 1};
            if (context.scale) {
                const auto scale_count = numericCount(*context.scale);
                if (scale_count != 1 && scale_count != components)
                    malformed("GPMF scale dimensions do not match samples");
                const auto scale_width = numericWidth(context.scale->type);
                for (unsigned i = 0; i < components; ++i) {
                    scales[i] = number(context.scale->type, context.scale->data.subspan((scale_count == 1 ? 0 : i) * scale_width, scale_width));
                    if (scales[i] == 0)
                        malformed("GPMF scale cannot be zero");
                }
            }
            result.values.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                const auto value = number(record.type, record.data.subspan(i * width, width)) / scales[i % components];
                if (!std::isfinite(value))
                    malformed("Nonfinite GPMF scaled sample");
                result.values.push_back(value);
            }
            if (context.matrix) {
                const auto matrix_width = numericWidth(context.matrix->type);
                result.calibration_matrix.reserve(matrix_count);
                for (std::size_t i = 0; i < matrix_count; ++i)
                    result.calibration_matrix.push_back(number(context.matrix->type, context.matrix->data.subspan(i * matrix_width, matrix_width)));
            }
            return result;
        }
        void stream(const Record& parent, Context context, std::vector<GpmfStream>& output, std::size_t& remaining) {
            auto bytes = parent.data;
            while (auto field = next(bytes)) {
                if (field->key == "ACCL" || field->key == "GYRO" || field->key == "GPS5") {
                    output.push_back(samples(*field, context, remaining));
                    continue;
                }
                validateNested(*field, 2);
                if (field->key == "STNM")
                    context.name = field;
                else if (field->key == "SIUN")
                    context.si_units = field;
                else if (field->key == "UNIT")
                    context.units = field;
                else if (field->key == "SCAL")
                    context.scale = field;
                else if (field->key == "MTRX")
                    context.matrix = field;
                else if (field->key == "ORIN")
                    context.orientation_in = field;
                else if (field->key == "ORIO")
                    context.orientation_out = field;
                else if (field->key == "TSMP")
                    context.total_samples = field;
                else if (field->key == "STMP")
                    context.sensor_timestamp = field;
                else if (field->key == "GPSF")
                    context.gps_fix = field;
                else if (field->key == "GPSP")
                    context.gps_precision = field;
                else if (field->key == "GPSU")
                    context.gps_utc = field;
            }
        }
    } // namespace
    Result<std::vector<GpmfPacket>> interpretGpmfMetadata(const TimedMetadataObservation& observation, std::size_t decoded_byte_limit) {
        try {
            constexpr auto gpmd = 'g' | std::uint32_t('p') << 8 | std::uint32_t('m') << 16 | std::uint32_t('d') << 24;
            if (observation.codec_tag != gpmd || observation.packets.empty())
                throw Failure{ErrorCode::Unsupported, "GPMF interpretation requires a nonempty gpmd observation"};
            auto remaining = decoded_byte_limit;
            std::vector<GpmfPacket> result;
            result.reserve(std::min(observation.packets.size(), decoded_byte_limit / sizeof(GpmfPacket)));
            for (const auto& packet : observation.packets) {
                retain(sizeof(GpmfPacket), remaining);
                GpmfPacket decoded{packet.pts, packet.duration, {}};
                auto bytes = Bytes(packet.bytes);
                bool device = false;
                while (auto root = next(bytes)) {
                    if (root->key != "DEVC") {
                        validateNested(*root, 0);
                        continue;
                    }
                    if (root->type != 0)
                        malformed("GPMF DEVC must contain nested records");
                    device = true;
                    Context context;
                    auto children = root->data;
                    while (auto field = next(children)) {
                        if (field->key == "STRM") {
                            if (field->type != 0)
                                malformed("GPMF STRM must contain nested records");
                            stream(*field, context, decoded.streams, remaining);
                        } else {
                            validateNested(*field, 1);
                            if (field->key == "DVID")
                                context.device_id = field;
                            else if (field->key == "DVNM")
                                context.device_name = field;
                        }
                    }
                }
                if (!device)
                    malformed("GPMF payload requires a DEVC record");
                result.push_back(std::move(decoded));
            }
            return result;
        } catch (Failure& error) {
            return make_error({.code = error.code, .domain = ErrorDomain::IO, .detail = std::move(error.detail), .detection = LFS_SOURCE_SITE_CURRENT()});
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::media
