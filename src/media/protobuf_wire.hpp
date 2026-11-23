// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace lfs::media::detail::protobuf {
    using Bytes = std::span<const std::uint8_t>;
    struct Failure {
        ErrorCode code;
        std::string detail;
    };
    class Reader {
    public:
        constexpr explicit Reader(std::string_view format) : format_(format) {}

        [[noreturn]] static void fail(ErrorCode code, std::string_view before, std::string_view format, std::string_view after) {
            throw Failure{code, std::string(before) + std::string(format) + std::string(after)};
        }
        std::uint64_t varint(Bytes& bytes) const {
            std::uint64_t value = 0;
            for (unsigned i = 0; i < 10; ++i) {
                if (bytes.empty())
                    fail(ErrorCode::DataLoss, "Truncated ", format_, " protobuf varint");
                const auto byte = bytes.front();
                bytes = bytes.subspan(1);
                if (i == 9 && byte > 1)
                    fail(ErrorCode::DataLoss, "Overflowing ", format_, " protobuf varint");
                value |= std::uint64_t(byte & 127) << (7 * i);
                if (!(byte & 128))
                    return value;
            }
            fail(ErrorCode::DataLoss, "Unterminated ", format_, " protobuf varint");
        }
        struct Field {
            std::uint32_t number;
            unsigned wire;
            std::uint64_t integer = 0;
            Bytes bytes{};
            std::string_view format;
            void expect(unsigned type) const {
                if (wire != type)
                    fail(ErrorCode::DataLoss, "Wrong wire type for supported ", format, " field");
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
                    fail(ErrorCode::DataLoss, "Nonfinite ", format, " scalar");
                return value;
            }
            bool text(std::string_view expected) const {
                expect(2);
                return bytes.size() == expected.size() &&
                       std::equal(bytes.begin(), bytes.end(), expected.begin());
            }
        };
        Field next(Bytes& bytes) const {
            const auto tag = varint(bytes);
            if (!(tag >> 3) || (tag >> 3) > 0x1fffffff)
                fail(ErrorCode::DataLoss, "Invalid ", format_, " protobuf field number");
            Field field{.number = static_cast<std::uint32_t>(tag >> 3), .wire = static_cast<unsigned>(tag & 7), .format = format_};
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
            case 4: fail(ErrorCode::Unsupported, "", format_, " protobuf groups are unsupported");
            default: fail(ErrorCode::DataLoss, "Invalid ", format_, " protobuf wire type");
            }
            if (size > bytes.size())
                fail(ErrorCode::DataLoss, "Truncated ", format_, " protobuf field");
            field.bytes = bytes.first(static_cast<std::size_t>(size));
            bytes = bytes.subspan(static_cast<std::size_t>(size));
            return field;
        }
        std::optional<Field> last(Bytes bytes, std::uint32_t number) const {
            std::optional<Field> result;
            while (!bytes.empty()) {
                auto field = next(bytes);
                if (field.number == number)
                    result = field;
            }
            return result;
        }

    private:
        std::string_view format_;
    };
} // namespace lfs::media::detail::protobuf
