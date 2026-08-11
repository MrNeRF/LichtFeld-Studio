/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "wire_reader.hpp"

#include <limits>

namespace lfs::onnx_vulkan::detail {

    std::expected<std::uint64_t, std::string> WireReader::varint() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 70; shift += 7) {
            if (offset_ == bytes_.size())
                return std::unexpected("truncated varint at byte " + std::to_string(offset_));
            const auto byte = std::to_integer<std::uint8_t>(bytes_[offset_++]);
            if (shift == 63 && (byte & 0xfeu) != 0)
                return std::unexpected("varint overflow at byte " + std::to_string(offset_ - 1));
            value |= static_cast<std::uint64_t>(byte & 0x7fu) << shift;
            if ((byte & 0x80u) == 0)
                return value;
        }
        return std::unexpected("varint exceeds 10 bytes");
    }

    std::expected<FieldKey, std::string> WireReader::key() {
        auto raw = varint();
        if (!raw)
            return std::unexpected(raw.error());
        const auto number = static_cast<std::uint32_t>(*raw >> 3);
        if (number == 0 || number > 0x1fffffffu)
            return std::unexpected("invalid protobuf field number " + std::to_string(number));
        const auto wire = static_cast<std::uint8_t>(*raw & 7u);
        if (wire != 0 && wire != 1 && wire != 2 && wire != 5)
            return std::unexpected("unsupported protobuf wire type " + std::to_string(wire));
        return FieldKey{number, static_cast<WireType>(wire)};
    }

    std::expected<std::span<const std::byte>, std::string> WireReader::bytes() {
        auto length = varint();
        if (!length)
            return std::unexpected(length.error());
        if (*length > remaining())
            return std::unexpected("length-delimited field exceeds enclosing message");
        const auto size = static_cast<std::size_t>(*length);
        const auto result = bytes_.subspan(offset_, size);
        offset_ += size;
        return result;
    }

    std::expected<std::uint32_t, std::string> WireReader::fixed32() {
        if (remaining() < sizeof(std::uint32_t))
            return std::unexpected("truncated fixed32 field");
        std::uint32_t value = 0;
        std::memcpy(&value, bytes_.data() + offset_, sizeof(value));
        offset_ += sizeof(value);
        if constexpr (std::endian::native == std::endian::big)
            value = std::byteswap(value);
        return value;
    }

    std::expected<std::uint64_t, std::string> WireReader::fixed64() {
        if (remaining() < sizeof(std::uint64_t))
            return std::unexpected("truncated fixed64 field");
        std::uint64_t value = 0;
        std::memcpy(&value, bytes_.data() + offset_, sizeof(value));
        offset_ += sizeof(value);
        if constexpr (std::endian::native == std::endian::big)
            value = std::byteswap(value);
        return value;
    }

    std::expected<void, std::string> WireReader::skip(const WireType type) {
        switch (type) {
        case WireType::Varint:
            if (auto value = varint(); !value)
                return std::unexpected(value.error());
            return {};
        case WireType::Fixed64:
            if (auto value = fixed64(); !value)
                return std::unexpected(value.error());
            return {};
        case WireType::LengthDelimited:
            if (auto value = bytes(); !value)
                return std::unexpected(value.error());
            return {};
        case WireType::Fixed32:
            if (auto value = fixed32(); !value)
                return std::unexpected(value.error());
            return {};
        }
        return std::unexpected("invalid protobuf wire type");
    }

} // namespace lfs::onnx_vulkan::detail
