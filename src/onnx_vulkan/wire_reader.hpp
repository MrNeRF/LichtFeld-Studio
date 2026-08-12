/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string>

namespace lfs::onnx_vulkan::detail {

    enum class WireType : std::uint8_t {
        Varint = 0,
        Fixed64 = 1,
        LengthDelimited = 2,
        Fixed32 = 5,
    };

    struct FieldKey {
        std::uint32_t number = 0;
        WireType type = WireType::Varint;
    };

    struct WireError {
        std::string message;
    };

    class WireReader {
    public:
        explicit WireReader(const std::span<const std::byte> bytes) noexcept
            : bytes_(bytes) {}

        [[nodiscard]] bool empty() const noexcept { return offset_ == bytes_.size(); }
        [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
        [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

        [[nodiscard]] std::expected<std::uint64_t, WireError> varint();
        [[nodiscard]] std::expected<FieldKey, WireError> key();
        [[nodiscard]] std::expected<std::span<const std::byte>, WireError> bytes();
        [[nodiscard]] std::expected<std::uint32_t, WireError> fixed32();
        [[nodiscard]] std::expected<std::uint64_t, WireError> fixed64();
        [[nodiscard]] std::expected<void, WireError> skip(WireType type);

        [[nodiscard]] static std::int64_t as_int64(const std::uint64_t value) noexcept {
            return std::bit_cast<std::int64_t>(value);
        }

    private:
        std::span<const std::byte> bytes_;
        std::size_t offset_ = 0;
    };

} // namespace lfs::onnx_vulkan::detail
