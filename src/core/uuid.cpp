/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/uuid.hpp"

#include <algorithm>
#include <cerrno>
#include <mutex>
#include <random>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
#include <bcrypt.h>
// clang-format on
#elif defined(__linux__)
#include <sys/random.h>
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <unistd.h>
#endif

namespace lfs::core {

    namespace {
        constexpr char HEX_DIGITS[] = "0123456789abcdef";

        [[nodiscard]] constexpr int decodeHex(const char value) noexcept {
            if (value >= '0' && value <= '9')
                return value - '0';
            if (value >= 'a' && value <= 'f')
                return value - 'a' + 10;
            if (value >= 'A' && value <= 'F')
                return value - 'A' + 10;
            return -1;
        }

        void fillRandomBytes(std::array<std::uint8_t, 16>& bytes) {
#if defined(_WIN32)
            const NTSTATUS status = BCryptGenRandom(
                nullptr,
                reinterpret_cast<PUCHAR>(bytes.data()),
                static_cast<ULONG>(bytes.size()),
                BCRYPT_USE_SYSTEM_PREFERRED_RNG);
            if (!BCRYPT_SUCCESS(status)) {
                throw UuidGenerationError("BCryptGenRandom failed while generating UUIDv4");
            }
#elif defined(__linux__)
            std::size_t offset = 0;
            while (offset < bytes.size()) {
                const ssize_t count = ::getrandom(bytes.data() + offset, bytes.size() - offset, 0);
                if (count > 0) {
                    offset += static_cast<std::size_t>(count);
                    continue;
                }
                if (count < 0 && errno == EINTR) {
                    continue;
                }
                const int error = count < 0 ? errno : EIO;
                throw UuidGenerationError(
                    "getrandom failed while generating UUIDv4: " +
                    std::generic_category().message(error));
            }
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
            if (::getentropy(bytes.data(), bytes.size()) != 0) {
                throw UuidGenerationError(
                    "getentropy failed while generating UUIDv4: " +
                    std::generic_category().message(errno));
            }
#else
            // The fallback is permitted only when the implementation documents a
            // non-deterministic random_device, represented by positive entropy().
            static std::mutex mutex;
            static std::random_device random_device;
            std::lock_guard lock(mutex);
            if (random_device.entropy() <= 0.0) {
                throw UuidGenerationError(
                    "No documented non-deterministic random_device is available for UUIDv4");
            }
            std::uniform_int_distribution<unsigned int> distribution(0, 255);
            for (std::uint8_t& byte : bytes) {
                byte = static_cast<std::uint8_t>(distribution(random_device));
            }
#endif
        }
    } // namespace

    bool Uuid::is_nil() const noexcept {
        return std::ranges::all_of(bytes, [](const std::uint8_t byte) { return byte == 0; });
    }

    std::string Uuid::to_string() const {
        std::string result(36, '\0');
        std::size_t output = 0;
        for (std::size_t input = 0; input < bytes.size(); ++input) {
            if (input == 4 || input == 6 || input == 8 || input == 10) {
                result[output++] = '-';
            }
            result[output++] = HEX_DIGITS[bytes[input] >> 4];
            result[output++] = HEX_DIGITS[bytes[input] & 0x0f];
        }
        return result;
    }

    std::optional<Uuid> Uuid::from_string(const std::string_view value) {
        if (value.size() != 36 || value[8] != '-' || value[13] != '-' ||
            value[18] != '-' || value[23] != '-') {
            return std::nullopt;
        }

        Uuid result;
        std::size_t input = 0;
        for (std::size_t output = 0; output < result.bytes.size(); ++output) {
            if (input == 8 || input == 13 || input == 18 || input == 23) {
                ++input;
            }
            const int high = decodeHex(value[input++]);
            const int low = decodeHex(value[input++]);
            if (high < 0 || low < 0) {
                return std::nullopt;
            }
            result.bytes[output] = static_cast<std::uint8_t>((high << 4) | low);
        }
        return result;
    }

    Uuid generate_uuid_v4() {
        Uuid result;
        fillRandomBytes(result.bytes);
        result.bytes[6] = static_cast<std::uint8_t>((result.bytes[6] & 0x0f) | 0x40);
        result.bytes[8] = static_cast<std::uint8_t>((result.bytes[8] & 0x3f) | 0x80);
        return result;
    }

} // namespace lfs::core
