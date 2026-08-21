/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <streambuf>

namespace lfs::io::project {

    // Get-area windows stay at most INT_MAX bytes because MSVC stores the
    // get-area length as int.
    class SpanStreambuf final : public std::streambuf {
    public:
        explicit SpanStreambuf(
            std::span<const std::byte> bytes,
            std::size_t window_bytes = static_cast<std::size_t>(
                std::numeric_limits<int>::max()))
            : bytes_(bytes),
              window_bytes_(std::clamp(
                  window_bytes, std::size_t{1},
                  static_cast<std::size_t>(
                      std::numeric_limits<int>::max()))) {
            assert(window_bytes >= 1 &&
                   window_bytes <=
                       static_cast<std::size_t>(
                           std::numeric_limits<int>::max()));
            set_window(0);
        }

    protected:
        int_type underflow() override {
            if (gptr() < egptr()) {
                return traits_type::to_int_type(*gptr());
            }
            const auto pos = logical_position();
            if (pos >= size()) {
                return traits_type::eof();
            }
            set_window(pos);
            if (gptr() < egptr()) {
                return traits_type::to_int_type(*gptr());
            }
            return traits_type::eof();
        }

        std::streamsize xsgetn(char* s,
                               std::streamsize count) override {
            if (s == nullptr || count <= 0) {
                return 0;
            }
            const auto pos = logical_position();
            const auto total = size();
            if (pos >= total) {
                return 0;
            }
            const auto want = static_cast<std::uint64_t>(count);
            const auto take = std::min(want, total - pos);
            std::memcpy(s, bytes_.data() + pos,
                        static_cast<std::size_t>(take));
            set_window(pos + take);
            return static_cast<std::streamsize>(take);
        }

        pos_type seekoff(const off_type offset,
                         const std::ios_base::seekdir direction,
                         const std::ios_base::openmode mode) override {
            if ((mode & std::ios_base::in) == 0) {
                return pos_type(off_type(-1));
            }
            const auto total = size();
            std::uint64_t base = 0;
            if (direction == std::ios_base::beg) {
                base = 0;
            } else if (direction == std::ios_base::cur) {
                base = logical_position();
            } else if (direction == std::ios_base::end) {
                base = total;
            } else {
                return pos_type(off_type(-1));
            }

            std::uint64_t target = 0;
            if (offset >= 0) {
                const auto add = static_cast<std::uint64_t>(offset);
                if (add > total - base) {
                    return pos_type(off_type(-1));
                }
                target = base + add;
            } else {
                std::uint64_t mag = 0;
                if (offset ==
                    std::numeric_limits<off_type>::min()) {
                    mag = static_cast<std::uint64_t>(
                              std::numeric_limits<off_type>::max()) +
                          1;
                } else {
                    mag = static_cast<std::uint64_t>(-offset);
                }
                if (mag > base) {
                    return pos_type(off_type(-1));
                }
                target = base - mag;
            }

            const auto window_len =
                (eback() != nullptr && egptr() != nullptr)
                    ? static_cast<std::uint64_t>(egptr() - eback())
                    : 0;
            if (eback() != nullptr && target >= window_start_ &&
                target <= window_start_ + window_len) {
                setg(eback(),
                     eback() + static_cast<std::ptrdiff_t>(
                                   target - window_start_),
                     egptr());
            } else {
                set_window(target);
            }
            return pos_type(static_cast<off_type>(target));
        }

        pos_type seekpos(const pos_type position,
                         const std::ios_base::openmode mode) override {
            return seekoff(static_cast<off_type>(position),
                           std::ios_base::beg, mode);
        }

    private:
        [[nodiscard]] std::uint64_t size() const noexcept {
            return static_cast<std::uint64_t>(bytes_.size());
        }

        [[nodiscard]] char* data() const noexcept {
            return const_cast<char*>(
                reinterpret_cast<const char*>(bytes_.data()));
        }

        [[nodiscard]] std::uint64_t
        logical_position() const noexcept {
            if (eback() == nullptr || gptr() == nullptr) {
                return window_start_;
            }
            return window_start_ +
                   static_cast<std::uint64_t>(gptr() - eback());
        }

        void set_window(const std::uint64_t pos) {
            const auto total = size();
            window_start_ = pos > total ? total : pos;
            char* const first = data() + window_start_;
            if (window_start_ >= total) {
                setg(first, first, first);
                return;
            }
            const auto remaining = total - window_start_;
            const auto count = std::min(
                remaining,
                static_cast<std::uint64_t>(window_bytes_));
            setg(first, first,
                 first + static_cast<std::ptrdiff_t>(count));
        }

        std::span<const std::byte> bytes_{};
        std::uint64_t window_start_ = 0;
        std::size_t window_bytes_ = 1;
    };

} // namespace lfs::io::project
