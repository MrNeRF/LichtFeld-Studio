/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lfs::core {

    // A stable virtual reservation whose physical prefix is committed in
    // allocation-granularity increments. This is intentionally not
    // exportable: it has no shareable handles or initialization memset.
    class VmmDeviceBuffer final {
    public:
        VmmDeviceBuffer() noexcept = default;
        VmmDeviceBuffer(const VmmDeviceBuffer&) = delete;
        VmmDeviceBuffer& operator=(const VmmDeviceBuffer&) = delete;
        VmmDeviceBuffer(VmmDeviceBuffer&& other) noexcept;
        VmmDeviceBuffer& operator=(VmmDeviceBuffer&& other) noexcept;
        ~VmmDeviceBuffer();

        [[nodiscard]] static Result<VmmDeviceBuffer> create(
            std::size_t reservation_bytes, const char* label = nullptr);

        [[nodiscard]] Status commit(std::size_t required_bytes);
        // Releases whole allocation chunks at or above the aligned target.
        // If the target falls inside the kept chunk, committed_bytes() remains
        // at that chunk's end, above the requested prefix.
        [[nodiscard]] Status decommit_tail(std::size_t new_prefix_bytes);
        void release() noexcept;

        [[nodiscard]] void* data() const noexcept {
            return reinterpret_cast<void*>(base_);
        }
        [[nodiscard]] std::size_t reservation_bytes() const noexcept {
            return reservation_bytes_;
        }
        [[nodiscard]] std::size_t committed_bytes() const noexcept {
            return committed_bytes_;
        }
        [[nodiscard]] std::size_t granularity_bytes() const noexcept {
            return granularity_bytes_;
        }
        [[nodiscard]] explicit operator bool() const noexcept {
            return base_ != 0;
        }

        template <class T>
        [[nodiscard]] T* as() const noexcept {
            return reinterpret_cast<T*>(data());
        }

        static constexpr std::size_t kGranularityBytes = 2u * 1024u * 1024u;

    private:
        struct Chunk {
            std::uint64_t handle = 0;
            std::size_t offset = 0;
            std::size_t bytes = 0;
            bool mapped = true;
        };

        std::uint64_t base_ = 0;
        std::size_t reservation_bytes_ = 0;
        std::size_t committed_bytes_ = 0;
        std::size_t granularity_bytes_ = kGranularityBytes;
        int device_ = 0;
        const char* label_ = nullptr;
        std::vector<Chunk> chunks_;
    };

} // namespace lfs::core
