/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace lfs::core {

    // Read-only memory-mapped file. Movable, not copyable; the OS page cache
    // manages residency, so mapping a file far larger than RAM is fine.
    class LFS_CORE_API MappedFile {
    public:
        enum class Advice {
            Sequential,
            Random,
        };

        MappedFile() = default;
        ~MappedFile();
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator=(const MappedFile&) = delete;
        MappedFile(MappedFile&& other) noexcept;
        MappedFile& operator=(MappedFile&& other) noexcept;

        [[nodiscard]] bool open(const std::filesystem::path& path, Advice advice = Advice::Sequential);
        void close();

        [[nodiscard]] const std::uint8_t* data() const { return data_; }
        [[nodiscard]] std::size_t size() const { return size_; }
        [[nodiscard]] bool valid() const { return data_ != nullptr; }

    private:
        const std::uint8_t* data_ = nullptr;
        std::size_t size_ = 0;
#ifdef _WIN32
        void* file_ = reinterpret_cast<void*>(-1);
        void* mapping_ = nullptr;
#else
        int fd_ = -1;
#endif
    };

} // namespace lfs::core
