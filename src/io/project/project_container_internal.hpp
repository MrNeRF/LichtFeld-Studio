/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lfs::io::project {
    struct ChunkInfo;
}

namespace lfs::io::project::detail {

    [[nodiscard]] lfs::Result<std::vector<std::byte>>
    decompress_framed_zstd_for_testing(
        const std::filesystem::path& path, std::uint64_t offset,
        std::span<const std::byte> stored, std::uint64_t expected_size,
        std::uint64_t maximum_decoded_size);

    // Test-only override of MAX_PAYLOAD_MATERIALIZED_BYTES for payload-class
    // chunks (TENSOR_PAYLOAD / CKPT / PPIS). nullopt restores the production cap.
    void set_max_payload_materialized_bytes_for_testing(
        std::optional<std::uint64_t> maximum_decoded_size);

    // Same cap read_chunk uses for a materialized decode of `row`.
    [[nodiscard]] std::uint64_t
    max_materialized_bytes_for(const ChunkInfo& row) noexcept;

    [[nodiscard]] lfs::Error project_error(
        lfs::ErrorCode code, std::string user_message, std::string detail,
        const std::filesystem::path& path = {}, std::optional<std::uint64_t> offset = std::nullopt,
        std::string_view field = {}, std::optional<std::int64_t> native_code = std::nullopt,
        std::string_view native_name = {});

    [[nodiscard]] lfs::Result<std::uint64_t>
    checked_add(std::uint64_t lhs, std::uint64_t rhs, const std::filesystem::path& path,
                std::uint64_t field_offset, std::string_view field);

    [[nodiscard]] lfs::Result<std::uint64_t>
    checked_multiply(std::uint64_t lhs, std::uint64_t rhs, const std::filesystem::path& path,
                     std::uint64_t field_offset, std::string_view field);

    class NativeFile {
    public:
        NativeFile(const NativeFile&) = delete;
        NativeFile& operator=(const NativeFile&) = delete;
        NativeFile(NativeFile&&) = delete;
        NativeFile& operator=(NativeFile&&) = delete;
        ~NativeFile();

        [[nodiscard]] static lfs::Result<std::shared_ptr<NativeFile>>
        open_read(const std::filesystem::path& path);
        [[nodiscard]] static lfs::Result<std::shared_ptr<NativeFile>>
        open_read_write(const std::filesystem::path& path);
        [[nodiscard]] static lfs::Result<std::shared_ptr<NativeFile>>
        create_new(const std::filesystem::path& path);

        [[nodiscard]] lfs::Result<std::uint64_t> size() const;
        [[nodiscard]] lfs::Result<void>
        read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
        [[nodiscard]] lfs::Result<void>
        write_exact(std::uint64_t offset, std::span<const std::byte> source);
        // Head publication uses exactly one positional OS write. EINTR may be
        // retried; a short write is returned as failure and the slot CRC makes
        // the incomplete publication invalid.
        [[nodiscard]] lfs::Result<void>
        write_single(std::uint64_t offset, std::span<const std::byte> source);
        [[nodiscard]] lfs::Result<void> truncate(std::uint64_t size);
        [[nodiscard]] lfs::Result<void> sync_data();
        [[nodiscard]] lfs::Result<void> sync_all();

        [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

#ifdef _WIN32
        [[nodiscard]] HANDLE native_handle() const noexcept { return handle_; }
#else
        [[nodiscard]] int native_handle() const noexcept { return fd_; }
#endif

    private:
#ifdef _WIN32
        NativeFile(std::filesystem::path path, HANDLE handle);
        HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
        NativeFile(std::filesystem::path path, int fd);
        int fd_ = -1;
#endif
        std::filesystem::path path_;
    };

    class WriterLock {
    public:
        WriterLock(const WriterLock&) = delete;
        WriterLock& operator=(const WriterLock&) = delete;
        WriterLock(WriterLock&& other) noexcept;
        WriterLock& operator=(WriterLock&& other) noexcept;
        ~WriterLock();

        [[nodiscard]] static lfs::Result<WriterLock>
        acquire(const std::filesystem::path& project_path);
        // A nonzero wait retries a denied lock until the deadline so an
        // in-process writer holding the master can drain.
        [[nodiscard]] static lfs::Result<WriterLock>
        acquire(const std::filesystem::path& project_path,
                std::chrono::milliseconds wait);
        [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    private:
#ifdef _WIN32
        WriterLock(std::filesystem::path path, HANDLE handle);
        HANDLE handle_ = INVALID_HANDLE_VALUE;
        OVERLAPPED operation_{};
#else
        WriterLock(std::filesystem::path path, int fd);
        int fd_ = -1;
#endif
        std::filesystem::path path_;
    };

    struct AtomicReplaceState {
        std::optional<std::filesystem::path> backup_path;
    };

    [[nodiscard]] std::filesystem::path
    make_sibling_temp_path(const std::filesystem::path& destination, std::string_view tag);
    [[nodiscard]] lfs::Result<void>
    ensure_parent_directory(const std::filesystem::path& path);
    [[nodiscard]] lfs::Result<void>
    preflight_disk_space(const std::filesystem::path& path, std::uint64_t required_bytes);
    [[nodiscard]] lfs::Result<void>
    sync_parent_directory(const std::filesystem::path& path);
    [[nodiscard]] lfs::Result<AtomicReplaceState>
    atomic_replace(const std::filesystem::path& replacement,
                   const std::filesystem::path& destination);
    [[nodiscard]] lfs::Result<void>
    finish_atomic_replace(const AtomicReplaceState& state,
                          const std::filesystem::path& destination);
    [[nodiscard]] lfs::Result<void>
    rollback_atomic_replace(const AtomicReplaceState& state,
                            const std::filesystem::path& destination);

    [[nodiscard]] std::uint64_t unix_time_ns() noexcept;

} // namespace lfs::io::project::detail
