/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/mapped_file.hpp"

#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lfs::core {

#ifdef _WIN32
    namespace {
        HANDLE asHandle(void* const value) { return static_cast<HANDLE>(value); }
    } // namespace
#endif

    MappedFile::~MappedFile() { close(); }

    MappedFile::MappedFile(MappedFile&& other) noexcept {
        *this = std::move(other);
    }

    MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
#ifdef _WIN32
            file_ = std::exchange(other.file_, reinterpret_cast<void*>(-1));
            mapping_ = std::exchange(other.mapping_, nullptr);
#else
            fd_ = std::exchange(other.fd_, -1);
#endif
        }
        return *this;
    }

    bool MappedFile::open(const std::filesystem::path& path, const Advice advice) {
        close();
#ifdef _WIN32
        const DWORD hint = advice == Advice::Random
                               ? FILE_FLAG_RANDOM_ACCESS
                               : FILE_FLAG_SEQUENTIAL_SCAN;
        file_ = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, hint, nullptr);
        if (asHandle(file_) == INVALID_HANDLE_VALUE) {
            return false;
        }
        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(asHandle(file_), &file_size) || file_size.QuadPart <= 0) {
            close();
            return false;
        }
        size_ = static_cast<std::size_t>(file_size.QuadPart);
        mapping_ = CreateFileMappingW(asHandle(file_), nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            close();
            return false;
        }
        data_ = static_cast<const std::uint8_t*>(
            MapViewOfFile(asHandle(mapping_), FILE_MAP_READ, 0, 0, 0));
        if (data_ == nullptr) {
            close();
            return false;
        }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            return false;
        }
        struct stat st {};
        if (fstat(fd_, &st) != 0 || st.st_size <= 0) {
            close();
            return false;
        }
        size_ = static_cast<std::size_t>(st.st_size);
        void* mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            close();
            return false;
        }
        data_ = static_cast<const std::uint8_t*>(mapped);
        madvise(mapped, size_, advice == Advice::Random ? MADV_RANDOM : MADV_SEQUENTIAL);
#endif
        return true;
    }

    void MappedFile::close() {
#ifdef _WIN32
        if (data_ != nullptr) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_ != nullptr) {
            CloseHandle(asHandle(mapping_));
            mapping_ = nullptr;
        }
        if (asHandle(file_) != INVALID_HANDLE_VALUE) {
            CloseHandle(asHandle(file_));
            file_ = reinterpret_cast<void*>(-1);
        }
#else
        if (data_ != nullptr) {
            munmap(const_cast<std::uint8_t*>(data_), size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
    }

} // namespace lfs::core
