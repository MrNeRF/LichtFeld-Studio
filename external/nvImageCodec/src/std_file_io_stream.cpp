/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "std_file_io_stream.h"
#include <cstdio>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <string>
#include <sys/stat.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

namespace nvimgcodec {

    namespace {
#ifdef _WIN32
        // Helper to convert UTF-8 string to wide string
        std::wstring utf8_to_wstring(const std::string& utf8_str) {
            if (utf8_str.empty())
                return {};
            const int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(),
                                                        static_cast<int>(utf8_str.size()),
                                                        nullptr, 0);
            if (size_needed <= 0)
                return {};
            std::wstring wstr(size_needed, 0);
            const int converted = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(),
                                                      static_cast<int>(utf8_str.size()),
                                                      &wstr[0], size_needed);
            if (converted <= 0)
                return {};
            wstr.resize(converted);
            return wstr;
        }
#endif
    } // namespace

    StdFileIoStream::StdFileIoStream(const std::string& path, bool to_write)
        : FileIoStream(path),
          path_(path) {
#ifdef _WIN32
        const auto wpath = utf8_to_wstring(path_);
        fp_ = _wfopen(wpath.c_str(), to_write ? L"wb" : L"rb");
#else
        fp_ = std::fopen(path_.c_str(), to_write ? "wb" : "rb");
#endif
        if (fp_ == nullptr)
            throw std::runtime_error("Could not open file " + path + ": " + std::strerror(errno));
    }

    void StdFileIoStream::close() {
        if (fp_ != nullptr) {
            std::fclose(fp_);
            fp_ = nullptr;
        }
    }

    void StdFileIoStream::seek(size_t pos, int whence) {
        if (std::fseek(fp_, static_cast<long>(pos), whence))
            throw std::runtime_error(std::string("Seek operation failed: ") + std::strerror(errno));
    }

    size_t StdFileIoStream::tell() const {
        return std::ftell(fp_);
    }

    size_t StdFileIoStream::read(void* buffer, size_t n_bytes) {
        size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
        return n_read;
    }

    size_t StdFileIoStream::write(void* buffer, size_t n_bytes) {
        size_t n_written = std::fwrite(buffer, 1, n_bytes, fp_);
        return n_written;
    }

    size_t StdFileIoStream::putc(unsigned char ch) {
        size_t n_written = std::fputc(ch, fp_);
        return n_written;
    }

    std::shared_ptr<void> StdFileIoStream::get(size_t /*n_bytes*/) {
        // this function should return a pointer inside mmaped file
        // it doesn't make sense in case of StdFileIoStream
        return {};
    }

    size_t StdFileIoStream::size() const {
#ifdef _WIN32
        struct _stat64 sb;
        const auto wpath = utf8_to_wstring(path_);
        if (_wstat64(wpath.c_str(), &sb) == -1) {
            throw std::runtime_error("Unable to stat file " + path_ + ": " + std::strerror(errno));
        }
#else
        struct stat sb;
        if (stat(path_.c_str(), &sb) == -1) {
            throw std::runtime_error("Unable to stat file " + path_ + ": " + std::strerror(errno));
        }
#endif
        return sb.st_size;
    }

    void* StdFileIoStream::map(size_t offset, size_t size) const {
        if (buffer_data_.load() == nullptr) {
            nvtx3::scoped_range marker{"file read"};
            std::lock_guard<std::mutex> lock(mutex_);
            if (buffer_data_.load() == nullptr) {
                std::ifstream file;
#ifdef _WIN32
                const auto wpath = utf8_to_wstring(path_);
                file.open(wpath, std::ios::binary);
#else
                file.open(path_, std::ios::binary);
#endif
                assert(file.is_open());
                const auto file_size = this->size();
                buffer_.resize(file_size);
                if (!file.read(reinterpret_cast<char*>(buffer_.data()), file_size)) {
                    throw std::runtime_error("Error reading file: " + path_);
                }
                buffer_data_.store(buffer_.data());
            }
        }
        assert(offset + size <= buffer_.size());
        return (void*)(buffer_data_ + offset);
    }

} // namespace nvimgcodec