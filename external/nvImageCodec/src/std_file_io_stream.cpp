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
#include <errno.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <nvtx3/nvtx3.hpp>

namespace nvimgcodec {

StdFileIoStream::StdFileIoStream(const std::string& path, bool to_write)
    : FileIoStream(path)
    , path_(path)
    , to_write_(to_write)
{
    fp_ = std::fopen(path_.c_str(), to_write ? "wb" : "rb");
    if (fp_ == nullptr)
        throw std::runtime_error("Could not open file " + path + ": " + std::strerror(errno));
}

void StdFileIoStream::close()
{
    if (fp_ != nullptr) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

void StdFileIoStream::seek(size_t pos, int whence)
{
    if (std::fseek(fp_, static_cast<long>(pos), whence))
        throw std::runtime_error(std::string("Seek operation failed: ") + std::strerror(errno));
}

size_t StdFileIoStream::tell() const
{
    return std::ftell(fp_);
}

size_t StdFileIoStream::read(void* buffer, size_t n_bytes)
{
    size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
    return n_read;
}

size_t StdFileIoStream::write(void* buffer, size_t n_bytes)
{
    size_t n_written = std::fwrite(buffer, 1, n_bytes, fp_);
    return n_written;
}

size_t StdFileIoStream::putc(unsigned char ch)
{
    size_t n_written = std::fputc(ch, fp_);
    return n_written;
}

std::shared_ptr<void> StdFileIoStream::get(size_t /*n_bytes*/)
{
    // this function should return a pointer inside mmaped file
    // it doesn't make sense in case of StdFileIoStream
    return {};
}

size_t StdFileIoStream::size() const
{
    struct stat sb;
    if (stat(path_.c_str(), &sb) == -1) {
        throw std::runtime_error("Unable to stat file " + path_ + ": " + std::strerror(errno));
    }
    return sb.st_size;
}

void* StdFileIoStream::map(size_t offset, size_t size) const {
    if (to_write_) {
        return nullptr;
    }
    if (buffer_data_.load() == nullptr) {
        nvtx3::scoped_range marker{"file read"};
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_data_.load() == nullptr) {
            std::ifstream file(path_, std::ios::binary);
            assert(file.is_open()); // we know it can be opened
            auto file_size = this->size();
            buffer_.resize(file_size);
            if (!file.read(reinterpret_cast<char*>(buffer_.data()), file_size)) {
                throw std::runtime_error("Error reading file: " + path_);;
            }
            buffer_data_.store(buffer_.data());
        }
    }
    assert(offset + size <= buffer_.size());
    return (void*)(buffer_data_ + offset);
}

} //namespace nvimgcodec