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

#pragma once

#include <cstring>
#include <functional>

#include "io_stream.h"

namespace nvimgcodec {

template <typename T>
class MemIoStream : public IoStream
{
  public:
    MemIoStream() = default;
    ~MemIoStream() = default;
    MemIoStream(T* mem, size_t bytes)
        : start_{mem}
        , size_{bytes}
    {
    }

    MemIoStream(void* ctx, std::function<unsigned char*(void* ctx, size_t)> resize_buffer_func)
        : resize_buffer_ctx_(ctx)
        , resize_buffer_func_(resize_buffer_func)
    {
    }

    size_t read(void* buf, size_t bytes) override
    {
        size_t left = size_ - pos_;
        if (left < bytes)
            bytes = left;
        std::memcpy(buf, start_ + pos_, bytes);
        pos_ += bytes;
        return bytes;
    }
    size_t write(void* buf, size_t bytes)
    {
        if constexpr (!std::is_const<T>::value) {
            size_t left = size_ - pos_;
            if (left < bytes)
                bytes = left;

            std::memcpy(static_cast<void*>(start_ + pos_), buf, bytes);
            pos_ += bytes;
            return bytes;
        } else {
            assert(!"Forbiden write for const type");
            return 0;
        }
    }

    size_t putc(unsigned char ch)
    {

        if constexpr (!std::is_const<T>::value) {
            size_t left = size_ - pos_;
            if (left < 1)
                return 0;
            std::memcpy(static_cast<void*>(start_ + pos_), &ch, 1);
            pos_++;
            return 1;
        } else {
            assert(!"Forbiden write for const type");
            return 0;
        }
    

    }

    size_t tell() const override { return pos_; }

    void seek(size_t offset, int whence = SEEK_SET) override
    {
        if (whence == SEEK_CUR) {
            offset += pos_;
        } else if (whence == SEEK_END) {
            offset += size_;
        } else {
            assert(whence == SEEK_SET);
        }
        if (offset < 0 || offset > size_t(size_))
            throw std::out_of_range("The requested position in the stream is out of range");
        pos_ = offset;
    }

    size_t size() const override { return size_; }

    void reserve(size_t bytes) override
    {
        if (resize_buffer_func_ && (bytes > size_)) {
            start_ = resize_buffer_func_(resize_buffer_ctx_, bytes);
            size_ = bytes;
        }
    }

    void flush() override
    {
        if (resize_buffer_func_&& (size_ != pos_)) {
            start_ = resize_buffer_func_(resize_buffer_ctx_, pos_);
            size_ = pos_;
        }
    }

    void* map(size_t offset, size_t size) const override {
        assert(offset + size <= size_);
        return (void*)(start_ + offset);
    }

  private:
    T* start_ = nullptr;
    size_t size_ = 0;
    size_t pos_ = 0;
    void* resize_buffer_ctx_ = nullptr;
    std::function<unsigned char*(void*, size_t)> resize_buffer_func_ = nullptr;
};

} // namespace nvimgcodec