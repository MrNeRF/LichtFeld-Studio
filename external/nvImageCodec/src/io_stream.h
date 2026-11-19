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

#include <cassert>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace nvimgcodec {

/**
 * @brief An abstract file-like interface for reading and writting data.
 */
class IoStream
{
  public:
    virtual ~IoStream() = default;

    /**
     * @brief Reads all requested data from the stream; if not all of the data can be read,
     *        an exception is thrown.
     *
     * @param buf   the output buffer
     * @param bytes the number of bytes to read
     */
    inline void readBytes(void* buf, size_t bytes)
    {
        char* b = static_cast<char*>(buf);
        while (bytes) {
            size_t n = read(b, bytes);
            if (n == 0)
                throw std::runtime_error("End of stream");
            if (n < 0)
                throw std::runtime_error("An error occurred while reading data.");
            b += n;
            assert(static_cast<size_t>(n) <= bytes);
            bytes -= n;
        }
    }

    /**
     * @brief Reads one object of given type from the stream
     *
     * @tparam T  the type of the object to read; should be trivially copyable or otherwise
     *            safe to be overwritten with memcpy or similar.
     */
    template <typename T>
    inline T readOne()
    {
        T t;
        readAll(&t, 1);
        return t;
    }

    /**
     * @brief Reads `count` instances of type T from the stream to the provided buffer
     *
     * If the function cannot read the requested number of objects, an exception is thrown
     *
     * @tparam T    the type of the object to read; should be trivially copyable or otherwise
     *              safe to be overwritten with memcpy or similar.
     * @param buf   the output buffer
     * @param count the number of objects to read
     */
    template <typename T>
    inline void readAll(T* buf, size_t count)
    {
        readBytes(buf, sizeof(T) * count);
    }

    /**
     * @brief Skips `count` objects in the stream
     *
     * Skips over the length of `count` objects of given type (by default char,
     * because sizeof(char) == 1).
     *
     * NOTE: Negative skips are allowed.
     *
     * @tparam T type of the object to skip; defaults to `char`
     * @param count the number of objects to skip
     */
    template <typename T = char>
    void skip(size_t count = 1)
    {
        seek(count * sizeof(T), SEEK_CUR);
    }

    /**
     * @brief Reads data from the stream and advances the read pointer; partial reads are allowed.
     *
     * A valid implementation of this function reads up to `bytes` bytes from the stream and
     * stores them in `buf`. If the function cannot read all of the requested bytes due to
     * end-of-file, it shall read all it can and return the number of bytes actually read.
     * When reading from a regular file and the file pointer is already at the end, the function
     * shall return 0.
     *
     * This function does not throw EndOfStream.
     *
     * @param buf       the output buffer
     * @param bytes     the number of bytes to read
     * @return size _t  the number of bytes actually read or
     *                  0 in case of end-of-stream condition
     */
    virtual size_t read(void* buf, size_t bytes) = 0;
    virtual size_t write(void* buf, size_t bytes) = 0;
    virtual size_t putc(unsigned char buf) = 0;

    /**
     * @brief Moves the read pointer in the stream.
     *
     * If the new pointer would be out of range, the function may either clamp it to a valid range
     * or throw an error.
     *
     * @param pos     the offset to move
     * @param whence  the beginning - SEEK_SET, SEEK_CUR or SEEK_END
     */
    virtual void seek(size_t pos, int whence = SEEK_SET) = 0;

    /**
     * @brief Returns the current read position, in bytes from the beginning, in the stream.
     */
    virtual size_t tell() const = 0;

    /**
     * @brief Returns the length, in bytes, of the stream.
     */
    virtual size_t size() const = 0;

    /**
     * @brief Returns the size as a signed integer.
     */
    inline size_t ssize() const { return size(); }

    /**
     * @brief Provides expected bytes of data which are going to be written.
     *        Gives possibility to pre/re-allocate buffer for map function
     */
    virtual void reserve(size_t bytes){};

    /**
     * @brief Requests all data to be written to the output.
     */
    virtual void flush(){};

    /**
     * @brief  Maps stream data into memory
     * 
     * @param offset Offset in the stream to begin mapping.
     * @param size Length of the mapping
     *  
     * @return const void* pointer to mapped data or nullptr, if it cannot be mapped
     */
    virtual void* map(size_t offset, size_t size) const { return nullptr; };

    /**
     * @brief Unmaps previously mapped data
     * 
     * @param addr Pointer to mapped data
     * @param size Length of data to unmap 
     */
    virtual void unmap(void* ptr, size_t size){};
};

} // namespace nvimgcodec