/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "file_io_stream.h"

namespace nvimgcodec {

    class MmapedFileIoStream : public FileIoStream {
    public:
        explicit MmapedFileIoStream(const std::string& path, bool read_ahead);
        void close() override;
        std::shared_ptr<void> get(size_t n_bytes) override;
        size_t read(void* buffer, size_t n_bytes) override;
        void seek(size_t pos, int whence = SEEK_SET) override;
        size_t tell() const override;
        size_t size() const override;
        void* map(size_t offset, size_t size) const override;

        size_t write(void* buf, size_t bytes) override { throw std::runtime_error("writing not yet supported"); }
        size_t putc(unsigned char buf) override { throw std::runtime_error("writing not yet supported"); }
        ~MmapedFileIoStream() override;

    private:
        std::shared_ptr<void> p_;
        size_t length_;
        size_t pos_;
        bool read_ahead_whole_file_;
    };
} // namespace nvimgcodec