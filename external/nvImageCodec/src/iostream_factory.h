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

#include <memory>
#include <string>

#include "file_io_stream.h"
#include "iiostream_factory.h"
#include "mem_io_stream.h"

namespace nvimgcodec {

    class IoStream;
    class IoStreamFactory : public IIoStreamFactory {
    public:
        std::unique_ptr<IoStream> createFileIoStream(const std::string& file_name, bool read_ahead, bool use_mmap, bool to_write) const override {
            return FileIoStream::open(file_name, read_ahead, use_mmap, to_write);
        }

        std::unique_ptr<IoStream> createMemIoStream(const unsigned char* data, size_t size) const override {
            return std::make_unique<MemIoStream<const unsigned char>>(data, size);
        }

        std::unique_ptr<IoStream> createMemIoStream(void* ctx, std::function<unsigned char*(void* ctx, size_t)> resize_buffer_func) const override {
            return std::make_unique<MemIoStream<unsigned char>>(ctx, resize_buffer_func);
        }
    };

} // namespace nvimgcodec