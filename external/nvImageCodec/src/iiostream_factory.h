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

#include <functional>
#include <memory>
#include <string>

namespace nvimgcodec {

    class IoStream;
    class IIoStreamFactory {
    public:
        virtual ~IIoStreamFactory() = default;
        virtual std::unique_ptr<IoStream> createFileIoStream(
            const std::string& file_name, bool read_ahead, bool use_mmap, bool to_write) const = 0;
        virtual std::unique_ptr<IoStream> createMemIoStream(const unsigned char* data, size_t size) const = 0;
        virtual std::unique_ptr<IoStream> createMemIoStream(void* ctx, std::function<unsigned char*(void* ctx, size_t)> get_buffer_func) const = 0;
    };

} // namespace nvimgcodec
