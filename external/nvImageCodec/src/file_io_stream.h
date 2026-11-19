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

#include "io_stream.h"

namespace nvimgcodec {

class FileIoStream : public IoStream
{
  public:
    static std::unique_ptr<FileIoStream> open(
        const std::string& uri, bool read_ahead, bool use_mmap, bool to_write);

    virtual void close()                         = 0;
    virtual std::shared_ptr<void> get(size_t n_bytes) = 0;
    virtual ~FileIoStream()                   = default;

  protected:
    explicit FileIoStream(const std::string& path)
        : path_(path)
    {
    }

    std::string path_;
};

} // namespace nvimgcodec