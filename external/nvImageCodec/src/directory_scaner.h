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

#include <filesystem>
#include <string>
#include "idirectory_scaner.h"

namespace fs = std::filesystem;

namespace nvimgcodec {

class DirectoryScaner : public IDirectoryScaner
{
  public:
    void start(const fs::path& directory) override { dir_it_ = fs::directory_iterator(directory); }
    bool hasMore() override { return dir_it_ != fs::directory_iterator(); }
    fs::path next() override
    {
        fs::path tmp = dir_it_->path();
        dir_it_++;
        return tmp;
    }
    fs::file_status symlinkStatus(const fs::path& p) override { return fs::symlink_status(p); }
    bool exists(const fs::path& p) override { return std::filesystem::exists(p); }

  private:
    fs::directory_iterator dir_it_;
};

} // namespace nvimgcodec