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
#include <nvimgcodec.h>
#include <string>

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
#include <dlfcn.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif

namespace nvimgcodec {

    class ILibraryLoader {
    public:
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
        using LibraryHandle = void*;
#elif defined(_WIN32) || defined(_WIN64)
        using LibraryHandle = HMODULE;
#endif
        virtual ~ILibraryLoader() = default;

        virtual LibraryHandle loadLibrary(const std::string& library_path) = 0;
        virtual void unloadLibrary(LibraryHandle library_handle) = 0;
        virtual void* getFuncAddress(
            LibraryHandle library_handle, const std::string& func_name) = 0;
    };

} // namespace nvimgcodec