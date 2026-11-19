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

#include <nvimgcodec.h>

#include "ilibrary_loader.h"
#include <string>

namespace nvimgcodec {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
    class LibraryLoader : public ILibraryLoader {
    public:
        LibraryHandle loadLibrary(const std::string& library_path) override {
            return ::dlopen(library_path.c_str(), RTLD_LAZY);
        }
        void unloadLibrary(LibraryHandle library_handle) override {
            const int result = ::dlclose(library_handle);
            if (result != 0) {
                throw std::runtime_error(std::string("Failed to unload library ") + dlerror());
            }
        }
        void* getFuncAddress(LibraryHandle library_handle, const std::string& func_name) override {
            return ::dlsym(library_handle, func_name.c_str());
        }
    };

#elif defined(_WIN32) || defined(_WIN64)

    class LibraryLoader : public ILibraryLoader {
    public:
        LibraryHandle loadLibrary(const std::string& library_path) override {
            return ::LoadLibrary(library_path.c_str());
        }
        void unloadLibrary(LibraryHandle library_handle) override {
            const BOOL result = ::FreeLibrary(library_handle);
            if (result == 0) {
                DWORD err = ::GetLastError();
                throw std::runtime_error(std::string("Failed to unload library: '#" + std::to_string(err) + "'"));
            }
        }
        void* getFuncAddress(LibraryHandle library_handle, const std::string& func_name) override {
            return reinterpret_cast<void*>(::GetProcAddress(library_handle, func_name.c_str()));
        }
    };

#endif

} // namespace nvimgcodec
