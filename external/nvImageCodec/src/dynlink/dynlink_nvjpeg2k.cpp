/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "library_loader.h"
#include <cuda.h>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <nvjpeg2k_version.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

#define STR_IMPL_(x) #x           // stringify argument
#define STR(x)       STR_IMPL_(x) // indirection to expand argument macros

namespace {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
    static const char* __Nvjpeg2kLibNames[] = {
        "libnvjpeg2k.so." STR(NVJPEG2K_VER_MAJOR),
        "libnvjpeg2k.so"};
    fs::path GetDefaultNvJpeg2kPath() {
        return fs::path();
    }
#elif defined(_WIN32) || defined(_WIN64)
    static const char* __Nvjpeg2kLibNames[] = {
        "nvjpeg2k_" STR(NVJPEG2K_VER_MAJOR) ".dll",
        "nvjpeg2k.dll"};

    fs::path GetDefaultNvJpeg2kPath() {
        char dll_path[MAX_PATH];
        HMODULE hm = NULL;

        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&GetDefaultNvJpeg2kPath, &hm) != 0) {
            if (GetModuleFileName(hm, dll_path, sizeof(dll_path)) != 0) {
                fs::path path(dll_path);
                // if this comes from a shared_object like in Python site-packages,
                // go level up dir and add "nvjpeg2k/bin" to the path
                // Examples:
                //  C:/Python39/Lib/site-packages/nvidia/nvimgcodec/extensions/nvjpeg2k_ext_0.dll
                //                               |
                //                               V
                //  C:/Python39/Lib/site-packages/nvidia/nvjpeg2k/bin
                path = path.parent_path();
                path = path.parent_path();
                path = path.parent_path();
                path /= "nvjpeg2k";
                path /= "bin";
                return path;
            }
        }

        return fs::path();
    }
#endif

    nvimgcodec::ILibraryLoader::LibraryHandle loadNvjpeg2kLibrary() {
        nvimgcodec::LibraryLoader lib_loader;
        nvimgcodec::ILibraryLoader::LibraryHandle ret = nullptr;
        fs::path default_path = GetDefaultNvJpeg2kPath();
        for (const char* libname : __Nvjpeg2kLibNames) {
            if (!default_path.empty()) {
                fs::path lib_with_path = default_path / fs::path(libname);
                ret = lib_loader.loadLibrary(lib_with_path.string());
                if (ret != nullptr)
                    break;
            }
            ret = lib_loader.loadLibrary(libname);
            if (ret != nullptr)
                break;
        }
        if (!ret) {
            fprintf(stderr,
                    "Failed to load nvjpeg2k library! Please install nvJPEG2000 (see "
                    "https://docs.nvidia.com/cuda/nvjpeg2000/userguide.html#installing-nvjpeg2000).\n"
                    "Note: If using nvImageCodec's Python distribution, "
                    "it is enough to install the nvJPEG2000 wheel: e.g. `python3 -m pip install nvidia-nvjpeg2k-cu" STR(CUDA_VERSION_MAJOR) "`\n");
        }
        return ret;
    }
} // namespace

void* Nvjpeg2kLoadSymbol(const char* name) {
    nvimgcodec::LibraryLoader lib_loader;
    static nvimgcodec::ILibraryLoader::LibraryHandle nvjpeg2kDrvLib = loadNvjpeg2kLibrary();
    // check processing library, care later if symbol not found
    try {
        void* ret = nvjpeg2kDrvLib ? lib_loader.getFuncAddress(nvjpeg2kDrvLib, name) : nullptr;
        return ret;
    } catch (...) {
        return nullptr;
    }
}

bool nvjpeg2kIsSymbolAvailable(const char* name) {
    static std::mutex symbol_mutex;
    static std::unordered_map<std::string, void*> symbol_map;
    std::lock_guard<std::mutex> lock(symbol_mutex);
    auto it = symbol_map.find(name);
    if (it == symbol_map.end()) {
        auto* ptr = Nvjpeg2kLoadSymbol(name);
        symbol_map.insert({name, ptr});
        return ptr != nullptr;
    }
    return it->second != nullptr;
}
