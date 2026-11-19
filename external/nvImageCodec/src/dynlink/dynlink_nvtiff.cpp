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
#include <cuda.h>
#include <stdio.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "library_loader.h"
#include <nvtiff_version.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

#define STR_IMPL_(x) #x      //stringify argument
#define STR(x) STR_IMPL_(x)  //indirection to expand argument macros

namespace {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
  static const char* __NvtiffLibNames[] = {
    "libnvtiff.so." STR(NVTIFF_VER_MAJOR),
    "libnvtiff.so"
  };
  fs::path GetDefaultNvTiffPath()
  {
      return fs::path();
  }
#elif defined(_WIN32) || defined(_WIN64)
  static const char* __NvtiffLibNames[] = {
    "nvtiff_" STR(NVTIFF_VER_MAJOR) ".dll",
    "nvtiff.dll"
  };

  fs::path GetDefaultNvTiffPath()
  {
      char dll_path[MAX_PATH];
      HMODULE hm = NULL;

      if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
              (LPCSTR)&GetDefaultNvTiffPath, &hm) != 0) {
          if (GetModuleFileName(hm, dll_path, sizeof(dll_path)) != 0) {
              fs::path path(dll_path);
              // if this comes from a shared_object like in Python site-packages,
              // go level up dir and add "nvtiff/bin" to the path
              // Examples:
              //  C:/Python39/Lib/site-packages/nvidia/nvimgcodec/extensions/nvtiff_ext_0.dll
              //                               |
              //                               V
              //  C:/Python39/Lib/site-packages/nvidia/nvtiff/bin
              path = path.parent_path();
              path = path.parent_path();
              path = path.parent_path();
              path /= "nvtiff";
              path /= "bin";
              return path;
          }
      }

      return fs::path();
  }
#endif


nvimgcodec::ILibraryLoader::LibraryHandle loadNvtiffLibrary()
{
    nvimgcodec::LibraryLoader lib_loader;
    nvimgcodec::ILibraryLoader::LibraryHandle ret = nullptr;
    fs::path default_path = GetDefaultNvTiffPath();
    for (const char* libname : __NvtiffLibNames) {
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
            "Failed to load nvtiff library! Please install nvTIFF (see "
            "https://docs.nvidia.com/cuda/nvtiff/userguide.html#installing-nvtiff).\n"
            "Note: If using nvImageCodec's Python distribution, "
            "it is enough to install the nvTIFF wheel: e.g. `python3 -m pip install nvidia-nvtiff-cu" STR(CUDA_VERSION_MAJOR) "`\n");
    }
    return ret;
}
}  // namespace

void *NvtiffLoadSymbol(const char *name) {
  nvimgcodec::LibraryLoader lib_loader;
  static nvimgcodec::ILibraryLoader::LibraryHandle nvtiffDrvLib = loadNvtiffLibrary();
  // check processing library, care later if symbol not found
  try {
    void *ret = nvtiffDrvLib ? lib_loader.getFuncAddress(nvtiffDrvLib, name) : nullptr;
    return ret;
  } catch (...) {
    return nullptr;
  }
}

bool nvtiffIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvtiffLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
