/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include "library_loader.h"

namespace fs = std::filesystem;

#define STR_IMPL_(x) #x      //stringify argument
#define STR(x) STR_IMPL_(x)  //indirection to expand argument macros

namespace {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
  static const char* __NvjpegLibNames[] = {
    "libnvjpeg.so." STR(CUDA_VERSION_MAJOR),
    "libnvjpeg.so"
  };
  fs::path GetDefaultNvJpegPath()
  {
      return fs::path();
  }
#elif defined(_WIN32) || defined(_WIN64)
  static const char* __NvjpegLibNames[] = {
    "nvjpeg64_" STR(CUDA_VERSION_MAJOR) ".dll",
    "nvjpeg.dll"
  };
  fs::path GetDefaultNvJpegPath()
  {
      char dll_path[MAX_PATH];
      HMODULE hm = NULL;

      if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
              (LPCSTR)&GetDefaultNvJpegPath, &hm) != 0) {
          if (GetModuleFileName(hm, dll_path, sizeof(dll_path)) != 0) {
              fs::path path(dll_path);
              // if this comes from a shared_object like in Python site-packages,
              // go level up dir and add "nvjpeg/bin" to the path
              // Examples:
              //  C:/Python39/Lib/site-packages/nvidia/nvimgcodec/extensions/nvjpeg_ext_0.dll
              //                               |
              //                               V
              path = path.parent_path();
              path = path.parent_path();
              path = path.parent_path();
#if CUDA_VERSION_MAJOR < 13
              //  C:/Python39/Lib/site-packages/nvidia/nvjpeg/bin
              path /= "nvjpeg";
              path /= "bin";
#else
              //  C:/Python39/Lib/site-packages/nvidia/cu<CUDA_VERSION_MAJOR>/bin/x86_64
              path /= "cu" STR(CUDA_VERSION_MAJOR);
              path /= "bin";
              path /= "x86_64";
#endif
              return path;
          }
      }

      return fs::path();
  }
#endif

nvimgcodec::ILibraryLoader::LibraryHandle loadNvjpegLibrary()
{
    nvimgcodec::LibraryLoader lib_loader;
    nvimgcodec::ILibraryLoader::LibraryHandle ret = nullptr;
    auto default_path = GetDefaultNvJpegPath();
    for (const char* libname : __NvjpegLibNames) {
        if (!default_path.empty()) {
            fs::path lib_with_path = fs::path(default_path) / fs::path(libname);
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
            "Failed to load nvjpeg library! "
            "Please install CUDA toolkit or, if using nvImageCodec's Python distribution, the nvJPEG python wheel "
            "(e.g. python -m pip install nvidia-nvjpeg-cu" STR(CUDA_VERSION_MAJOR) ").\n");
    }
    return ret;
}

}  // namespace

void *NvjpegLoadSymbol(const char *name) {
  nvimgcodec::LibraryLoader lib_loader;
  static nvimgcodec::ILibraryLoader::LibraryHandle nvjpegDrvLib = loadNvjpegLibrary();
  // check processing library, care later if symbol not found
  try {
    void *ret = nvjpegDrvLib ? lib_loader.getFuncAddress(nvjpegDrvLib, name) : nullptr;
    return ret;
  } catch (...) {
    return nullptr;
  }
}

bool nvjpegIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvjpegLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
