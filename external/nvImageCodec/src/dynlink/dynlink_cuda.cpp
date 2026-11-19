/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dynlink_cuda.h"
#include <stdio.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "library_loader.h"
namespace {

#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
static const char __CudaLibName[] = "libcuda.so";
static const char __CudaLibName1[] = "libcuda.so.1";
#elif defined(_WIN32) || defined(_WIN64)
static const char __CudaLibName[] = "libcuda.dll";
static const char __CudaLibName1[] = "nvcuda.dll";
#endif

nvimgcodec::ILibraryLoader::LibraryHandle loadCudaLibrary()
{
    nvimgcodec::LibraryLoader lib_loader;
    nvimgcodec::ILibraryLoader::LibraryHandle ret = nullptr;
    ret = lib_loader.loadLibrary(__CudaLibName1);
    if (!ret) {
        ret = lib_loader.loadLibrary(__CudaLibName);

        if (!ret) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
            fprintf(stderr, "dlopen libcuda.so failed!. Please install GPU dirver");
#elif defined(_WIN32) || defined(_WIN64)
            fprintf(stderr, "LoadLibrary nvcuda.dll failed!. Please install GPU dirver");
#endif
        }
    }
    return ret;
}

} // namespace

void* CudaLoadSymbol(const char* name)
{
    static nvimgcodec::ILibraryLoader::LibraryHandle cudaDrvLib = loadCudaLibrary();
    nvimgcodec::LibraryLoader lib_loader;
    try {
        void *ret = cudaDrvLib ? lib_loader.getFuncAddress(cudaDrvLib, name) : nullptr;
        return ret;
    } catch (...) {
        return nullptr;
    }
}

bool cuInitChecked()
{
    static CUresult res = cuInit(0);
    return res == CUDA_SUCCESS;
}

bool cuIsSymbolAvailable(const char* name)
{
    static std::mutex symbol_mutex;
    static std::unordered_map<std::string, void*> symbol_map;
    std::lock_guard<std::mutex> lock(symbol_mutex);
    auto it = symbol_map.find(name);
    if (it == symbol_map.end()) {
        auto* ptr = CudaLoadSymbol(name);
        symbol_map.insert({name, ptr});
        return ptr != nullptr;
    }
    return it->second != nullptr;
}
