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
#include "stream_device.h"
#include <sstream>
#include <unordered_map>

namespace nvimgcodec {

CUdevice get_stream_device(cudaStream_t stream)
{
    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        std::stringstream ss;
        ss << "Unhandled CUDA error: " << cudaGetErrorName(last_error) << " " << cudaGetErrorString(last_error);
        throw std::runtime_error(ss.str());
    }

    if (stream == 0 || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) {
            throw std::runtime_error("Unable to get device id");
        }
        CUdevice device_handle;
        if (cuDeviceGet(&device_handle, dev) != CUDA_SUCCESS) {
            throw std::runtime_error(std::string("Unable to get device handle for device #") + std::to_string(dev));
        }
        return device_handle;
    }

    CUcontext context;
    CUresult result = cuStreamGetCtx(stream, &context);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("Unable to get context for stream ") + std::to_string(reinterpret_cast<uintptr_t>(stream)));
    }
    result = cuCtxPushCurrent(context);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("Unable to push context ") + std::to_string(reinterpret_cast<uintptr_t>(context)) +
                                 std::string(" for stream ") + std::to_string(reinterpret_cast<uintptr_t>(stream)));
    }
    CUdevice stream_device_handle;
    result = cuCtxGetDevice(&stream_device_handle);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("Unable to get device from context ") + std::to_string(reinterpret_cast<uintptr_t>(context)) +
                                 std::string(" for stream ") + std::to_string(reinterpret_cast<uintptr_t>(stream)));
    }
    result = cuCtxPopCurrent(&context);
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string("Unable to pop context ") + std::to_string(reinterpret_cast<uintptr_t>(context)) +
                                 std::string(" for stream ") + std::to_string(reinterpret_cast<uintptr_t>(stream)));
    }
    return stream_device_handle;
}

// Check if the device (and current driver) associated with the given stream
// can use asynchronous memory allocations and deallocations
bool can_use_async_mem_ops(cudaStream_t stream)
{
    CUdevice stream_device_handle = get_stream_device(stream);
    int attribute_res_val;
    CUresult result = cuDeviceGetAttribute(&attribute_res_val, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, stream_device_handle);
    if (result == CUDA_SUCCESS && attribute_res_val == 1) {
        return true;
    }
    return false;
}

// Retrieve the device id associated with the cudaStream_t
int get_stream_device_id(cudaStream_t stream) {
    static std::unordered_map<CUdevice, int> handle_to_device_id = []() {
        std::unordered_map<CUdevice, int> map;
        int device_count;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
            throw std::runtime_error("Unable to get device count");
        }
        for (int device_ordinal = 0; device_ordinal < device_count; ++device_ordinal) {
            CUdevice device_handle;
            if (cuDeviceGet(&device_handle, device_ordinal) != CUDA_SUCCESS) {
                throw std::runtime_error("Unable to get device handle for device #" + std::to_string(device_ordinal));
            }
            map[device_handle] = device_ordinal;
        }
        return map;
    }();
    CUdevice stream_device_handle = get_stream_device(stream);
    auto it = handle_to_device_id.find(stream_device_handle);
    if (it == handle_to_device_id.end()) {
        throw std::runtime_error("Device handle not found");
    }
    return it->second;
}

} // namespace nvimgcodec