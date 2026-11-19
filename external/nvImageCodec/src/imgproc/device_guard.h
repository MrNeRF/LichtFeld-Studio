/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvimgcodec { namespace {

bool cuInitChecked()
{
    static CUresult res = cuInit(0);
    return res == CUDA_SUCCESS;
}

// Sentinel distinguishable from any real CUcontext value (including NULL),
// so the destructor can faithfully restore a NULL "uninitialized" context.
#define NVIMGCODEC_INVALID_CONTEXT ((CUcontext)(intptr_t)-1) // NOLINT

struct PrimaryContext
{
    ~PrimaryContext()
    {
        if (handle.load(std::memory_order::acquire) != nullptr) {
            CUresult err = cuDevicePrimaryCtxRelease(device);
            if (err != CUDA_SUCCESS) {
                std::cerr << "cuDevicePrimaryCtxRelease failed: " << err << std::endl;
            }
        }
    }

    CUdevice device{};
    std::atomic<CUcontext> handle{nullptr};

    CUcontext Get()
    {
        CUcontext ctx = handle.load();
        if (ctx)
            return ctx;
        if (cuDevicePrimaryCtxRetain(&ctx, device) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to retain CUDA primary context");
        }
        CUcontext expected = nullptr;
        if (handle.compare_exchange_strong(expected, ctx)) {
            return ctx;
        }
        // Lost the race; release our redundant retain and use the winner.
        cuDevicePrimaryCtxRelease(device);
        return expected;
    }
};

inline std::vector<PrimaryContext>& DefaultContexts()
{
    static std::vector<PrimaryContext> ctxs = []() {
        int ndevs = 0;
        if (cudaGetDeviceCount(&ndevs) != cudaSuccess) {
            throw std::runtime_error("Unable to query CUDA device count");
        }
        std::vector<PrimaryContext> v(ndevs);
        for (int i = 0; i < ndevs; i++)
            v[i].device = i;
        return v;
    }();
    return ctxs;
}

// /**
//  * Simple RAII device handling:
//  * Switch to new device on construction, back to old
//  * device on destruction
//  */
class DeviceGuard
{
  public:
    /// @brief Saves current driver context and restores it upon object destruction
    DeviceGuard()
        : old_context_(NVIMGCODEC_INVALID_CONTEXT)
    {
        if (!cuInitChecked()) {
            throw std::runtime_error(
                "Failed to load libcuda.so. "
                "Check your library paths and if the driver is installed correctly.");
        }
        if (cuCtxGetCurrent(&old_context_) != CUDA_SUCCESS) {
            throw std::runtime_error("Unable to get current CUDA context");
        }
    }

    /// @brief Saves current driver context, sets the cached primary context for
    ///        @p new_device, and restores the saved context on destruction.
    ///        For @p new_device < 0 it is a no-op.
    explicit DeviceGuard(int new_device)
        : old_context_(NVIMGCODEC_INVALID_CONTEXT)
    {
        if (new_device >= 0) {
            if (!cuInitChecked()) {
                throw std::runtime_error(
                    "Failed to load libcuda.so. "
                    "Check your library paths and if the driver is installed correctly.");
            }
            if (cuCtxGetCurrent(&old_context_) != CUDA_SUCCESS) {
                throw std::runtime_error("Unable to get current CUDA context");
            }
            auto& contexts = DefaultContexts();
            if (new_device >= static_cast<int>(contexts.size())) {
                throw std::out_of_range("Invalid device ordinal: " + std::to_string(new_device));
            }
            CUcontext ctx = contexts[new_device].Get();
            if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
                throw std::runtime_error("Unable to set current CUDA context");
            }
        }
    }

    ~DeviceGuard()
    {
        // INVALID_CONTEXT marks an inactive guard (DeviceGuard(-1) or never-set);
        // any other value, including NULL, is a real captured context to restore.
        if (old_context_ != NVIMGCODEC_INVALID_CONTEXT) {
            CUresult err = cuCtxSetCurrent(old_context_);
            if (err != CUDA_SUCCESS) {
                std::cerr << "Failed to recover from DeviceGuard: " << err << std::endl;
            }
        }
    }

  private:
    CUcontext old_context_;
};

} } // namespace nvimgcodec
