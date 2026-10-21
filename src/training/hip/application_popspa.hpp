/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_popspa_api.h"
#include "application_popspa_generated.hpp"

namespace lfs_hip_popspa_detail {
    template <class T> T* pointer(uint64_t address) {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(address));
    }

    struct SelectionLayout {
        size_t keys[2]{}, indices[2]{}, priorities[2]{}, scratch = 0, scratch_bytes = 0, bytes = 0;

        SelectionLayout(size_t n, size_t k, hipStream_t stream) {
            if (!k || k == n)
                return;
            const auto reserve = [&](size_t count, size_t width) {
                require(bytes <= SIZE_MAX - 255, "HIP POPSpa workspace alignment overflow");
                bytes = (bytes + 255) & ~size_t(255);
                const auto offset = bytes, extent = multiply(count, width);
                require(extent <= SIZE_MAX - bytes, "HIP POPSpa workspace size overflow");
                bytes += extent;
                return offset;
            };
            for (auto& offset : keys) offset = reserve(n, sizeof(double));
            for (auto& offset : indices) offset = reserve(n, sizeof(int64_t));
            for (auto& offset : priorities) offset = reserve(n, sizeof(uint8_t));
            size_t scores_bytes = 0, priority_bytes = 0, index_bytes = 0;
            hip_check(hipcub::DeviceRadixSort::SortPairsDescending(
                          nullptr, scores_bytes, static_cast<double*>(nullptr), static_cast<double*>(nullptr),
                          static_cast<int64_t*>(nullptr), static_cast<int64_t*>(nullptr), int(n), 0, 64, stream),
                      "POPSpa score sort workspace query");
            hip_check(hipcub::DeviceRadixSort::SortPairsDescending(
                          nullptr, priority_bytes, static_cast<uint8_t*>(nullptr), static_cast<uint8_t*>(nullptr),
                          static_cast<int64_t*>(nullptr), static_cast<int64_t*>(nullptr), int(n), 0, 2, stream),
                      "POPSpa mask sort workspace query");
            hip_check(hipcub::DeviceRadixSort::SortKeys(
                          nullptr, index_bytes, static_cast<int64_t*>(nullptr), static_cast<int64_t*>(nullptr),
                          int(k), 0, 64, stream),
                      "POPSpa selected index sort workspace query");
            scratch_bytes = std::max({scores_bytes, priority_bytes, index_bytes});
            scratch = reserve(scratch_bytes, 1);
        }

        template <class T> T* at(const LfsHipPopspaCall& call, size_t offset) const {
            return reinterpret_cast<T*>(pointer<uint8_t>(call.workspace) + offset);
        }
    };

    __global__ void sequence(int64_t* indices, size_t n) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) indices[i] = int64_t(i);
    }
    __global__ void initialize_keys(const double* scores, double* keys, int64_t* indices, size_t n) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) {
            // Numeric ties, including signed zero, retain ascending row ids
            // through both stable radix sorts.
            const double value = scores[i];
            keys[i] = value == 0.0 ? 0.0 : value;
            indices[i] = int64_t(i);
        }
    }
    __global__ void mask_priorities(const int64_t* indices, const bool* active, const bool* frozen,
                                    uint8_t* priorities, size_t n) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) {
            const auto row = indices[i];
            priorities[i] = uint8_t((active[row] ? 2 : 0) + (frozen[row] ? 1 : 0));
        }
    }
    __global__ void full_projection(const float* raw, float* z, float* u, const bool* active,
                                    const bool* frozen, size_t n, bool update_dual) {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) {
            const float a = 1.f / (1.f + expf(-raw[i]));
            const float projected = a + u[i];
            z[i] = projected;
            if (update_dual && active[i] && !frozen[i])
                u[i] += a - projected;
        }
    }

    void dimensions(const LfsHipPopspaCall* call) {
        require(call && call->struct_bytes == sizeof(*call) && call->operation <= LFS_HIP_POPSPA_EFFECTIVE_RANK &&
                    call->queue && call->queue <= UINTPTR_MAX && call->count <= INT_MAX &&
                    call->keep_count <= call->count && call->active_count <= call->count && call->update_dual <= 1,
                "Invalid HIP POPSpa descriptor or dimensions");
    }

    int64_t* select(const LfsHipPopspaCall& call, const SelectionLayout& layout, hipStream_t stream) {
        const auto n = size_t(call.count), k = size_t(call.keep_count);
        auto* keys0 = layout.at<double>(call, layout.keys[0]);
        auto* keys1 = layout.at<double>(call, layout.keys[1]);
        auto* ids0 = layout.at<int64_t>(call, layout.indices[0]);
        auto* ids1 = layout.at<int64_t>(call, layout.indices[1]);
        auto* priorities0 = layout.at<uint8_t>(call, layout.priorities[0]);
        auto* priorities1 = layout.at<uint8_t>(call, layout.priorities[1]);
        const bool project = call.operation == LFS_HIP_POPSPA_PROJECT;
        if (project)
            hipLaunchKernelGGL(lfs_hip_popspa_kernels::shifted_kernel, dim3((n + 255) / 256), dim3(256),
                               0, stream, pointer<float>(call.raw), pointer<float>(call.u), keys0, n);
        hipLaunchKernelGGL(initialize_keys, dim3((n + 255) / 256), dim3(256), 0, stream,
                           project ? keys0 : pointer<double>(call.scores), keys0, ids0, n);
        auto* scratch = layout.at<uint8_t>(call, layout.scratch);
        size_t available = layout.scratch_bytes;
        hip_check(hipcub::DeviceRadixSort::SortPairsDescending(scratch, available, keys0, keys1, ids0, ids1,
                                                              int(n), 0, 64, stream), "POPSpa score sort");
        hipLaunchKernelGGL(mask_priorities, dim3((n + 255) / 256), dim3(256), 0, stream,
                           ids1, pointer<bool>(call.active), pointer<bool>(call.frozen), priorities0, n);
        available = layout.scratch_bytes;
        hip_check(hipcub::DeviceRadixSort::SortPairsDescending(scratch, available, priorities0, priorities1,
                                                              ids1, ids0, int(n), 0, 2, stream), "POPSpa mask sort");
        auto* selected = project ? ids1 : pointer<int64_t>(call.keep);
        available = layout.scratch_bytes;
        hip_check(hipcub::DeviceRadixSort::SortKeys(scratch, available, ids0, selected, int(k), 0, 64, stream),
                  "POPSpa ascending selected indices");
        return selected;
    }
}

extern "C" int lfs_hip_application_popspa_workspace(const LfsHipPopspaCall* call, uint64_t* bytes, LfsHipError* error) {
    if (bytes) *bytes = 0;
    return boundary(nullptr, error, [&] {
        using namespace lfs_hip_popspa_detail;
        dimensions(call);
        require(bytes, "Missing HIP POPSpa workspace result");
        DeviceScope device(call->device);
        if (call->operation == LFS_HIP_POPSPA_KEEP || call->operation == LFS_HIP_POPSPA_PROJECT)
            *bytes = SelectionLayout(call->count, call->keep_count, reinterpret_cast<hipStream_t>(call->queue)).bytes;
    });
}

extern "C" int lfs_hip_application_popspa(const LfsHipPopspaCall* call, LfsHipError* error) {
    return boundary(nullptr, error, [&] {
        using namespace lfs_hip_popspa_detail;
        using namespace lfs_hip_popspa_kernels;
        dimensions(call);
        DeviceScope device(call->device);
        const auto stream = reinterpret_cast<hipStream_t>(call->queue);
        const size_t n = call->count, k = call->keep_count;
        if (!n) return;
        const dim3 grid((n + 255) / 256), block(256);
        if (call->operation == LFS_HIP_POPSPA_KEEP || call->operation == LFS_HIP_POPSPA_PROJECT) {
            const bool project = call->operation == LFS_HIP_POPSPA_PROJECT;
            require(project ? (call->raw && call->z && call->u && call->active && call->frozen)
                            : (call->scores && call->active && call->frozen && (!k || call->keep)),
                    "Missing HIP POPSpa selection binding");
            if (k == n) {
                if (project)
                    hipLaunchKernelGGL(full_projection, grid, block, 0, stream, pointer<float>(call->raw),
                                       pointer<float>(call->z), pointer<float>(call->u), pointer<bool>(call->active),
                                       pointer<bool>(call->frozen), n, bool(call->update_dual));
                else
                    hipLaunchKernelGGL(sequence, grid, block, 0, stream, pointer<int64_t>(call->keep), n);
            } else {
                const SelectionLayout layout(n, k, stream);
                require(!layout.bytes || (call->workspace && call->workspace % 256 == 0 && call->workspace_bytes >= layout.bytes),
                        "HIP POPSpa workspace is absent, misaligned or too small");
                int64_t* selected = k ? select(*call, layout, stream) : nullptr;
                if (project) {
                    hip_check(hipMemsetAsync(pointer<float>(call->z), 0, multiply(n, sizeof(float)), stream), "POPSpa proxy clear");
                    if (k)
                        hipLaunchKernelGGL(scatter_proxy, dim3((k + 255) / 256), block, 0, stream,
                                           layout.at<double>(*call, layout.keys[0]), selected, pointer<float>(call->z), k);
                    if (call->update_dual)
                        hipLaunchKernelGGL(dual_kernel, grid, block, 0, stream, pointer<float>(call->raw), pointer<float>(call->z),
                                           pointer<float>(call->u), pointer<bool>(call->active), pointer<bool>(call->frozen), n);
                }
            }
        } else if (call->operation == LFS_HIP_POPSPA_PENALTY) {
            require(call->raw && call->z && call->u && call->active && call->frozen && call->gradient && call->loss &&
                        std::isfinite(call->rho) && call->rho >= 0.f, "Invalid HIP POPSpa penalty binding or weight");
            hipLaunchKernelGGL(penalty_kernel, grid, block, 0, stream, pointer<float>(call->raw), pointer<float>(call->z),
                               pointer<float>(call->u), pointer<bool>(call->active), pointer<bool>(call->frozen),
                               pointer<float>(call->gradient), pointer<float>(call->loss), n, call->rho);
        } else {
            require(call->raw && call->loss && std::isfinite(call->rank_weight) && call->rank_weight >= 0.f &&
                        std::isfinite(call->thin_weight) && call->thin_weight >= 0.f &&
                        std::isfinite(call->epsilon) && call->epsilon > 0.f, "Invalid HIP effective-rank binding or weights");
            if (call->active_count)
                hipLaunchKernelGGL(effective_rank_kernel, grid, block, 0, stream, pointer<float>(call->raw),
                                   pointer<float>(call->gradient), pointer<bool>(call->active), pointer<bool>(call->frozen),
                                   pointer<float>(call->loss), n, 1.f / float(call->active_count),
                                   call->rank_weight, call->thin_weight, call->epsilon);
        }
        hip_check(hipGetLastError(), "HIP POPSpa submission");
    });
}
