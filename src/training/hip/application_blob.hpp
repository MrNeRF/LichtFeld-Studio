/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_blob_api.h"
#include "application_blob_generated.hpp"
extern "C" int lfs_hip_application_blob(const LfsHipBlobCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=7 && call->queue &&
                call->queue<=UINTPTR_MAX && call->dispatch_count && call->dispatch_count<=INT_MAX,
                "Invalid HIP blob dispatch descriptor");
        const auto& p=call->parameters;
        require(call->operation==7 || (p.width>0 && p.height>0 && uint64_t(p.width)*p.height<=INT_MAX/3),
                "HIP blob image exceeds native index range");
        require(p.radius<=3 && p.byte_input<=1 && p.horizontal<=1 && p.x_max<=1 && p.y_max<=1,
                "Invalid HIP blob flags");
        if(call->operation==0)
            require(p.factor && p.factor<=46340 && p.source_width>=p.width && p.source_height>=p.height &&
                    uint64_t(p.source_width)*p.source_height<=INT_MAX/3 &&
                    uint64_t(p.width)*p.factor<=uint32_t(p.source_width) &&
                    uint64_t(p.height)*p.factor<=uint32_t(p.source_height),"Invalid HIP blob downsample extent");
        using namespace lfs_hip_application_blob_kernels;
        static_assert(sizeof(Params_0)==sizeof(LfsHipBlobParameters));
        static_assert(offsetof(Params_0,weights_0)==offsetof(LfsHipBlobParameters,weights));
        Params_0 native{};std::memcpy(&native,&p,sizeof(native));
        DeviceScope device(call->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const auto grid=dim3((call->dispatch_count+255u)/256u);
#define LFS_BLOB_CASE(N) case N: hipLaunchKernelGGL((blob_kernel<N>),grid,dim3(256),0,stream,native);break
        switch(call->operation) {
            LFS_BLOB_CASE(0);LFS_BLOB_CASE(1);LFS_BLOB_CASE(2);LFS_BLOB_CASE(3);
            LFS_BLOB_CASE(4);LFS_BLOB_CASE(5);LFS_BLOB_CASE(6);LFS_BLOB_CASE(7);
        }
#undef LFS_BLOB_CASE
        hip_check(hipGetLastError(),"HIP blob submission");
    });
}
