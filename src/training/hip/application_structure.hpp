/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_structure_api.h"
#include "application_structure_generated.hpp"
extern "C" int lfs_hip_application_structure(const LfsHipStructureCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=8 && call->queue &&
                call->queue<=UINTPTR_MAX && call->dispatch_count && call->dispatch_count<=INT_MAX,
                "Invalid HIP structure dispatch descriptor");
        const auto& p=call->parameters;
        require(p.width>0 && p.height>0 && uint64_t(p.width)*p.height<=INT_MAX &&
                p.radius<=10 && p.blocks<=1024 && p.image_byte<=1 && p.mask_byte<=1 &&
                p.first_scale<=1 && p.valid_padding<=1 && std::isfinite(p.gain) && std::isfinite(p.sigma_sq),
                "Invalid HIP structure dimensions or parameters");
        using namespace lfs_hip_application_structure_kernels;
        static_assert(sizeof(Params_0)==sizeof(LfsHipStructureParameters));
        static_assert(offsetof(Params_0,width_0)==offsetof(LfsHipStructureParameters,width));
        static_assert(offsetof(Params_0,filter_g_0)==offsetof(LfsHipStructureParameters,filter_g));
        Params_0 native{};std::memcpy(&native,&p,sizeof(native));
        DeviceScope device(call->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const auto grid=dim3((call->dispatch_count+255u)/256u);
#define LFS_STRUCTURE_CASE(N) case N: hipLaunchKernelGGL((structure_kernel<N>),grid,dim3(256),0,stream,native);break
        switch(call->operation) {
            LFS_STRUCTURE_CASE(0);LFS_STRUCTURE_CASE(1);LFS_STRUCTURE_CASE(2);
            LFS_STRUCTURE_CASE(3);LFS_STRUCTURE_CASE(4);LFS_STRUCTURE_CASE(5);
            LFS_STRUCTURE_CASE(6);LFS_STRUCTURE_CASE(7);LFS_STRUCTURE_CASE(8);
        }
#undef LFS_STRUCTURE_CASE
        hip_check(hipGetLastError(),"HIP structure submission");
    });
}
