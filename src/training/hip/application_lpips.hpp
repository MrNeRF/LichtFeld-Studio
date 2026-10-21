/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_lpips_api.h"
#include "application_lpips_generated.hpp"
extern "C" int lfs_hip_application_lpips(const LfsHipLpipsCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=4 &&
                call->queue && call->queue<=UINTPTR_MAX && call->dispatch_count<=UINT32_MAX,
                "Invalid HIP LPIPS descriptor");
        if(!call->dispatch_count)return;
        DeviceScope selected(call->device);
        using namespace lfs_hip_application_lpips_kernels;
        static_assert(sizeof(Parameters_0)==sizeof(LfsHipLpipsPush) && sizeof(Parameters_0)==128);
        Parameters_0 parameters{};std::memcpy(&parameters,&call->parameters,sizeof(parameters));
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const dim3 grid(unsigned((call->dispatch_count+255)/256)),block(256);
        auto launch=[&]<unsigned Operation>() {
            hipLaunchKernelGGL((lpips_kernel<Operation>),grid,block,0,stream,parameters);
        };
        switch(call->operation) {
        case 0:launch.template operator()<0>();break;
        case 1:launch.template operator()<1>();break;
        case 2:launch.template operator()<2>();break;
        case 3:launch.template operator()<3>();break;
        case 4:launch.template operator()<4>();break;
        }
        hip_check(hipGetLastError(),"HIP LPIPS native launch");
    });
}
