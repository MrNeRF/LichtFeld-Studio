/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_extra_api.h"
#include "application_extra_generated.hpp"
extern "C" int lfs_hip_application_extra(const LfsHipExtraCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=1 && call->count<=INT_MAX &&
                call->queue && call->queue<=UINTPTR_MAX && std::isfinite(call->weight),"Invalid HIP regularization descriptor");
        if(!call->count || call->weight==0)return;
        DeviceScope device(call->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        auto f=[&](unsigned i){return mcmc_pointer<float>(call->bindings[i]);};
        using namespace lfs_hip_application_extra_kernels;
        const unsigned grid=std::min<uint64_t>((call->count+255)/256,1024);
        if(call->operation==0) {
            hipLaunchKernelGGL(fused_scale_regularization_kernel,dim3(grid),dim3(256),0,stream,f(0),f(1),f(3),size_t(call->count),call->weight/float(call->count));
            hip_check(hipGetLastError(),"HIP scale regularization");
            hipLaunchKernelGGL(final_scale_reduce_kernel,dim3(1),dim3(256),0,stream,f(3),f(2),int(grid),call->weight,size_t(call->count));
        } else if(call->operation==1) {
            hipLaunchKernelGGL(fused_opacity_regularization_kernel,dim3(grid),dim3(256),0,stream,f(0),f(1),f(3),size_t(call->count),call->weight/float(call->count));
            hip_check(hipGetLastError(),"HIP opacity regularization");
            hipLaunchKernelGGL(final_opacity_reduce_kernel,dim3(1),dim3(256),0,stream,f(3),f(2),int(grid),call->weight,size_t(call->count));
        }
        hip_check(hipGetLastError(),"HIP regularization submission");
    });
}
