/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_controller_api.h"
namespace lfs_hip_application_controller_kernels {
__global__ void prepare(LfsHipControllerCall p) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    auto* out=reinterpret_cast<float*>(p.input);
    if(i<1600&&p.features)out[i]=reinterpret_cast<const float*>(p.features)[i];
    if(i==1600&&p.write_prior)out[i]=p.prior;
}
__global__ void backward(LfsHipControllerCall p) {
#pragma clang fp contract(off)
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    const auto* g=reinterpret_cast<const float*>(p.grad);
    const auto* a=reinterpret_cast<const float*>(p.activation);
    const auto* w=reinterpret_cast<const float*>(p.weight);
    if(i<p.m*p.n)reinterpret_cast<float*>(p.weight_gradient)[i]+=g[i/p.n]*a[i%p.n];
    if(i<p.m)reinterpret_cast<float*>(p.bias_gradient)[i]+=g[i];
    if(p.grad_input&&i<p.n) {
        float value=0;
        for(unsigned row=0;row<p.m;++row)value+=g[row]*w[row*p.n+i];
        reinterpret_cast<float*>(p.grad_input)[i]=a[i]>0?value:0;
    }
}
}
extern "C" int lfs_hip_application_controller(const LfsHipControllerCall* p,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->operation<=1&&p->queue&&p->queue<=UINTPTR_MAX,
                "Invalid HIP controller descriptor");
        if(p->operation==0)require(p->input&&p->write_prior<=1&&std::isfinite(p->prior),"Invalid HIP controller input");
        else require(p->m&&p->n&&uint64_t(p->m)*p->n<=INT_MAX&&p->grad&&p->activation&&p->weight&&p->weight_gradient&&p->bias_gradient,
                     "Invalid HIP controller backward dimensions or bindings");
        DeviceScope device(p->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue));
        if(p->operation==0)
            hipLaunchKernelGGL(lfs_hip_application_controller_kernels::prepare,dim3(7),dim3(256),0,stream,*p);
        else
            hipLaunchKernelGGL(lfs_hip_application_controller_kernels::backward,dim3((p->m*p->n+255u)/256u),dim3(256),0,stream,*p);
        hip_check(hipGetLastError(),"HIP controller submission");
    });
}
