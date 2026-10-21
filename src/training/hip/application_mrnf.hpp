/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_mrnf_api.h"
#include "application_mrnf_generated.hpp"
namespace lfs_hip_mrnf_kernels {
__global__ void identity_kernel(int64_t* indices,unsigned n) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)indices[i]=i;
}
__global__ void prune_kernel(const float* means,const float* scale,unsigned char* mask,unsigned n,
                             float x,float y,float z,float maximum,float logmax) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const float a=means[3*i],b=means[3*i+1],c=means[3*i+2];
    const float distance=fmaxf(fabsf(a-x),fmaxf(fabsf(b-y),fabsf(c-z)));
    if(scale[i]>logmax||(isfinite(a)&&isfinite(b)&&isfinite(c)&&distance>maximum))mask[i]=1;
}
}
extern "C" int lfs_hip_application_mrnf(const LfsHipMrnfCall* p,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->operation<=7&&p->queue&&p->queue<=UINTPTR_MAX&&p->count&&p->count<=INT_MAX/3,"Invalid HIP MRNF descriptor");
        require(p->bindings[0]&&(p->operation==7||p->bindings[1]),"Missing HIP MRNF binding");
        if(p->operation==0||p->operation==5||p->operation==6)require(p->bindings[2],"Missing HIP MRNF third binding");
        if(p->operation==3)require(p->bindings[5],"Missing HIP MRNF parent output");
        for(float v:p->value)require(std::isfinite(v),"Invalid HIP MRNF scalar");
        DeviceScope selected(p->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue));
        const dim3 grid((p->count+255u)/256u),block(256);
        using namespace lfs_hip_mrnf_kernels;
        auto f=[&](unsigned i){return reinterpret_cast<float*>(static_cast<uintptr_t>(p->bindings[i]));};
        auto b=[&](unsigned i){return reinterpret_cast<bool*>(static_cast<uintptr_t>(p->bindings[i]));};
        switch(p->operation) {
        case 0:hipLaunchKernelGGL(mrnf_noise_injection_kernel,grid,block,0,stream,f(0),f(1),f(2),b(3),p->mask_count,p->value[0],p->value[1],p->value[2],p->count,p->seed);break;
        case 1:
            if(p->bindings[3]){hipLaunchKernelGGL((mrnf_decay_kernel<true>),grid,block,0,stream,f(0),f(1),b(2),p->mask_count,p->value[0],p->value[1],p->value[2],p->count,f(3));}
            else {hipLaunchKernelGGL((mrnf_decay_kernel<false>),grid,block,0,stream,f(0),f(1),b(2),p->mask_count,p->value[0],p->value[1],p->value[2],p->count,f(3));}break;
        case 2:hipLaunchKernelGGL(fold_densification_error_and_zero_kernel,grid,block,0,stream,f(0),f(1),p->count);break;
        case 3:hipLaunchKernelGGL(replace_parent_weights_kernel,grid,block,0,stream,f(0),f(1),b(2),b(3),f(4),f(5),p->count);break;
        case 4:hipLaunchKernelGGL(gumbel_key_kernel,grid,block,0,stream,f(0),f(1),p->count,p->seed);break;
        case 5:hipLaunchKernelGGL(gumbel_key_for_indices_kernel,grid,block,0,stream,f(0),reinterpret_cast<const uint32_t*>(p->bindings[1]),f(2),p->count,p->seed);break;
        case 6:hipLaunchKernelGGL(prune_kernel,grid,block,0,stream,f(0),f(1),reinterpret_cast<unsigned char*>(p->bindings[2]),p->count,p->value[0],p->value[1],p->value[2],p->value[3],p->value[4]);break;
        case 7:hipLaunchKernelGGL(identity_kernel,grid,block,0,stream,reinterpret_cast<int64_t*>(p->bindings[0]),p->count);break;
        }
        hip_check(hipGetLastError(),"HIP MRNF submission");
    });
}
