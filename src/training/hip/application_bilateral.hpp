/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_color_api.h"
#include "bilateral_grid.cuh"
extern "C" int lfs_hip_application_bilateral(const LfsHipColorCall* p,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->operation<=7&&p->queue&&p->queue<=UINTPTR_MAX,"Invalid HIP bilateral descriptor");
        for(float v:p->values)require(std::isfinite(v),"Invalid HIP bilateral scalar");
        for(unsigned flag:p->flags)require(flag<=1,"Invalid HIP bilateral flag");
        require(!p->flags[3] || (p->operation==1 && !p->flags[1]),"Padded HIP bilateral layout requires affine backward");
        const unsigned required[]={0x0b,0x37,0x07,0x03,0x07,0x1f,0x0f,0x03};
        for(unsigned i=0;i<6;++i)if(required[p->operation]&(1u<<i))require(p->bindings[i],"Missing HIP bilateral binding");
        const auto* d=p->dimensions;const auto* v=p->values;
        if(p->operation<=4) {
            uint64_t cells=1;
            for(int i=0;i<5;++i){require(d[i]>0&&uint64_t(d[i])<=uint64_t(INT_MAX)/cells,"Invalid HIP bilateral grid extent");cells*=d[i];}
        } else require(d[0]>0,"Empty HIP bilateral update");
        if(p->operation<=1)require(d[0]==1&&d[5]>0&&d[6]>0&&uint64_t(d[5])*d[6]<=INT_MAX/3,"Invalid HIP bilateral image extent");
        DeviceScope selected(p->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue));
        auto f=[&](int i){return reinterpret_cast<float*>(static_cast<uintptr_t>(p->bindings[i]));};
        using namespace lfs::training::kernels;
        switch(p->operation) {
        case 0: {
            const auto fn=p->flags[1]?(p->flags[0]?launch_bilateral_grid_slice_forward_exposure_chroma_chw:launch_bilateral_grid_slice_forward_exposure_chroma):(p->flags[0]?launch_bilateral_grid_slice_forward_chw:launch_bilateral_grid_slice_forward);
            fn(f(0),f(1),f(3),d[2],d[3],d[4],d[5],d[6],f(2),stream);break;
        }
        case 1: {
            const auto fn=p->flags[1]?(p->flags[0]?launch_bilateral_grid_slice_backward_exposure_chroma_chw:launch_bilateral_grid_slice_backward_exposure_chroma):(p->flags[0]?launch_bilateral_grid_slice_backward_chw:launch_bilateral_grid_slice_backward);
            fn(f(0),f(1),f(2),f(4),f(5),d[2],d[3],d[4],d[5],d[6],f(3),stream,(p->flags[1]?p->flags[2]:p->flags[3])!=0);break;
        }
        case 2:launch_bilateral_grid_tv_forward(f(0),f(1),f(2),d[0],d[1],d[2],d[3],d[4],d[7],stream);break;
        case 3:launch_bilateral_grid_tv_backward(f(0),v[0],f(1),d[0],d[1],d[2],d[3],d[4],d[7],stream);break;
        case 4:launch_bilateral_grid_project_mean(f(0),f(1),f(2),d[0],d[1],d[2],d[3],d[4],p->flags[0]!=0,stream);break;
        case 5:launch_bilateral_grid_update_shared_offset(f(0),f(1),f(2),f(3),f(4),d[0],v[0],v[1],stream);break;
        case 6:launch_bilateral_grid_adam_update(f(0),f(1),f(2),f(3),d[0],v[0],v[1],v[2],v[3],v[4],v[5],stream);break;
        case 7:launch_bilateral_grid_scale_moments(f(0),f(1),d[0],v[0],v[1],stream);break;
        }
        hip_check(hipGetLastError(),"HIP bilateral submission");
    });
}
