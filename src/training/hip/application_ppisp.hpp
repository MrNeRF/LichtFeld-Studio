/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_color_api.h"
#include "ppisp.cuh"
extern "C" int lfs_hip_application_ppisp(const LfsHipColorCall* p,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->operation<=6&&p->queue&&p->queue<=UINTPTR_MAX,"Invalid HIP ISP descriptor");
        for(float v:p->values)require(std::isfinite(v),"Invalid HIP ISP scalar");
        const auto* d=p->dimensions;const auto* v=p->values;
        if(p->operation<=1)require(d[0]>0&&d[1]>0&&uint64_t(d[0])*d[1]<=INT_MAX/3&&d[6]>=0&&d[7]>=0,"Invalid HIP ISP image extent");
        DeviceScope selected(p->device);const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue));
        auto f=[&](int i){return reinterpret_cast<float*>(static_cast<uintptr_t>(p->bindings[i]));};
        using namespace lfs::training::kernels;
        switch(p->operation) {
        case 0:require(f(4)&&f(5)&&d[2]>=0&&d[3]>=0&&int64_t(d[2])+d[1]<=d[4]&&int64_t(d[3])+d[0]<=d[5],"Invalid HIP ISP crop/bindings");
            launch_ppisp_forward_chw_region(f(0),f(1),f(2),f(3),f(4),f(5),d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],stream);break;
        case 1:require(f(4)&&f(5)&&f(10),"Missing HIP ISP image/gradient binding");
            launch_ppisp_backward_chw(f(0),f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10),d[0],d[1],d[6],d[7],d[8],d[9],stream);break;
        case 2:require(d[0]>0&&f(0)&&f(1)&&f(2)&&f(3),"Invalid HIP ISP Adam bindings");
            launch_ppisp_adam_update(f(0),f(1),f(2),f(3),d[0],v[0],v[1],v[2],v[3],v[4],v[5],stream);break;
        case 3: {
            PPISPAdamGroup groups[4]{};uint64_t total=0;
            for(unsigned i=0;i<4;++i){require(d[i]>=0,"Invalid HIP ISP batch count");total+=d[i];require(!d[i]||(f(4*i)&&f(4*i+1)&&f(4*i+2)&&f(4*i+3)),"Missing HIP ISP batch binding");groups[i]={f(4*i),f(4*i+1),f(4*i+2),f(4*i+3),d[i]};}
            require(total>0&&total<=INT_MAX,"Invalid HIP ISP batch extent");
            launch_ppisp_adam_update_batched(groups[0],groups[1],groups[2],groups[3],v[0],v[1],v[2],v[3],v[4],v[5],stream);break;
        }
        case 4:require(d[0]>0&&f(0),"Invalid HIP ISP vignetting parameters");launch_ppisp_vignetting_reg(f(0),f(1),f(2),d[0],v[0],v[1],v[2],stream);break;
        case 5:require(d[0]>0&&f(0)&&f(1),"Invalid HIP ISP mean parameters");launch_ppisp_project_mean(f(0),f(1),d[0],stream);break;
        case 6:require(d[0]>0&&d[1]>0&&d[0]<=INT_MAX/15&&d[1]<=INT_MAX/8&&f(0)&&f(1)&&f(2)&&f(3),"Invalid HIP ISP initialization parameters");
            launch_ppisp_init_identity(f(0),f(1),f(2),f(3),d[0],d[1],stream);break;
        }
        hip_check(hipGetLastError(),"HIP ISP submission");
    });
}
