/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_color_api.h"
#include "depth_loss.hpp"
#include "normal_loss.hpp"
#include "normal_consistency_loss.hpp"
extern "C" int lfs_hip_application_geometry(const LfsHipColorCall* p,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->operation<4&&p->queue&&p->queue<=UINTPTR_MAX,"Invalid HIP geometry descriptor");
        const auto* d=p->dimensions;const auto* v=p->values;
        require(d[0]>0&&d[1]>0&&uint64_t(d[0])*d[1]<=INT_MAX/3,"Invalid HIP geometry image dimensions");
        for(float x:p->values)require(std::isfinite(x),"Nonfinite HIP geometry scalar");
        for(unsigned x:p->flags)require(x<=1,"Invalid HIP geometry flag");
        const unsigned required[]={0x7f,0x6f,0x17f,0x177};
        for(unsigned i=0;i<9;++i)if(required[p->operation]&(1u<<i))require(p->bindings[i],"Missing HIP geometry binding");
        if(p->operation>=2)require(p->bindings[10]||(v[0]!=0&&v[1]!=0),"HIP geometry requires valid intrinsics or camera rays");
        DeviceScope selected(p->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue));
        auto f=[&](int i){return reinterpret_cast<float*>(static_cast<uintptr_t>(p->bindings[i]));};
        using namespace lfs::training::kernels;
        switch(p->operation) {
        case 0: {
            require(d[2]==0||d[2]==1,"Invalid HIP depth anchor model");
            DepthAnchor anchor;anchor.valid=p->flags[0]!=0;anchor.model=d[2];anchor.scale=v[3];anchor.shift=v[4];anchor.floor=v[5];
            launch_depth_loss(f(0),f(1),f(2),f(3),f(4),f(5),f(6),d[0],d[1],v[0],v[1],v[2],&anchor,stream,f(7));break;
        }
        case 1:launch_normal_loss(f(0),f(1),f(2),f(3),f(5),f(6),d[0],d[1],v[0],stream,f(7));break;
        case 2:launch_normal_consistency_loss(f(0),f(1),f(2),f(3),f(4),f(5),f(6),f(8),d[0],d[1],v[0],v[1],v[2],v[3],v[4],stream,f(9),f(10),p->flags[0]!=0);break;
        case 3:launch_normal_prior_depth_loss(f(0),f(1),f(2),f(4),f(5),f(6),f(8),d[0],d[1],v[0],v[1],v[2],v[3],v[4],stream,f(9),f(10),p->flags[0]!=0);break;
        }
        hip_check(hipGetLastError(),"HIP geometry submission");
    });
}
extern "C" int lfs_hip_application_anchor(const LfsHipAnchorCall* p,float* output,uint32_t* count,LfsHipError* error) {
    if(count)*count=0;
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&count&&output&&p->capacity>=262144&&p->queue&&p->queue<=UINTPTR_MAX,"Invalid HIP anchor descriptor");
        require(p->width>0&&p->height>0&&uint64_t(p->width)*p->height<=INT_MAX&&p->points&&p->view&&p->prior,"Invalid HIP anchor operands");
        require(p->point_count<=uint64_t(INT_MAX)/3,"HIP anchor point count exceeds supported range");
        require(p->model==0||p->model==2||p->model==3||p->model==4,"Unsupported HIP anchor camera");
        DeviceScope selected(p->device);
        using namespace lfs::training::kernels;
        DepthCameraProjection projection;projection.model=p->model;
        std::copy_n(p->radial,6,projection.radial);std::copy_n(p->tangential,2,projection.tangential);std::copy_n(p->thin_prism,4,projection.thin_prism);
        auto f=[](uint64_t x){require(x<=UINTPTR_MAX,"Invalid HIP anchor address");return reinterpret_cast<const float*>(static_cast<uintptr_t>(x));};
        auto samples=collect_depth_anchor_samples(f(p->points),p->point_count,f(p->view),p->intrinsics[0],p->intrinsics[1],p->intrinsics[2],p->intrinsics[3],f(p->prior),p->width,p->height,p->near_plane,p->aabb_lo,p->aabb_hi,reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue)),projection);
        require(samples.size()<=p->capacity,"HIP anchor output capacity exceeded");
        static_assert(sizeof(lfs::gpu_ops::AnchorSample)==2*sizeof(float));
        std::memcpy(output,samples.data(),samples.size()*sizeof(lfs::gpu_ops::AnchorSample));*count=static_cast<uint32_t>(samples.size());
    });
}
