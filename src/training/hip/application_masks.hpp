/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_masks_api.h"
#include "mask_preprocess.hpp"
namespace lfs::training::kernels {void launch_hip_mesh(const LfsHipMeshCall&,hipStream_t);}
extern "C" int lfs_hip_application_masks(const LfsHipColorCall* p,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->operation<3&&p->queue&&p->queue<=UINTPTR_MAX,"Invalid HIP mask descriptor");
        const auto* d=p->dimensions;const auto* v=p->values;
        require(d[0]>0&&d[1]>0&&uint64_t(d[0])*d[1]<=INT_MAX-255&&p->flags[0]<=1&&p->flags[1]<=1,"Invalid HIP mask dimensions or mode");
        require(p->bindings[1]&&p->bindings[3],"Missing HIP mask operand");
        if(p->operation)require(p->bindings[0]&&p->bindings[4]&&p->bindings[5],"Missing HIP mask penalty operand");
        require(std::isfinite(v[0])&&std::isfinite(v[1]),"Invalid HIP mask scalar");
        DeviceScope selected(p->device);const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue));
        auto f=[&](int i){require(p->bindings[i]<=UINTPTR_MAX,"Invalid HIP mask address");return reinterpret_cast<float*>(static_cast<uintptr_t>(p->bindings[i]));};
        auto* bytes=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(p->bindings[1]));
        using namespace lfs::training::kernels;using namespace lfs::hip_mask_ops;
        if(p->operation==0) {
            if(p->flags[0])launch_fuse_photometric_mask_weight_u8(bytes,f(2),f(3),d[0],d[1],static_cast<MaskPhotoMode>(p->flags[1]),stream);
            else launch_fuse_photometric_mask_weight_f32(f(1),f(2),f(3),d[0],d[1],static_cast<MaskPhotoMode>(p->flags[1]),stream);
        } else if(p->operation==1) {
            if(p->flags[0])launch_fuse_mask_opacity_penalty_u8(f(0),bytes,f(2),f(3),f(4),f(5),d[0],d[1],v[0],v[1],static_cast<MaskOpacityMode>(p->flags[1]),stream);
            else launch_fuse_mask_opacity_penalty_f32(f(0),f(1),f(2),f(3),f(4),f(5),d[0],d[1],v[0],v[1],static_cast<MaskOpacityMode>(p->flags[1]),stream);
        } else {
            if(p->flags[0])launch_fuse_alpha_consistent_u8(f(0),bytes,f(2),f(3),f(4),f(5),d[0],d[1],v[1],stream);
            else launch_fuse_alpha_consistent_f32(f(0),f(1),f(2),f(3),f(4),f(5),d[0],d[1],v[1],stream);
        }
        hip_check(hipGetLastError(),"HIP mask submission");
    });
}
extern "C" int lfs_hip_application_mesh(const LfsHipMeshCall* p,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(p&&p->struct_bytes==sizeof(*p)&&p->operation<3&&p->queue&&p->queue<=UINTPTR_MAX,"Invalid HIP mesh mask descriptor");
        const auto& k=p->camera;
        require(k.width>0&&k.height>0&&uint64_t(k.width)*k.height<=INT_MAX&&p->mask,"Invalid HIP mesh mask image");
        require(p->vertex_count>0&&p->vertex_count<=INT_MAX/3&&p->vertices&&std::isfinite(k.fx)&&k.fx>0&&std::isfinite(k.fy)&&k.fy>0&&std::isfinite(k.cx)&&std::isfinite(k.cy),"Invalid HIP mesh mask camera or points");
        require(k.num_distortion>=0&&k.num_distortion<=12&&(k.distorted==0||k.distorted==1),"Invalid HIP mesh distortion descriptor");
        if(p->operation==2)require(p->radius>=0&&p->radius<=32767,"Invalid HIP point coverage radius");
        else require(p->face_count>0&&p->face_count<=INT_MAX/3&&p->indices&&p->large_faces&&p->counters&&std::isfinite(p->z_near)&&p->z_near>0&&(!k.distorted||p->samples),"Invalid HIP mesh coverage operands");
        DeviceScope selected(p->device);lfs::training::kernels::launch_hip_mesh(*p,reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(p->queue)));hip_check(hipGetLastError(),"HIP mesh coverage submission");
    });
}
