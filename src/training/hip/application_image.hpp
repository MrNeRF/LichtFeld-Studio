/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_image_api.h"
#include "application_image_generated.hpp"
extern "C" int lfs_hip_application_image(const LfsHipImageCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=LFS_HIP_IMAGE_QUANTIZE &&
                call->queue && call->queue<=UINTPTR_MAX && call->count<=INT_MAX,"Invalid HIP training-image descriptor");
        DeviceScope device(call->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const auto* p=call->bindings;
        auto f=[&](unsigned i){return mcmc_pointer<float>(p[i]);};
        using namespace lfs_hip_application_image_kernels;
        const dim3 block(256),grid(blocks(call->count));
        if(call->operation==LFS_HIP_IMAGE_HEATMAP) {
            hipLaunchKernelGGL(update_camera_loss_heatmap_kernel,dim3(1),dim3(1),0,stream,f(0),call->slot,call->value,f(1),f(2),size_t(call->count));
        } else if(call->operation==LFS_HIP_IMAGE_ROI) {
            Affine3x4 transform{};
            for(int row=0;row<3;++row) transform.rows[row]=make_float4(call->affine[row*4],call->affine[row*4+1],call->affine[row*4+2],call->affine[row*4+3]);
            hipLaunchKernelGGL(roi_weight_map_kernel,grid,block,0,stream,f(0),f(1),call->fx,call->fy,call->cx,call->cy,call->w,call->h,transform,
                make_float3(call->minimum[0],call->minimum[1],call->minimum[2]),make_float3(call->maximum[0],call->maximum[1],call->maximum[2]),
                call->inverse!=0,call->value,f(3),f(4),f(5),f(2));
        } else if(call->operation==LFS_HIP_IMAGE_RESIZE) {
            hipLaunchKernelGGL(bilinear_resize_chw_kernel,grid,block,0,stream,f(0),f(1),call->c,call->source_h,call->source_w,call->h,call->w);
        } else if(call->operation==LFS_HIP_IMAGE_RANDOM) {
            hipLaunchKernelGGL(random_background_kernel,grid,block,0,stream,f(0),int(call->count),uint32_t(call->seed));
        } else if(call->operation==LFS_HIP_IMAGE_CANNY) {
            const dim3 image_grid((size_t(call->w)+31)/32,(size_t(call->h)+31)/32),image_block(32,32);
            if(call->byte_input) hipLaunchKernelGGL((fused_canny_edge_filter_chw_kernel<uint8_t>),image_grid,image_block,0,stream,mcmc_pointer<uint8_t>(p[0]),f(1),f(2),call->h,call->w);
            else hipLaunchKernelGGL((fused_canny_edge_filter_chw_kernel<float>),image_grid,image_block,0,stream,f(0),f(1),f(2),call->h,call->w);
        } else if(call->operation==LFS_HIP_IMAGE_NORMALIZE) {
            if(call->count) hipLaunchKernelGGL(normalize_by_device_scalar_kernel,grid,block,0,stream,f(0),size_t(call->count),f(1),call->value);
        } else if(call->operation==LFS_HIP_IMAGE_QUANTIZE) {
            if(call->count) hipLaunchKernelGGL(quantize_to_grid_kernel,grid,block,0,stream,f(0),f(1),size_t(call->count),call->value);
        }
        hip_check(hipGetLastError(),"HIP application training-image submission");
    });
}
