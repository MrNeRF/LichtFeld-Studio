/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_morton_api.h"
#include "application_morton_generated.hpp"
extern "C" int lfs_hip_application_morton(const LfsHipMortonCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=4 && call->count<=INT_MAX &&
                call->queue && call->queue<=UINTPTR_MAX,"Invalid HIP Morton descriptor");
        if(!call->count)return;
        DeviceScope device(call->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const auto* p=call->bindings;const int n=int(call->count);const dim3 grid((call->count+255)/256),block(256);
        auto f=[&](unsigned i){return mcmc_pointer<float>(p[i]);};auto b=[&](unsigned i){return mcmc_pointer<uint8_t>(p[i]);};
        using namespace lfs_hip_application_morton_kernels;
        if(call->operation==0) {
            hipLaunchKernelGGL(encode_from_gpu_bounds,grid,block,0,stream,f(0),f(1),f(2),mcmc_pointer<int32_t>(p[3]),size_t(n));
        } else if(call->operation==4) {
            require(call->width>0 && call->width<=45,"Invalid HIP Morton Q16 cell count");
            namespace quant=lfs_hip_application_sh_quant;
            const size_t tiles=(size_t(n)+31)/32,cell_bytes=tiles*32*2;
            require(call->scratch_bytes>=cell_bytes && p[4],"HIP Morton Q16 scratch smaller than one cell");
            const unsigned group=unsigned(std::min<uint64_t>(call->scratch_bytes/cell_bytes,call->width));
            auto* src=mcmc_pointer<uint16_t>(p[0]);auto* src_bounds=mcmc_pointer<float2>(p[1]);auto* perm=mcmc_pointer<int64_t>(p[2]);
            auto* dst_bounds=mcmc_pointer<float2>(p[3]);auto* scratch=mcmc_pointer<uint16_t>(p[4]);
            hipLaunchKernelGGL(quant::gathered_u16_block_bounds_kernel,grid,block,0,stream,src,src_bounds,perm,dst_bounds,unsigned(n),unsigned(n),call->width);
            hip_check(hipGetLastError(),"HIP Morton Q16 bounds");
            for(unsigned first=0;first<call->width;first+=group){
                const unsigned last=std::min(first+group,call->width);const size_t width=size_t(last-first)*32*2;
                hip_check(hipMemsetAsync(scratch,0,width*tiles,stream),"HIP Morton Q16 scratch clear");
                hipLaunchKernelGGL(quant::encode_u16_gathered_cell_range_kernel,grid,block,0,stream,src,src_bounds,perm,dst_bounds,scratch,unsigned(n),unsigned(n),call->width,first,last);
                hip_check(hipGetLastError(),"HIP Morton Q16 grouped encode");
                hip_check(hipMemcpy2DAsync(src+size_t(first)*32,size_t(call->width)*32*2,scratch,width,width,tiles,hipMemcpyDeviceToDevice,stream),"HIP Morton Q16 grouped copy");
            }
        } else {
            require(call->width>0 && call->width<=48 && (call->bits==8 || call->bits==16),"Invalid HIP Morton joint codec");
            const auto* perm=mcmc_pointer<int64_t>(p[2]);
            auto launch=[&]<int Bits>() {
                if(call->operation==1) {
                    hipLaunchKernelGGL((joint_permute_contiguous_bounds_cu<Bits>),grid,block,0,stream,b(0),f(1),f(4),perm,n,int(call->width));
                    hip_check(hipGetLastError(),"HIP Morton row moment bounds");
                    hipLaunchKernelGGL((joint_permute_contiguous_encode_cu<Bits>),grid,block,0,stream,b(0),f(1),b(3),f(4),perm,n,int(call->width));
                } else {
                    hipLaunchKernelGGL((joint_permute_shN_bounds_cu<Bits>),grid,block,0,stream,b(0),f(1),f(4),perm,n,int(call->width));
                    hip_check(hipGetLastError(),"HIP Morton SH moment bounds");
                    if(call->operation==2) {
                        hipLaunchKernelGGL((joint_permute_shN_encode_cu<Bits>),grid,block,0,stream,b(0),f(1),b(3),f(4),perm,n,int(call->width),0,int(call->width));
                    } else {
                        const size_t tiles=(size_t(n)+31)/32,unit=32*4*(Bits==8?2:4),slot_bytes=tiles*unit;
                        require(call->scratch_bytes>=slot_bytes && p[5],"HIP Morton SH scratch smaller than one slot");
                        const unsigned group=unsigned(std::min<uint64_t>(call->scratch_bytes/slot_bytes,call->width));
                        for(unsigned first=0;first<call->width;first+=group){
                            const unsigned last=std::min(first+group,call->width);const size_t width=size_t(last-first)*unit;
                            hip_check(hipMemsetAsync(b(5),0,width*tiles,stream),"HIP Morton SH scratch clear");
                            hipLaunchKernelGGL((joint_permute_shN_encode_cu<Bits>),grid,block,0,stream,b(0),f(1),b(5),f(4),perm,n,int(call->width),int(first),int(last));
                            hip_check(hipGetLastError(),"HIP Morton SH grouped encode");
                            hip_check(hipMemcpy2DAsync(b(0)+first*unit,size_t(call->width)*unit,b(5),width,width,tiles,hipMemcpyDeviceToDevice,stream),"HIP Morton SH grouped copy");
                        }
                    }
                }
            };
            if(call->bits==8)launch.template operator()<8>();else launch.template operator()<16>();
        }
        hip_check(hipGetLastError(),"HIP application Morton submission");
    });
}
