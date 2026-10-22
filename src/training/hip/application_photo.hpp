/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_photo_api.h"
#include "application_photo_generated.hpp"
#include "photo_plane_policy.hpp"
namespace {
bool application_photo_plane_enabled(const LfsHipPhotoCall& call) {
    const char* profiles=std::getenv("LFS_HIP_DISABLE_ARCH_PROFILES");
    const char* plane=std::getenv("LFS_HIP_DISABLE_PLANE_SSIM");
    const bool disabled=(profiles && std::string_view(profiles)!="0") ||
                        (plane && std::string_view(plane)!="0");
    // Do not query device properties for paths that cannot be admitted.
    if (!lfs::training::hip::photo_plane_backward_enabled(
            call.path,call.byte_target,call.n,call.c,call.h,call.w,"gfx1150",disabled))
        return false;
    struct DeviceProfile { int ordinal=-1; std::string architecture; };
    thread_local DeviceProfile profile;
    if (profile.ordinal!=call.device) {
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties,call.device),"HIP photometric device profile");
        profile.architecture=lfs_hip_base_architecture(properties.gcnArchName);
        profile.ordinal=call.device;
    }
    return lfs::training::hip::photo_plane_backward_enabled(
        call.path,call.byte_target,call.n,call.c,call.h,call.w,profile.architecture,disabled);
}
template<class Target,class Mask>
void application_photo_launch(const LfsHipPhotoCall& call) {
    using namespace lfs_hip_application_photo_kernels;
    const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call.queue));
    const auto* p=call.bindings;
    auto f=[&](unsigned slot){ return mcmc_pointer<float>(p[slot]); };
    const auto* target=mcmc_pointer<Target>(p[2]); const auto* mask=mcmc_pointer<Mask>(p[3]);
    const int n=call.n,c=call.c,h=call.h,w=call.w;
    const size_t count=size_t(n)*c*h*w;
    auto* partial=mcmc_pointer<__half>(p[4]);
    const dim3 grid((w+15)/16,(h+15)/16,n),block(16,16);
    const float c1=.01f*.01f,c2=.03f*.03f;
    const bool valid=call.valid_padding!=0, masked=call.path==4 || call.path==5, decoupled=call.path==3 || call.path==5;
    const size_t normalized=size_t(n)*c*(valid && h>10 ? h-10 : h)*(valid && w>10 ? w-10 : w);
    const float norm=1.f/float(normalized);
    if(call.path==7) {
        hipLaunchKernelGGL(ssim_to_error_map_kernel,dim3(blocks(size_t(h)*w)),dim3(256),0,stream,f(5),f(9),c,h,w);
        return;
    }
    if(call.path==6) {
        hipLaunchKernelGGL((maskedSSIMMapCUDA<Target,Mask>),grid,block,0,stream,h,w,c,f(0),target,mask,f(5));
        return;
    }
    if(call.backward) {
        if(call.path==1) {
            hipLaunchKernelGGL((fusedssim_backwardCUDA<Target>),grid,block,0,stream,h,w,c,c1,c2,f(0),target,f(12),f(7),partial,partial+count,partial+2*count);
        } else if(masked) {
            if(decoupled) {
                hipLaunchKernelGGL((maskedFusedL1SSIMBackwardCUDA<Target,Mask,false>),grid,block,0,stream,
                    call.weight,call.normalization,h,w,c,c1,c2,f(0),target,mask,f(7),partial,static_cast<__half*>(nullptr),static_cast<__half*>(nullptr));
                hipLaunchKernelGGL((maskedFusedL1SSIMBackwardCUDA<Target,Mask,true,true,true>),grid,block,0,stream,
                    1.f,call.normalization,h,w,c,c1,c2,f(1),target,mask,f(8),partial+count,partial+2*count,partial+3*count);
            } else {
                hipLaunchKernelGGL((maskedFusedL1SSIMBackwardCUDA<Target,Mask>),grid,block,0,stream,
                    call.weight,call.normalization,h,w,c,c1,c2,f(0),target,mask,f(7),partial,partial+count,partial+2*count);
            }
        } else if(decoupled) {
            hipLaunchKernelGGL((fusedL1SSIMBackwardCUDA<Target,false>),grid,block,0,stream,
                call.weight,h,w,c,c1,c2,norm,valid,f(0),target,f(7),partial,static_cast<__half*>(nullptr),static_cast<__half*>(nullptr));
            hipLaunchKernelGGL((fusedL1SSIMBackwardCUDA<Target,true,true>),grid,block,0,stream,
                1.f,h,w,c,c1,c2,norm,valid,f(1),target,f(8),partial+count,partial+2*count,partial+3*count);
        } else if(call.path==2) {
            if constexpr (std::is_same_v<Target,float>) {
                const bool plane=application_photo_plane_enabled(call);
                if(std::getenv("LFS_HIP_VALIDATION_PHOTO_DISPATCH"))
                    std::fprintf(stderr,"HIP fused SSIM backward: device=%d plane=%d target=Float32\n",call.device,int(plane));
                if(plane) {
                    hipLaunchKernelGGL((fusedL1SSIMBackwardHIP<Target,true,false>),
                        dim3((w+31)/32,(h+7)/8,n*c),dim3(32,8),0,stream,
                        call.weight,h,w,c,norm,valid,f(0),target,f(7),partial,partial+count,partial+2*count);
                    return;
                }
            }
            hipLaunchKernelGGL((fusedL1SSIMBackwardCUDA<Target>),grid,block,0,stream,
                call.weight,h,w,c,c1,c2,norm,valid,f(0),target,f(7),partial,partial+count,partial+2*count);
        }
        return;
    }
    const int reduction_blocks=int(std::min<size_t>((count+255)/256,1024));
    const dim3 red_grid(reduction_blocks),red_block(256);
    if(call.path==0) {
        hipLaunchKernelGGL((reduction::fused_l1_kernel<Target>),red_grid,red_block,0,stream,f(0),target,f(7),f(10),count,1.f/float(count));
        hipLaunchKernelGGL(reduction::final_reduce_kernel,dim3(1),red_block,0,stream,f(10),f(9),reduction_blocks,1.f/float(count));
        return;
    }
    if(call.path==1 || call.path==8 || call.path==9) {
        hipLaunchKernelGGL((fusedssimCUDA<Target>),grid,block,0,stream,h,w,c,c1,c2,f(0),target,f(5),f(6),partial,
                          partial ? partial+count : nullptr,partial ? partial+2*count : nullptr);
        if(call.path==9) return;
        hipLaunchKernelGGL(reduction::fused_ssim_mean_kernel,red_grid,red_block,0,stream,f(5),f(10),n,c,h,w,valid);
        hipLaunchKernelGGL(reduction::final_mean_reduce_kernel,dim3(1),red_block,0,stream,f(10),f(9),reduction_blocks,normalized);
        return;
    }
    if(masked) {
        if(decoupled) {
            hipLaunchKernelGGL((maskedDecoupledFusedL1SSIMForwardCUDA<Target,Mask>),grid,block,0,stream,
                h,w,c,c1,c2,call.weight,f(0),f(1),target,mask,partial,partial+count,partial+2*count,partial+3*count,f(5),f(6));
        } else {
            hipLaunchKernelGGL((maskedFusedL1SSIMForwardCUDA<Target,Mask>),grid,block,0,stream,
                h,w,c,c1,c2,f(0),target,mask,partial,partial+count,partial+2*count,f(5),f(6));
        }
        hipLaunchKernelGGL((reduction::masked_fused_l1_ssim_sum_kernel<Target,Mask>),red_grid,red_block,0,stream,
            f(0),target,f(5),mask,f(10),call.weight,n,c,h,w);
        if(call.denominator>0) {
            hipLaunchKernelGGL(reduction::fixed_denominator_mean_kernel,dim3(1),red_block,0,stream,f(10),f(9),reduction_blocks,call.denominator);
        } else {
            const int mask_blocks=int(std::min<size_t>((size_t(h)*w+255)/256,1024));
            hipLaunchKernelGGL((reduction::mask_sum_kernel<Mask>),dim3(mask_blocks),red_block,0,stream,mask,f(10)+1024,h,w);
            hipLaunchKernelGGL(reduction::final_masked_mean_reduce_kernel,dim3(1),red_block,0,stream,
                f(10),f(10)+1024,f(9),f(11),reduction_blocks,mask_blocks,n,c);
        }
    } else {
        if(decoupled) {
            hipLaunchKernelGGL((decoupledFusedL1SSIMForwardCUDA<Target>),grid,block,0,stream,
                h,w,c,c1,c2,call.weight,f(0),f(1),target,partial,partial+count,partial+2*count,partial+3*count,f(5),f(6));
        } else {
            hipLaunchKernelGGL((fusedL1SSIMForwardCUDA<Target>),grid,block,0,stream,
                h,w,c,c1,c2,f(0),target,partial,partial+count,partial+2*count,f(5),f(6));
        }
        hipLaunchKernelGGL((reduction::fused_l1_ssim_sum_kernel<Target>),red_grid,red_block,0,stream,
            f(0),target,f(5),f(10),call.weight,n,c,h,w,valid);
        hipLaunchKernelGGL(reduction::final_mean_reduce_kernel,dim3(1),red_block,0,stream,f(10),f(9),reduction_blocks,normalized);
    }
}
}
extern "C" int lfs_hip_application_photo(const LfsHipPhotoCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->path<=9 && call->queue && call->queue<=UINTPTR_MAX &&
                call->n>0 && call->c>0 && call->h>0 && call->w>0 && uint64_t(call->n)*call->c*call->h*call->w<=INT_MAX &&
                call->backward<=1 && call->byte_target<=1 && call->byte_mask<=1 && call->valid_padding<=1 &&
                std::isfinite(call->weight) && call->weight>=0 && call->weight<=1 &&
                std::isfinite(call->denominator) && call->denominator>=0,"Invalid HIP photometric descriptor");
        DeviceScope device(call->device);
        if(call->byte_target) {
            if(call->byte_mask) application_photo_launch<uint8_t,uint8_t>(*call);
            else application_photo_launch<uint8_t,float>(*call);
        } else {
            if(call->byte_mask) application_photo_launch<float,uint8_t>(*call);
            else application_photo_launch<float,float>(*call);
        }
        hip_check(hipGetLastError(),"HIP application photometric submission");
    });
}
