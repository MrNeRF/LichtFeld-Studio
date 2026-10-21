/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_refine_api.h"
#include "application_refine_generated.hpp"
extern "C" int lfs_hip_application_refine(const LfsHipRefineCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=LFS_HIP_REFINE_ROTATION &&
                call->queue && call->queue<=UINTPTR_MAX && call->count<=uint64_t(INT_MAX),"Invalid HIP refinement descriptor");
        DeviceScope device(call->device);
        if(!call->count && call->operation!=LFS_HIP_REFINE_COUNTS) return;
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const dim3 grid(blocks(call->count)), block(256);
        const auto* p=call->bindings;
        auto f=[&](unsigned i) { return mcmc_pointer<float>(p[i]); };
        auto b=[&](unsigned i) { return mcmc_pointer<bool>(p[i]); };
        auto idx=[&](unsigned i) { return mcmc_pointer<int64_t>(p[i]); };
        using namespace lfs_hip_refine_kernels;
        switch(call->operation) {
        case LFS_HIP_REFINE_SPLIT:
            hipLaunchKernelGGL(long_axis_split_gaussians_inplace_kernel,grid,block,0,stream,
                f(0),f(1),f(2),f(3),static_cast<float*>(nullptr),f(4),
                f(5),f(6),f(7),f(8),static_cast<float*>(nullptr),f(9),idx(10),int(call->count),0);
            break;
        case LFS_HIP_REFINE_FILL:
            hipLaunchKernelGGL(fill_free_slots_fused_kernel,grid,block,0,stream,
                idx(0),size_t(call->count),f(1),f(2),f(3),f(4),f(5),f(6),f(7),f(8),f(9),f(10),0,b(11),size_t(call->rows));
            break;
        case LFS_HIP_REFINE_COUNTS:
            hipLaunchKernelGGL(packed_refine_counts_kernel,dim3(1),block,0,stream,
                b(0),size_t(call->lengths[0]),b(1),size_t(call->lengths[1]),f(2),size_t(call->lengths[2]),f(3),size_t(call->lengths[3]),idx(4));
            break;
        case LFS_HIP_REFINE_NORMALIZE: {
            // Tiny fixed workspace is owned by the Core Tensor runtime, never
            // an input-sized sort or an untracked allocator in the trainer.
            auto* state=mcmc_pointer<radix_select::State>(p[1]);
            auto* histogram=mcmc_pointer<unsigned>(p[2]);
            hip_check(hipMemsetAsync(state,0,sizeof(*state),stream),"HIP refinement median state clear");
            hip_check(hipMemsetAsync(histogram,0,radix_select::kBins*sizeof(unsigned),stream),"HIP refinement median histogram clear");
            hipLaunchKernelGGL(zero_nan_kernel,grid,block,0,stream,f(0),size_t(call->count));
            const dim3 radix_grid(std::clamp<size_t>((call->count+4095)/4096,1,512));
#define LFS_HIP_MEDIAN_PASS(PASS) \
            hipLaunchKernelGGL((radix_select::histogram_kernel<true,PASS>),radix_grid,block,0,stream,f(0),size_t(call->count),state,histogram); \
            hipLaunchKernelGGL((radix_select::pick_kernel<PASS>),dim3(1),block,0,stream,true,size_t(call->count),state,histogram)
            LFS_HIP_MEDIAN_PASS(0);
            LFS_HIP_MEDIAN_PASS(1);
            LFS_HIP_MEDIAN_PASS(2);
#undef LFS_HIP_MEDIAN_PASS
            hipLaunchKernelGGL(div_by_positive_median_or_zero_kernel,grid,block,0,stream,f(0),size_t(call->count),state);
            break;
        }
        case LFS_HIP_REFINE_CLIP:
            if(!(call->value>0 && call->value<1)) return;
            hipLaunchKernelGGL(clip_log_scale_by_screen_share_kernel,grid,block,0,stream,f(0),f(1),b(2),size_t(call->lengths[0]),call->value,size_t(call->count));
            break;
        case LFS_HIP_REFINE_SCORES:
            hipLaunchKernelGGL(oversize_split_scores_kernel,grid,block,0,stream,f(0),f(1),b(2),size_t(call->lengths[0]),f(3),call->value,size_t(call->count));
            break;
        case LFS_HIP_REFINE_DEAD:
            hipLaunchKernelGGL(compute_dead_mask_kernel,grid,block,0,stream,f(0),f(1),mcmc_pointer<uint8_t>(p[2]),size_t(call->count),call->value);
            break;
        case LFS_HIP_REFINE_ROTATION:
            hipLaunchKernelGGL(compute_near_zero_rotation_mask_kernel,grid,block,0,stream,f(0),mcmc_pointer<uint8_t>(p[1]),size_t(call->count));
            break;
        }
        hip_check(hipGetLastError(),"HIP refinement submission");
    });
}
