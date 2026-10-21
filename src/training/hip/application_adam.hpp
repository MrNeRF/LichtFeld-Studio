/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_adam_api.h"
#ifdef LFS_HIP_FAST_RUNTIME
#pragma push_macro("LFS_CUDA_CHECK_MSG")
#pragma push_macro("LFS_CUDA_LAUNCH_CHECK")
#undef LFS_CUDA_CHECK_MSG
#undef LFS_CUDA_LAUNCH_CHECK
#include "application_adam_generated.hpp"
#pragma pop_macro("LFS_CUDA_LAUNCH_CHECK")
#pragma pop_macro("LFS_CUDA_CHECK_MSG")
#endif
extern "C" int lfs_hip_application_adam(const LfsHipAdamCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=2 && call->queue && call->queue<=UINTPTR_MAX,"Invalid HIP Adam descriptor");
#ifndef LFS_HIP_FAST_RUNTIME
        throw std::invalid_argument("Application HIP Adam requires native Fast kernels");
#else
        DeviceScope device(call->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        using namespace fast_lfs::optimizer;
        using namespace fast_lfs::optimizer::kernels::adam;
        if(call->operation==0) {
            require(call->entries>=0 && call->entries<=6,"Invalid HIP Adam batch size");
            if(!call->entries) return;
            JointContiguousBatch batch{}; int maximum=0;
            bool mean_step=false;
            for(int i=0;i<call->entries;++i) {
                const auto& input=call->batch[i];
                require(input.primitives>0 && input.attributes>0 && input.attributes<=16 && input.parameter && input.packed && input.bounds && input.gradient,"Invalid HIP Adam batch entry");
                batch.entries[i]={app_pointer(input.parameter),app_pointer<uint8_t>(input.packed),app_pointer(input.bounds),app_pointer(input.gradient),
                    input.primitives,input.attributes,input.lr,input.bc1_rcp,input.bc2_sqrt_rcp,input.mean_step,input.screen_share};
                maximum=std::max(maximum,input.primitives); mean_step|=input.mean_step!=0;
            }
            auto launch=[&]<bool MeanStep>() {
                hipLaunchKernelGGL((adam_step_joint_contiguous_batched_cu<16,MeanStep>),dim3((size_t(maximum)+255)/256,call->entries),dim3(256),0,stream,
                    batch,app_pointer<bool>(call->masks[0]),call->lengths[0],call->frozen_lr,
                    app_pointer<bool>(call->masks[1]),call->lengths[1],call->crop_lr,call->beta1,call->beta2,call->eps,
                    app_pointer(call->masks[2]),call->lengths[2],call->median_extent,app_pointer<bool>(call->masks[3]),call->lengths[3],
                    app_pointer(call->masks[4]),call->lengths[4],call->screen_limit,call->screen_penalty);
            };
            if(mean_step) launch.template operator()<true>(); else launch.template operator()<false>();
        } else if(call->operation==1) {
            require(call->primitives>0 && call->layout_slots>0 && (call->active_bases==4 || call->active_bases==9 || call->active_bases==16),"Invalid HIP Adam SH layout");
            fast_lfs::rasterization::FusedAdamSettings settings{};
            settings.enabled=true; settings.beta1=call->beta1; settings.beta2=call->beta2; settings.eps=call->eps;
            settings.shN=app_adam_group(call->sh);
            auto launch=[&]<int Bases>() {
                hipLaunchKernelGGL((lfs_hip_application_adam_kernels::adam_step_shN_joint_from_grad_cu<Bases>),dim3((size_t(call->primitives)+255)/256),dim3(256),0,stream,
                    settings,app_pointer(call->gradient),unsigned(call->layout_slots));
            };
            if(call->active_bases==16) launch.template operator()<16>();
            else if(call->active_bases==9) launch.template operator()<9>(); else launch.template operator()<4>();
        } else {
            require(call->count>=0 && call->primitives>0 && call->attributes>0 && (call->bits==8 || call->bits==16) && (call->layout==0 || call->layout==1),"Invalid HIP Adam reset layout");
            if(!call->count) return;
            require(call->indices && call->flags && call->touched && call->sh.packed && call->sh.bounds,"Missing HIP Adam reset bindings");
            hipLaunchKernelGGL(joint_encode_zero_mark_cu,dim3((size_t(call->count)+255)/256),dim3(256),0,stream,
                app_pointer<uint8_t>(call->flags),app_pointer<uint8_t>(call->touched),app_pointer<int64_t>(call->indices),call->count,call->primitives);
            auto launch=[&]<int Bits>() {
                if(call->layout==1) {
                    hipLaunchKernelGGL((joint_encode_zero_shN_cu<Bits>),dim3((size_t(call->primitives)+255)/256),dim3(256),0,stream,
                        app_pointer<uint8_t>(call->sh.packed),app_pointer(call->sh.bounds),app_pointer<uint8_t>(call->flags),app_pointer<uint8_t>(call->touched),call->attributes,call->primitives);
                } else {
                    hipLaunchKernelGGL((joint_encode_zero_rows_cu<Bits>),dim3((size_t(call->primitives)+255)/256),dim3(256),0,stream,
                        app_pointer<uint8_t>(call->sh.packed),app_pointer(call->sh.bounds),app_pointer<uint8_t>(call->flags),app_pointer<uint8_t>(call->touched),call->attributes,call->primitives);
                }
            };
            if(call->bits==16) launch.template operator()<16>(); else launch.template operator()<8>();
        }
        hip_check(hipGetLastError(),"HIP application Adam submission");
#endif
    });
}
