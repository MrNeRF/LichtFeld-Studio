/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_mcmc_api.h"
#include "application_mcmc_generated.hpp"
#include <mutex>
#include <climits>
#include <unordered_set>
namespace {
    std::mutex application_mcmc_initialization_mutex;
    std::unordered_set<int> application_mcmc_initialized_devices;
    template<class T> T* mcmc_pointer(uint64_t address) { return reinterpret_cast<T*>(static_cast<uintptr_t>(address)); }
}
extern "C" {
int lfs_hip_application_mcmc_initialize(int32_t ordinal, uint64_t queue_address, int32_t n_max, LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(queue_address && queue_address<=UINTPTR_MAX && n_max>0 && n_max<=51,"Invalid HIP MCMC initialization");
        DeviceScope selected(ordinal);
        std::lock_guard lock(application_mcmc_initialization_mutex);
        if(application_mcmc_initialized_devices.contains(ordinal)) return;
        const auto queue=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(queue_address));
        unsigned flags=0;
        hip_check(hipStreamGetFlags(queue,&flags),"HIP MCMC initialization queue");
        std::array<float,51*51> coefficients{};
        // Initialize the complete table once per physical HIP device. Different
        // n_max callers cannot erase coefficients used by another model.
        for(int n=0;n<51;++n) {
            float binomial=1.f;
            for(int k=0;k<=n;++k) {
                const float sign=(k%2==0) ? 1.f : -1.f;
                coefficients[n*51+k]=binomial*sign*(1.f/std::sqrt(float(k+1)));
                if(k<n) binomial*=float(n-k)/float(k+1);
            }
        }
        hip_check(hipMemcpyToSymbolAsync(HIP_SYMBOL(lfs_hip_mcmc_kernels::d_relocation_coefficients),
            coefficients.data(),sizeof(coefficients),0,hipMemcpyHostToDevice,queue),"HIP MCMC coefficient upload");
        hip_check(hipStreamSynchronize(queue),"HIP MCMC coefficient completion");
        application_mcmc_initialized_devices.insert(ordinal);
    });
}
int lfs_hip_application_mcmc(const LfsHipMcmcCall* call, LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->queue && call->queue<=UINTPTR_MAX &&
                call->operation<=LFS_HIP_MCMC_FOLD_ERROR && call->count<=uint64_t(INT_MAX),"Invalid HIP MCMC call");
        DeviceScope selected(call->device);
        if(!call->count) return;
        const auto queue=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const auto* p=call->bindings;
        auto require_bindings=[&](unsigned count) { for(unsigned i=0;i<count;++i) require(p[i]!=0,"Missing HIP MCMC tensor binding"); };
        const dim3 grid(blocks(call->count)), block(256);
        using namespace lfs_hip_mcmc_kernels;
        switch(call->operation) {
        case LFS_HIP_MCMC_RELOCATE: {
            require_bindings(5);
            require(std::isfinite(call->value) && call->value>=0.f && call->value<1.f,"Invalid HIP MCMC minimum opacity");
            {
                std::lock_guard lock(application_mcmc_initialization_mutex);
                require(application_mcmc_initialized_devices.contains(call->device),"HIP MCMC coefficients are not initialized");
            }
            hipLaunchKernelGGL(relocation_kernel,grid,block,0,queue,mcmc_pointer<float>(p[0]),mcmc_pointer<float>(p[1]),mcmc_pointer<int32_t>(p[2]),call->value,
                mcmc_pointer<float>(p[3]),mcmc_pointer<float>(p[4]),size_t(call->count));
            break;
        }
        case LFS_HIP_MCMC_NOISE:
            require_bindings(4);
            require(std::isfinite(call->value) && call->value>=0.f,"Invalid HIP MCMC noise rate");
            hipLaunchKernelGGL(inject_noise_kernel,grid,block,0,queue,mcmc_pointer<float>(p[0]),mcmc_pointer<float>(p[1]),mcmc_pointer<float>(p[2]),
                mcmc_pointer<float>(p[3]),mcmc_pointer<bool>(p[4]),size_t(call->mask_count),call->value,size_t(call->count),call->seed);
            break;
        case LFS_HIP_MCMC_COPY:
            require_bindings(7);
            hipLaunchKernelGGL(copy_gaussian_params_kernel,grid,block,0,queue,mcmc_pointer<int64_t>(p[0]),mcmc_pointer<int64_t>(p[1]),mcmc_pointer<float>(p[2]),
                mcmc_pointer<float>(p[3]),static_cast<float*>(nullptr),mcmc_pointer<float>(p[4]),mcmc_pointer<float>(p[5]),mcmc_pointer<float>(p[6]),
                size_t(call->count),size_t(0),0,size_t(call->model_count));
            break;
        case LFS_HIP_MCMC_UPDATE:
            require_bindings(5);
            hipLaunchKernelGGL(update_scaling_opacity_kernel,grid,block,0,queue,mcmc_pointer<int64_t>(p[0]),mcmc_pointer<float>(p[1]),mcmc_pointer<float>(p[2]),
                mcmc_pointer<float>(p[3]),mcmc_pointer<float>(p[4]),size_t(call->count),0,size_t(call->model_count));
            break;
        case LFS_HIP_MCMC_SAMPLE_ALIVE:
            if(!call->domain_count) return;
            require_bindings(7);
            hipLaunchKernelGGL(multinomial_sample_and_gather_kernel,grid,block,0,queue,mcmc_pointer<float>(p[0]),mcmc_pointer<float>(p[1]),mcmc_pointer<int64_t>(p[2]),
                mcmc_pointer<float>(p[3]),size_t(call->domain_count),size_t(call->count),call->seed,mcmc_pointer<int64_t>(p[4]),mcmc_pointer<float>(p[5]),mcmc_pointer<float>(p[6]),size_t(call->model_count));
            break;
        case LFS_HIP_MCMC_SAMPLE_ALL:
            if(!call->model_count) return;
            require_bindings(6);
            hipLaunchKernelGGL(multinomial_sample_all_kernel,grid,block,0,queue,mcmc_pointer<float>(p[0]),mcmc_pointer<float>(p[1]),mcmc_pointer<float>(p[2]),
                size_t(call->model_count),size_t(call->count),call->seed,mcmc_pointer<int64_t>(p[3]),mcmc_pointer<float>(p[4]),mcmc_pointer<float>(p[5]));
            break;
        case LFS_HIP_MCMC_FOLD_ERROR:
            require_bindings(2);
            hipLaunchKernelGGL(max_error_and_zero_densification_kernel,grid,block,0,queue,mcmc_pointer<float>(p[0]),mcmc_pointer<float>(p[1]),size_t(call->count));
            break;
        }
        hip_check(hipGetLastError(),"HIP application MCMC submission");
    });
}
}
