/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_sh_api.h"
#include "application_sh_generated.hpp"
extern "C" int lfs_hip_application_sh(LfsHipShCall* call,LfsHipError* error) {
    return boundary(nullptr,error,[&] {
        require(call && call->struct_bytes==sizeof(*call) && call->operation<=LFS_HIP_SH_FILL && call->queue && call->queue<=UINTPTR_MAX &&
                call->count<=uint64_t(INT_MAX) && call->primitives<=uint64_t(INT_MAX) && call->source_rest<=15 && call->destination_rest<=15,
                "Invalid HIP SH descriptor");
        DeviceScope device(call->device);
        const auto stream=reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(call->queue));
        const auto* p=call->bindings;
        auto f=[&](unsigned slot){return mcmc_pointer<float>(p[slot]);};
        auto u16=[&](unsigned slot){return mcmc_pointer<uint16_t>(p[slot]);};
        auto i32=[&](unsigned slot){return mcmc_pointer<int>(p[slot]);};
        const unsigned src_slots=lfs::core::sh_float4_slots_for_rest(call->source_rest);
        const unsigned dst_slots=lfs::core::sh_float4_slots_for_rest(call->destination_rest);
        const unsigned cells=call->source_rest*3;
        const unsigned n=unsigned(call->primitives),count=unsigned(call->count);
        const dim3 grid(blocks(call->count)),block(256);
        namespace layout=lfs_hip_application_sh_layout;
        namespace quant=lfs_hip_application_sh_quant;
        if(call->operation==LFS_HIP_SH_BLOCK_RUNS) {
            if(!count) { hip_check(hipMemsetAsync(i32(4),0,sizeof(int),stream),"HIP SH empty run count"); return; }
            size_t bytes=0;
            hip_check(hipcub::DeviceScan::ExclusiveSum(nullptr,bytes,i32(5),i32(6),count,stream),"HIP SH run scan workspace query");
            if(call->query) { call->workspace_bytes=bytes; return; }
            require(call->workspace_bytes>=bytes && (!bytes || p[7]),"HIP SH run workspace is too small");
            hipLaunchKernelGGL(quant::mark_sorted_block_run_starts_kernel,grid,block,0,stream,f(0),i32(5),int(count));
            size_t available=call->workspace_bytes;
            hip_check(hipcub::DeviceScan::ExclusiveSum(mcmc_pointer<void>(p[7]),available,i32(5),i32(6),count,stream),"HIP SH run scan");
            hipLaunchKernelGGL(quant::compact_sorted_block_runs_kernel,grid,block,0,stream,f(0),i32(5),i32(6),i32(1),i32(2),i32(4),int(count));
        } else if(call->operation==LFS_HIP_SH_ENCODE) {
            if(!n || !cells) return;
            hipLaunchKernelGGL(quant::encode_float4_to_u16_block_kernel,dim3((size_t(n)+255)/256),block,0,stream,
                f(0),u16(1)+call->code_offset,mcmc_pointer<float2>(p[2])+call->bounds_offset/2,n,src_slots,cells);
        } else if(call->operation==LFS_HIP_SH_DECODE) {
            if(!n || !cells) return;
            hipLaunchKernelGGL(quant::decode_u16_to_float4_kernel,dim3(blocks(n)),block,0,stream,u16(0),mcmc_pointer<float2>(p[1]),f(2),n,src_slots,cells);
        } else if(call->operation==LFS_HIP_SH_BLOCK_IDS) {
            if(!count) return;
            hipLaunchKernelGGL(quant::fill_quant_block_ids_f32_kernel,grid,block,0,stream,mcmc_pointer<int64_t>(p[0]),f(1),count);
        } else if(call->operation==LFS_HIP_SH_TOUCHED) {
            if(!count || !n || !cells) return;
            hipLaunchKernelGGL(quant::reencode_touched_q16_block_kernel,dim3(count),block,0,stream,u16(0),mcmc_pointer<float2>(p[1]),f(2),mcmc_pointer<int64_t>(p[7]),
                mcmc_pointer<int64_t>(p[3]),i32(4),i32(5),i32(6),count,n,unsigned(call->source_rows),cells);
        } else if(call->operation==LFS_HIP_SH_RANGE) {
            if(!call->float_count) return;
            const dim3 range_grid(blocks(call->float_count));
            if(call->storage==2) {
                hipLaunchKernelGGL(quant::decode_u16_range_to_canonical_kernel,range_grid,block,0,stream,u16(0),mcmc_pointer<float2>(p[1]),f(2),
                    call->float_offset,call->float_count,call->destination_rest*3,cells);
            } else if(call->storage==1) {
                hipLaunchKernelGGL(quant::decode_f16_range_to_canonical_kernel,range_grid,block,0,stream,mcmc_pointer<__half>(p[0]),f(2),
                    call->float_offset,call->float_count,call->destination_rest*3,src_slots);
            } else {
                hipLaunchKernelGGL(layout::undo_reorder_sh_range_kernel,range_grid,block,0,stream,mcmc_pointer<float4>(p[0]),f(2),
                    call->float_offset,call->float_count,call->destination_rest*3,src_slots);
            }
        } else if(call->operation==LFS_HIP_SH_ZERO) {
            if(!count || !src_slots) return;
            hipLaunchKernelGGL((layout::zero_at_indices_kernel<int>),grid,block,0,stream,mcmc_pointer<float4>(p[0]),i32(1),count,src_slots);
        } else if(call->operation==LFS_HIP_SH_GATHER) {
            if(!count || !src_slots) return;
            if(call->byte_elements) {
                hipLaunchKernelGGL((layout::gather_self_kernel<int,uchar4>),grid,block,0,stream,mcmc_pointer<uchar4>(p[0]),mcmc_pointer<uchar4>(p[2]),i32(1),count,unsigned(call->destination_offset),src_slots);
            } else if(call->index64) {
                hipLaunchKernelGGL((layout::gather_self_kernel<int64_t>),grid,block,0,stream,mcmc_pointer<float4>(p[0]),mcmc_pointer<float4>(p[2]),mcmc_pointer<int64_t>(p[1]),count,unsigned(call->destination_offset),src_slots);
            } else {
                hipLaunchKernelGGL((layout::gather_self_kernel<int>),grid,block,0,stream,mcmc_pointer<float4>(p[0]),mcmc_pointer<float4>(p[2]),i32(1),count,unsigned(call->destination_offset),src_slots);
            }
        } else if(call->operation==LFS_HIP_SH_GATHER_CANONICAL) {
            if(!count || !call->destination_rest) return;
            hipLaunchKernelGGL((layout::gather_to_linear_kernel<int64_t>),grid,block,0,stream,mcmc_pointer<float4>(p[0]),mcmc_pointer<int64_t>(p[1]),f(2),count,call->destination_rest,src_slots);
        } else if(call->operation==LFS_HIP_SH_APPEND) {
            if(!count || !dst_slots) return;
            hipLaunchKernelGGL(layout::gather_from_linear_kernel,grid,block,0,stream,mcmc_pointer<float4>(p[1]),unsigned(call->destination_offset),f(0),count,call->source_rest,dst_slots);
        } else if(call->operation==LFS_HIP_SH_SCATTER) {
            if(!count || !dst_slots) return;
            hipLaunchKernelGGL(layout::scatter_linear_kernel,grid,block,0,stream,mcmc_pointer<float4>(p[2]),i32(1),f(0),count,call->source_rest,dst_slots);
        } else if(call->operation==LFS_HIP_SH_FILL) {
            hip_check(hipMemsetAsync(mcmc_pointer<void>(p[0]),int(call->byte_elements),call->count,stream),"HIP SH byte fill");
        }
        hip_check(hipGetLastError(),"HIP application SH submission");
    });
}
