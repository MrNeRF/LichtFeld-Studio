/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/mcmc_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_mcmc_api.h"
#include <core/tensor_execution.hpp>
#include <array>
#include <cmath>
#include <limits>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
void require(bool value,const char* message) { if(!value) throw std::invalid_argument(message); }
void size(In tensor,size_t elements) {
    require(tensor.is_valid() && tensor.numel()==elements,"HIP MCMC tensor dimensions mismatch");
    require(core::gpu_backend_of(tensor)==core::GpuBackend::HIP,"HIP MCMC received a tensor from another backend");
}
LfsHipMcmcCall call_for(hip::ApplicationCall& owner,unsigned operation,size_t count) {
    require(count<=size_t(std::numeric_limits<int>::max()),"HIP MCMC count exceeds int32");
    LfsHipMcmcCall call{}; call.struct_bytes=sizeof(call); call.operation=operation;
    call.device=owner.queue.device; call.queue=owner.queue.native_queue; call.count=count;
    return call;
}
void submit(hip::ApplicationCall& owner,const LfsHipMcmcCall& call) {
    LfsHipError error{};
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_mcmc)>("lfs_hip_application_mcmc");
    const int code=fn(&call,&error);
    if(code) throw hip::TrainingError(code,error);
}
void initialize(int maximum) {
    require(maximum>0 && maximum<=51,"HIP MCMC relocation ratio bound must be in 1..51");
    hip::ApplicationCall owner;
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_mcmc_initialize)>("lfs_hip_application_mcmc_initialize");
    LfsHipError error{};
    const int code=fn(owner.queue.device,owner.queue.native_queue,maximum,&error);
    if(code) throw hip::TrainingError(code,error);
}
void relocate(In opacity,In scales,In ratios,Out new_opacity,Out new_scales,float minimum) {
    const auto count=opacity.numel();
    size(scales,count*3); size(ratios,count); size(new_opacity,count); size(new_scales,count*3);
    if(!count) return;
    require(ratios.dtype()==DataType::Int32,"HIP MCMC relocation ratios require Int32");
    // This reduction validates external callers before the constant-table lookup.
    // It does not replace relocation arithmetic with host computation.
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    const auto bounds=ratios.to(DataType::Float32);
    require(bounds.min_scalar()>=1.f && bounds.max_scalar()<=51.f,"HIP MCMC relocation ratios outside 1..51");
    hip::ApplicationCall owner;
    auto call=call_for(owner,LFS_HIP_MCMC_RELOCATE,count); call.value=minimum;
    call.bindings[0]=owner.bind(opacity,DataType::Float32); call.bindings[1]=owner.bind(scales,DataType::Float32);
    call.bindings[2]=owner.bind(ratios,DataType::Int32); call.bindings[3]=owner.bind(new_opacity,DataType::Float32); call.bindings[4]=owner.bind(new_scales,DataType::Float32);
    submit(owner,call);
}
void noise(In opacity,In scales,In rotations,In frozen,Out means,uint64_t seed,float lr) {
    require(means.is_valid() && means.ndim()==2 && means.shape()[1]==3,"HIP MCMC means need N x 3");
    const auto count=means.shape()[0];
    size(opacity,count); size(scales,count*3); size(rotations,count*4);
    if(!count) return;
    hip::ApplicationCall owner;
    auto call=call_for(owner,LFS_HIP_MCMC_NOISE,count); call.seed=seed; call.value=lr;
    call.bindings[0]=owner.bind(opacity,DataType::Float32); call.bindings[1]=owner.bind(scales,DataType::Float32); call.bindings[2]=owner.bind(rotations,DataType::Float32);
    call.bindings[3]=owner.bind(means,DataType::Float32); call.bindings[4]=owner.bind(frozen,DataType::Bool); call.mask_count=frozen.is_valid() ? frozen.numel() : 0;
    submit(owner,call);
}
void copy_rows(In source,In destination,const McmcRows& rows) {
    const auto count=destination.numel(); size(source,count);
    require(rows.means.is_valid() && rows.means.ndim()==2 && rows.means.shape()[1]==3,"HIP MCMC row storage needs N x 3 means");
    const auto n=rows.means.shape()[0]; size(rows.sh0,n*3); size(rows.raw_scales,n*3); size(rows.raw_quats,n*4); size(rows.raw_opacity,n);
    if(!count) return;
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_MCMC_COPY,count); call.model_count=n;
    call.bindings[0]=owner.bind(source,DataType::Int64); call.bindings[1]=owner.bind(destination,DataType::Int64);
    call.bindings[2]=owner.bind(rows.means,DataType::Float32); call.bindings[3]=owner.bind(rows.sh0,DataType::Float32);
    call.bindings[4]=owner.bind(rows.raw_scales,DataType::Float32); call.bindings[5]=owner.bind(rows.raw_quats,DataType::Float32); call.bindings[6]=owner.bind(rows.raw_opacity,DataType::Float32);
    submit(owner,call);
}
void update_rows(In indices,In scales,In opacity,Out scales_out,Out opacity_out) {
    const auto count=indices.numel(), n=opacity_out.numel();
    size(scales,count*3); size(opacity,count); size(scales_out,n*3);
    if(!count) return;
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_MCMC_UPDATE,count); call.model_count=n;
    call.bindings[0]=owner.bind(indices,DataType::Int64); call.bindings[1]=owner.bind(scales,DataType::Float32); call.bindings[2]=owner.bind(opacity,DataType::Float32);
    call.bindings[3]=owner.bind(scales_out,DataType::Float32); call.bindings[4]=owner.bind(opacity_out,DataType::Float32);
    submit(owner,call);
}
void sample(In weights,In opacity,In scales,In alive,Out indices,Out sampled_opacity,Out sampled_scales,SampleDomain domain,uint64_t seed) {
    const auto count=indices.numel(), n=opacity.numel();
    size(weights,n); size(scales,n*3); size(sampled_opacity,count); size(sampled_scales,count*3);
    require(weights.dtype()==DataType::Float32 && (domain==SampleDomain::All || domain==SampleDomain::AliveIndices),"Invalid HIP MCMC sampling weights/domain");
    if(!count || !n || (domain==SampleDomain::AliveIndices && !alive.numel())) return;
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    Tensor probabilities=domain==SampleDomain::AliveIndices ? weights.index_select(0,alive) : weights;
    auto cdf=probabilities.cumsum(0);
    hip::ApplicationCall owner;
    auto call=call_for(owner,domain==SampleDomain::AliveIndices ? LFS_HIP_MCMC_SAMPLE_ALIVE : LFS_HIP_MCMC_SAMPLE_ALL,count);
    call.model_count=n; call.domain_count=probabilities.numel(); call.seed=seed;
    call.bindings[0]=owner.bind(opacity,DataType::Float32); call.bindings[1]=owner.bind(scales,DataType::Float32);
    unsigned slot=2;
    if(domain==SampleDomain::AliveIndices) call.bindings[slot++]=owner.bind(alive,DataType::Int64);
    call.bindings[slot++]=owner.bind(cdf,DataType::Float32); call.bindings[slot++]=owner.bind(indices,DataType::Int64);
    call.bindings[slot++]=owner.bind(sampled_opacity,DataType::Float32); call.bindings[slot]=owner.bind(sampled_scales,DataType::Float32);
    submit(owner,call);
}
void fold_error(Out maximum,Out densification) {
    const auto count=maximum.numel(); size(densification,count*2);
    if(!count) return;
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_MCMC_FOLD_ERROR,count);
    call.bindings[0]=owner.bind(maximum,DataType::Float32); call.bindings[1]=owner.bind(densification,DataType::Float32);
    submit(owner,call);
}
}
const gpu_ops::McmcOps& hip_mcmc_ops() {
    static const gpu_ops::McmcOps table{initialize,relocate,noise,copy_rows,update_rows,sample,fold_error};
    return table;
}
}
