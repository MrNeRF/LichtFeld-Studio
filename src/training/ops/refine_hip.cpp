/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/refine_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_refine_api.h"
#include <core/tensor_execution.hpp>
#include <limits>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
void require(bool condition,const char* message) { if(!condition) throw std::invalid_argument(message); }
void size(In value,size_t count) {
    require(value.is_valid() && value.numel()==count,"HIP refinement tensor dimensions mismatch");
}
LfsHipRefineCall call_for(hip::ApplicationCall& owner,unsigned operation,size_t count) {
    require(count<=size_t(std::numeric_limits<int>::max()),"HIP refinement count exceeds int32");
    LfsHipRefineCall result{};
    result.struct_bytes=sizeof(result); result.operation=operation; result.device=owner.queue.device;
    result.queue=owner.queue.native_queue; result.count=count;
    return result;
}
void submit(hip::ApplicationCall& owner,const LfsHipRefineCall& call) {
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_refine)>("lfs_hip_application_refine");
    LfsHipError error{};
    if(int code=fn(&call,&error)) throw hip::TrainingError(code,error);
}
void bind_rows(hip::ApplicationCall& owner,LfsHipRefineCall& call,unsigned slot,In means,In rotations,In scales,In sh0,In opacity,size_t n) {
    size(means,n*3); size(rotations,n*4); size(scales,n*3); size(sh0,n*3); size(opacity,n);
    for(const auto* tensor:{&means,&rotations,&scales,&sh0,&opacity})
        call.bindings[slot++]=owner.bind(*tensor,DataType::Float32);
}
void split(const RefineOutputs& parents,const RefineOutputs& children,In indices) {
    const auto count=indices.numel(); if(!count) return;
    require(parents.means.ndim()==2 && parents.means.shape()[1]==3,"HIP split requires N x 3 parent means");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_SPLIT,count);
    bind_rows(owner,call,0,parents.means,parents.rotations,parents.scales,parents.sh0,parents.opacity,parents.means.shape()[0]);
    bind_rows(owner,call,5,children.means,children.rotations,children.scales,children.sh0,children.opacity,count);
    call.bindings[10]=owner.bind(indices,DataType::Int64);
    submit(owner,call);
}
void fill_slots(In indices,const RefineInputs& source,const RefineOutputs& destination,Out free_mask) {
    const auto count=indices.numel(); if(!count) return;
    require(destination.means.ndim()==2 && destination.means.shape()[1]==3,"HIP slot destination needs N x 3 means");
    const auto n=destination.means.shape()[0];
    // IGS keeps the free mask at reserved capacity while model views expose
    // only live rows. The kernel indexes the mask within the live row count.
    require(free_mask.is_valid() && free_mask.numel()>=n,"HIP slot free mask is shorter than the model");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_FILL,count); call.rows=n;
    call.bindings[0]=owner.bind(indices,DataType::Int64);
    bind_rows(owner,call,1,source.means,source.rotations,source.scales,source.sh0,source.opacity,count);
    bind_rows(owner,call,6,destination.means,destination.rotations,destination.scales,destination.sh0,destination.opacity,n);
    call.bindings[11]=owner.bind(free_mask,DataType::Bool);
    submit(owner,call);
}
void counts(In bool0,In bool1,In float0,In float1,Out counts4) {
    size(counts4,4);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_COUNTS,1);
    unsigned slot=0;
    for(const auto* tensor:{&bool0,&bool1,&float0,&float1}) {
        call.bindings[slot]=owner.bind(*tensor,slot<2 ? DataType::Bool : DataType::Float32);
        call.lengths[slot]=tensor->is_valid() ? tensor->numel() : 0;
        ++slot;
    }
    call.bindings[4]=owner.bind(counts4,DataType::Int64);
    submit(owner,call);
}
void normalize(Out values) {
    if(!values.numel()) return;
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    auto state=Tensor::empty({4},core::Device::GPU,DataType::UInt32);
    auto histogram=Tensor::empty({2048},core::Device::GPU,DataType::UInt32);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_NORMALIZE,values.numel());
    call.bindings[0]=owner.bind(values,DataType::Float32);
    call.bindings[1]=owner.bind(state,DataType::UInt32); call.bindings[2]=owner.bind(histogram,DataType::UInt32);
    submit(owner,call);
}
void clip_scales(Out scales,In shares,In frozen,float limit) {
    require(scales.ndim()==2 && scales.shape()[1]==3,"HIP clip scales need N x 3");
    const auto n=scales.shape()[0]; size(shares,n); if(!n) return;
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_CLIP,n); call.value=limit;
    call.bindings[0]=owner.bind(scales,DataType::Float32); call.bindings[1]=owner.bind(shares,DataType::Float32); call.bindings[2]=owner.bind(frozen,DataType::Bool);
    call.lengths[0]=frozen.is_valid() ? frozen.numel() : 0;
    submit(owner,call);
}
void oversize_scores(In error,In shares,In frozen,Out scores,float limit) {
    const auto n=error.numel(); size(shares,n); size(scores,n); if(!n) return;
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_SCORES,n); call.value=limit;
    call.bindings[0]=owner.bind(error,DataType::Float32); call.bindings[1]=owner.bind(shares,DataType::Float32);
    call.bindings[2]=owner.bind(frozen,DataType::Bool); call.bindings[3]=owner.bind(scores,DataType::Float32);
    call.lengths[0]=frozen.is_valid() ? frozen.numel() : 0;
    submit(owner,call);
}
void dead_mask(In opacity,In rotations,Out mask,float minimum) {
    const auto n=opacity.numel(); size(rotations,n*4); size(mask,n); if(!n) return;
    require(mask.dtype()==DataType::Bool || mask.dtype()==DataType::UInt8,"HIP dead mask requires byte output");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_DEAD,n); call.value=minimum;
    call.bindings[0]=owner.bind(opacity,DataType::Float32); call.bindings[1]=owner.bind(rotations,DataType::Float32); call.bindings[2]=owner.bind(mask,mask.dtype());
    submit(owner,call);
}
void rotation_mask(In rotations,Out mask) {
    require(rotations.ndim()==2 && rotations.shape()[1]==4,"HIP rotation mask needs N x 4");
    const auto n=rotations.shape()[0]; size(mask,n); if(!n) return;
    require(mask.dtype()==DataType::Bool || mask.dtype()==DataType::UInt8,"HIP rotation mask requires byte output");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_REFINE_ROTATION,n);
    call.bindings[0]=owner.bind(rotations,DataType::Float32); call.bindings[1]=owner.bind(mask,mask.dtype());
    submit(owner,call);
}
}
const gpu_ops::RefineOps& hip_refine_ops() {
    static const gpu_ops::RefineOps table{split,fill_slots,counts,normalize,clip_scales,oversize_scores,dead_mask,rotation_mask};
    return table;
}
}
