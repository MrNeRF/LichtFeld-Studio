/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/morton_hip.hpp"
#include "lfs/training/ops/sh_hip.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "core/splat_data.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor_completion.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_morton_api.h"
#include <algorithm>
#include <climits>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;using core::Device;using core::TensorShape;using core::GpuBackend;
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
void target(core::TensorExecutionTarget value){require(value.backend()==GpuBackend::HIP && value.is_default_queue(),"HIP Morton requires its native default queue");}
void extent(In value,size_t bytes){require(value.is_valid() && value.bytes()>=bytes,"HIP Morton storage shorter than declared layout");}
void permutation_shape(In value,size_t n){require(value.is_valid() && value.dtype()==DataType::Int64 && value.numel()==n,"HIP Morton permutation requires N native Int64 indices");}
LfsHipMortonCall descriptor(hip::ApplicationCall& owner,unsigned operation,size_t n){
    require(n<=size_t(INT_MAX)/48,"HIP Morton layout exceeds kernel index budget");
    LfsHipMortonCall result{};result.struct_bytes=sizeof(result);result.operation=operation;result.device=owner.queue.device;result.queue=owner.queue.native_queue;result.count=n;return result;
}
uint64_t raw(hip::ApplicationCall& owner,In value){return owner.bind(value,value.dtype());}
void submit(hip::ApplicationCall& owner,const LfsHipMortonCall& call){
    auto fn=owner.module->symbol<decltype(&lfs_hip_application_morton)>("lfs_hip_application_morton");LfsHipError error{};
    if(int code=fn(&call,&error))throw hip::TrainingError(code,error);
}
Tensor permutation(In means){
    require(means.is_valid() && means.ndim()==2 && means.size(1)==3 && means.dtype()==DataType::Float32 && core::gpu_backend_of(means)==GpuBackend::HIP,"HIP Morton requires native Float32 N,3 means");
    if(!means.size(0))return {};
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(GpuBackend::HIP));
    auto dense=means.contiguous(),lo=dense.min(0),hi=dense.max(0);auto keys=Tensor::empty({means.size(0)},Device::GPU);
    hip::ApplicationCall owner;auto call=descriptor(owner,0,means.size(0));call.bindings[0]=owner.bind(dense,DataType::Float32);
    call.bindings[1]=owner.bind(lo,DataType::Float32);call.bindings[2]=owner.bind(hi,DataType::Float32);call.bindings[3]=owner.bind(keys,DataType::Float32);submit(owner,call);
    // Positive 30-bit integer keys are stored as their float bit patterns. The
    // existing native radix sort preserves their exact integer order and ties.
    return keys.sort(0,false).second;
}
size_t packed_bytes(const JointCodecParams& p){
    require(p.primitives>=0 && p.primitives<=INT_MAX/48 && p.attributes_or_slots>0 && p.attributes_or_slots<=48 &&
            (p.bits==8 || p.bits==16) && (p.layout==JointLayout::Rows || (p.layout==JointLayout::SwizzledSH && p.attributes_or_slots<=12)),"Invalid HIP Morton joint layout");
    const size_t rows=p.layout==JointLayout::Rows ? size_t(p.primitives) : core::sh_swizzled_padded_n(p.primitives);
    return rows*p.attributes_or_slots*(p.layout==JointLayout::Rows?1:4)*(p.bits==8?2:4);
}
LfsHipMortonCall joint(hip::ApplicationCall& owner,In packed,In bounds,In perm,Out next_bounds,const JointCodecParams& p,unsigned operation){
    extent(packed,packed_bytes(p));const size_t bound_bytes=(size_t(p.primitives)+255)/256*16;
    extent(bounds,bound_bytes);extent(next_bounds,bound_bytes);permutation_shape(perm,p.primitives);
    auto call=descriptor(owner,operation,p.primitives);call.width=p.attributes_or_slots;call.bits=p.bits;
    call.bindings[0]=owner.bind(packed,DataType::UInt8);call.bindings[1]=owner.bind(bounds,DataType::Float32);
    call.bindings[2]=owner.bind(perm,DataType::Int64);call.bindings[4]=owner.bind(next_bounds,DataType::Float32);return call;
}
void permute_joint(In packed,In bounds,In perm,Out next,Out next_bounds,const JointCodecParams& p){
    if(p.primitives==0)return;
    hip::ApplicationCall owner;auto call=joint(owner,packed,bounds,perm,next_bounds,p,p.layout==JointLayout::Rows?1:2);
    extent(next,packed_bytes(p));call.bindings[3]=owner.bind(next,DataType::UInt8);submit(owner,call);
}
void permute_joint_grouped(Out packed,In bounds,In perm,Out next_bounds,Out scratch,const JointCodecParams& p){
    require(p.layout==JointLayout::SwizzledSH,"HIP grouped Morton requires SH moments");if(p.primitives==0)return;
    hip::ApplicationCall owner;auto call=joint(owner,packed,bounds,perm,next_bounds,p,3);
    const size_t unit=core::sh_swizzled_padded_n(p.primitives)*4*(p.bits==8?2:4);
    extent(scratch,unit);call.scratch_bytes=scratch.bytes();call.bindings[5]=owner.bind(scratch,DataType::UInt8);submit(owner,call);
}
void copy_back(Out live,const void* source,size_t bytes,core::TensorExecutionTarget execution){
    target(execution);if(!bytes)return;extent(live,bytes);
    const auto width=core::dtype_size(live.dtype());require(bytes%width==0,"HIP Morton copy must contain complete elements");
    const core::TensorExecutionTarget::Scope scope(execution);
    auto borrowed=Tensor::from_blob(const_cast<void*>(source),{bytes/width},Device::GPU,live.dtype(),execution);
    live.flatten().slice(0,0,bytes/width).copy_(borrowed);
}
void permute_sh_q16(core::SplatData& splat,In perm,core::TensorExecutionTarget execution,const IdleArenaScratch&){
    target(execution);const auto n=size_t(splat.size());const auto rest=unsigned(splat.max_sh_coeffs_rest());
    if(!n || !rest)return;require(rest<=15 && splat.shN_value_quantized(),"HIP Morton Q16 requires quantized SH and valid rest");permutation_shape(perm,n);
    auto& live=splat.shN();auto& bounds=splat.shN_value_bounds();
    const size_t cells=rest*3,cell_bytes=core::sh_swizzled_padded_n(n)*2,bound_count=(n+255)/256*2;
    extent(live,cell_bytes*cells);extent(bounds,bound_count*4);
    const core::TensorExecutionTarget::Scope scope(execution);
    auto next_bounds=Tensor::empty({bound_count},Device::GPU);const size_t group=std::min(cells,std::max<size_t>(1,(64ull<<20)/cell_bytes));
    auto scratch=Tensor::empty({group*cell_bytes},Device::GPU,DataType::UInt8);
    hip::ApplicationCall owner;auto call=descriptor(owner,4,n);call.width=unsigned(cells);call.scratch_bytes=scratch.bytes();
    call.bindings[0]=raw(owner,live);call.bindings[1]=owner.bind(bounds,DataType::Float32);call.bindings[2]=owner.bind(perm,DataType::Int64);
    call.bindings[3]=owner.bind(next_bounds,DataType::Float32);call.bindings[4]=owner.bind(scratch,DataType::UInt8);submit(owner,call);
    bounds.flatten().slice(0,0,bound_count).copy_(next_bounds);
    core::TensorCompletion completion;completion.include(live);completion.include(bounds);completion.wait();
}
void permute_sh_fp32(core::SplatData& splat,In perm,core::TensorExecutionTarget execution,const IdleArenaScratch&){
    target(execution);const core::TensorExecutionTarget::Scope scope(execution);
    const bool expanded=sh_value::ensure_shN_fp32_for_mutation(splat);auto& live=splat.shN();
    const size_t n=splat.size();const auto rest=unsigned(splat.max_sh_coeffs_rest());
    if(!n || !rest || !live.is_valid() || live.dtype()!=DataType::Float32){if(expanded)(void)sh_value::commit_shN_after_mutation(splat);return;}
    permutation_shape(perm,n);const size_t logical=core::sh_swizzled_float_count(n,rest);extent(live,logical*4);
    auto gathered=Tensor::zeros({logical},Device::GPU);
    hip_sh_ops().gather_swizzled(live,perm,gathered,{n,n,0,rest,rest});
    live.flatten().slice(0,0,logical).copy_(gathered);if(expanded)(void)sh_value::commit_shN_after_mutation(splat);
}
void gather_gradient(In source,In perm,Out destination,uint32_t rest,core::TensorExecutionTarget execution){
    target(execution);const core::TensorExecutionTarget::Scope scope(execution);
    hip_sh_ops().gather_swizzled(source,perm,destination,{perm.numel(),perm.numel(),0,rest,rest});
}
}
const gpu_ops::MortonOps& hip_morton_ops(){static const gpu_ops::MortonOps table{permutation,permute_joint,permute_joint_grouped,permute_sh_q16,permute_sh_fp32,copy_back,gather_gradient};return table;}
}
