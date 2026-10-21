/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/sh_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_sh_api.h"
#include <core/sh_layout.hpp>
#include <core/tensor_execution.hpp>
#include <climits>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
using core::Device;
using ExecutionScope=core::TensorExecutionTarget::Scope;
void require(bool condition,const char* message) { if(!condition) throw std::invalid_argument(message); }
void extent(In tensor,size_t bytes) { require(tensor.is_valid() && tensor.bytes()>=bytes,"HIP SH storage is too small"); }
void rest(uint32_t value) { require(value>0 && value<=15,"HIP SH coefficient count must be in 1..15"); }
size_t swizzled(size_t n,uint32_t coefficients) {
    require(n<=size_t(INT_MAX)/48,"HIP SH rows exceed kernel index budget");
    return core::sh_swizzled_float_count(n,coefficients);
}
size_t codes(size_t n,uint32_t coefficients) { require(n<=size_t(INT_MAX)/48,"HIP SH codes exceed kernel index budget"); return core::sh_swizzled_padded_n(n)*coefficients*3*2; }
size_t bounds_bytes(size_t n) { return ((n+255)/256)*8; }
LfsHipShCall call_for(hip::ApplicationCall& owner,unsigned operation,size_t n,size_t count=0) {
    require(n<=INT_MAX && count<=INT_MAX,"HIP SH dimensions exceed int32");
    LfsHipShCall result{}; result.struct_bytes=sizeof(result); result.operation=operation; result.device=owner.queue.device;
    result.queue=owner.queue.native_queue; result.primitives=n; result.count=count;
    return result;
}
uint64_t raw(hip::ApplicationCall& owner,In tensor) { return owner.bind(tensor,tensor.dtype()); }
void submit(hip::ApplicationCall& owner,LfsHipShCall& call) {
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_sh)>("lfs_hip_application_sh");
    LfsHipError error{};
    if(int code=fn(&call,&error)) throw hip::TrainingError(code,error);
}
struct Runs:BackendState { Tensor flags,compact,workspace; size_t count=0; };
State create_run_scratch() { return std::make_unique<Runs>(); }
void encode_q16(In input,Out output,Out bounds,size_t n,uint32_t coefficients,size_t code_offset,size_t bounds_offset) {
    if(!n || !coefficients) return; rest(coefficients);
    require(bounds_offset%2==0,"HIP SH bounds chunk offset must address float2");
    const auto code_bytes=codes(n,coefficients),bound_bytes=bounds_bytes(n);
    require(code_offset<=(SIZE_MAX-code_bytes)/2 && bounds_offset<=(SIZE_MAX-bound_bytes)/4,"HIP SH chunk offset overflow");
    extent(input,swizzled(n,coefficients)*4); extent(output,code_offset*2+code_bytes); extent(bounds,bounds_offset*4+bound_bytes);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_ENCODE,n);
    call.source_rest=coefficients; call.code_offset=code_offset; call.bounds_offset=bounds_offset;
    call.bindings[0]=owner.bind(input,DataType::Float32); call.bindings[1]=raw(owner,output); call.bindings[2]=owner.bind(bounds,DataType::Float32);
    submit(owner,call);
}
void decode_q16(In input,In bounds,Out output,size_t n,uint32_t coefficients) {
    if(!n || !coefficients) return; rest(coefficients);
    extent(input,codes(n,coefficients)); extent(bounds,bounds_bytes(n)); extent(output,swizzled(n,coefficients)*4);
    if(input.has_exportable_provenance() && bounds.has_exportable_provenance())
        require(input.exportable_bound_generation()==bounds.exportable_bound_generation(),"HIP SH codes/bounds generations differ");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_DECODE,n); call.source_rest=coefficients;
    call.bindings[0]=raw(owner,input); call.bindings[1]=owner.bind(bounds,DataType::Float32); call.bindings[2]=owner.bind(output,DataType::Float32);
    submit(owner,call);
}
void block_ids(In indices,Out output) {
    const auto count=indices.numel(); if(!count) return; extent(output,count*4);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_BLOCK_IDS,0,count);
    call.bindings[0]=owner.bind(indices,DataType::Int64); call.bindings[1]=owner.bind(output,DataType::Float32);
    submit(owner,call);
}
void block_runs(BackendState& state,In keys,Out unique,Out offsets,Out run_count) {
    const auto count=keys.numel(); extent(run_count,4); extent(unique,count*4); extent(offsets,count*4);
    const ExecutionScope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    auto& scratch=static_cast<Runs&>(state);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_BLOCK_RUNS,0,count);
    call.bindings[0]=owner.bind(keys,DataType::Float32); call.bindings[1]=owner.bind(unique,DataType::Int32); call.bindings[2]=owner.bind(offsets,DataType::Int32); call.bindings[4]=owner.bind(run_count,DataType::Int32);
    if(count && scratch.count!=count) {
        call.query=1; submit(owner,call); call.query=0;
        Runs replacement;
        replacement.flags=Tensor::empty({count},Device::GPU,DataType::Int32);
        replacement.compact=Tensor::empty({count},Device::GPU,DataType::Int32);
        if(call.workspace_bytes) replacement.workspace=Tensor::empty({size_t(call.workspace_bytes)},Device::GPU,DataType::UInt8);
        replacement.count=count;
        scratch=std::move(replacement);
    }
    call.bindings[5]=owner.bind(scratch.flags,DataType::Int32); call.bindings[6]=owner.bind(scratch.compact,DataType::Int32); call.bindings[7]=owner.bind(scratch.workspace,DataType::UInt8);
    call.workspace_bytes=scratch.workspace.is_valid() ? scratch.workspace.bytes() : 0;
    submit(owner,call);
}
void reencode_touched(Out output,Out bounds,In canonical,In destinations,In unique,In offsets,In run_count,In order,const Q16TouchParams& params) {
    if(!params.sorted_count || !params.primitives || !params.rest) return;
    rest(params.rest); require(params.decode_source_rows<=params.primitives,"HIP SH decode source exceeds live rows");
    extent(output,codes(params.primitives,params.rest)); extent(bounds,bounds_bytes(params.primitives)); extent(canonical,params.sorted_count*params.rest*3*4);
    extent(destinations,params.sorted_count*8); extent(unique,params.sorted_count*4); extent(offsets,params.sorted_count*4); extent(run_count,4);
    if(order.is_valid()) extent(order,params.sorted_count*8);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_TOUCHED,params.primitives,params.sorted_count);
    call.source_rows=params.decode_source_rows; call.source_rest=params.rest;
    call.bindings[0]=raw(owner,output); call.bindings[1]=owner.bind(bounds,DataType::Float32); call.bindings[2]=owner.bind(canonical,DataType::Float32);
    call.bindings[3]=owner.bind(destinations,DataType::Int64); call.bindings[4]=owner.bind(unique,DataType::Int32); call.bindings[5]=owner.bind(offsets,DataType::Int32);
    call.bindings[6]=owner.bind(run_count,DataType::Int32); call.bindings[7]=owner.bind(order,DataType::Int64);
    submit(owner,call);
}
void decode_range(In values,In bounds,Out canonical,const ShRangeParams& params) {
    if(!params.float_count) return; rest(params.layout_rest); rest(params.destination_rest);
    require(params.destination_rest<=params.layout_rest && params.primitives>0 && params.primitives<=size_t(INT_MAX)/48,"HIP SH range layout exceeds source");
    const auto total=uint64_t(params.primitives)*params.destination_rest*3;
    require(params.canonical_float_offset<=total && params.float_count<=total-params.canonical_float_offset && params.float_count<=INT_MAX,"HIP SH range exceeds canonical rows");
    const auto storage=static_cast<unsigned>(params.storage); require(storage<=2,"Invalid HIP SH storage codec");
    extent(values,storage==2 ? codes(params.primitives,params.layout_rest) : swizzled(params.primitives,params.layout_rest)*(storage==1 ? 2 : 4));
    if(storage==2) extent(bounds,bounds_bytes(params.primitives)); extent(canonical,size_t(params.float_count)*4);
    require(canonical.dtype()==DataType::Float32 || canonical.dtype()==DataType::UInt8,
            "HIP SH range destination must be Float32 or a byte scratch buffer");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_RANGE,params.primitives);
    call.source_rest=params.layout_rest; call.destination_rest=params.destination_rest; call.storage=storage;
    call.float_offset=params.canonical_float_offset; call.float_count=params.float_count;
    call.bindings[0]=raw(owner,values); call.bindings[1]=owner.bind(bounds,DataType::Float32); call.bindings[2]=raw(owner,canonical);
    require(call.bindings[0]%(storage==0?4:2)==0 && call.bindings[2]%4==0,"HIP SH range buffers are misaligned");
    submit(owner,call);
}
void zero_rows(Out values,In indices,uint32_t coefficients) {
    if(!indices.numel() || !coefficients) return; rest(coefficients);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_ZERO,0,indices.numel()); call.source_rest=coefficients;
    call.bindings[0]=owner.bind(values,DataType::Float32); call.bindings[1]=owner.bind(indices,DataType::Int32);
    submit(owner,call);
}
void gather_swizzled(In input,In indices,Out output,const ShRowsParams& params) {
    if(!params.count || !params.source_rest) return; rest(params.source_rest);
    require(params.source_rest==params.destination_rest && params.destination_offset<=UINT32_MAX-params.count,"HIP SH gather layout/offset mismatch");
    require(input.dtype()==DataType::Float32 || input.dtype()==DataType::UInt8,"HIP SH gather supports Float32/UInt8");
    const bool bytes=input.dtype()==DataType::UInt8;
    require(output.dtype()==input.dtype() && (!bytes || indices.dtype()==DataType::Int32),"HIP SH gather dtype mismatch");
    extent(indices,params.count*(indices.dtype()==DataType::Int64 ? 8 : 4));
    extent(input,swizzled(params.source_rows,params.source_rest)*(bytes ? 1 : 4));
    extent(output,swizzled(params.destination_offset+params.count,params.source_rest)*(bytes ? 1 : 4));
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_GATHER,0,params.count);
    call.source_rest=params.source_rest; call.index64=indices.dtype()==DataType::Int64; call.byte_elements=bytes; call.destination_offset=params.destination_offset;
    require(call.index64 || indices.dtype()==DataType::Int32,"HIP SH gather requires Int32/Int64 indices");
    call.bindings[0]=raw(owner,input); call.bindings[1]=raw(owner,indices); call.bindings[2]=raw(owner,output);
    submit(owner,call);
}
void gather_canonical(In input,In indices,Out output,const ShRowsParams& params) {
    if(!params.count || !params.destination_rest) return; rest(params.source_rest); rest(params.destination_rest);
    require(params.destination_rest<=params.source_rest,"HIP SH canonical gather exceeds source layout");
    extent(input,swizzled(params.source_rows,params.source_rest)*4); extent(indices,params.count*8); extent(output,params.count*params.destination_rest*3*4);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_GATHER_CANONICAL,0,params.count);
    call.source_rest=params.source_rest; call.destination_rest=params.destination_rest;
    call.bindings[0]=owner.bind(input,DataType::Float32); call.bindings[1]=owner.bind(indices,DataType::Int64); call.bindings[2]=owner.bind(output,DataType::Float32);
    submit(owner,call);
}
void append_canonical(In input,Out output,const ShRowsParams& params) {
    if(!params.count || !params.destination_rest) return; rest(params.source_rest); rest(params.destination_rest);
    require(params.source_rest<=params.destination_rest && params.destination_offset<=UINT32_MAX-params.count,"HIP SH append layout/offset mismatch");
    extent(input,params.count*params.source_rest*3*4); extent(output,swizzled(params.destination_offset+params.count,params.destination_rest)*4);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_APPEND,0,params.count);
    call.source_rest=params.source_rest; call.destination_rest=params.destination_rest; call.destination_offset=params.destination_offset;
    call.bindings[0]=owner.bind(input,DataType::Float32); call.bindings[1]=owner.bind(output,DataType::Float32);
    submit(owner,call);
}
void scatter_canonical(In input,In indices,Out output,const ShRowsParams& params) {
    if(!params.count || !params.destination_rest) return; rest(params.source_rest); rest(params.destination_rest);
    require(params.source_rest<=params.destination_rest,"HIP SH scatter layout mismatch");
    extent(input,params.count*params.source_rest*3*4); extent(indices,params.count*4); extent(output,swizzled(params.source_rows,params.destination_rest)*4);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_SCATTER,0,params.count);
    call.source_rest=params.source_rest; call.destination_rest=params.destination_rest;
    call.bindings[0]=owner.bind(input,DataType::Float32); call.bindings[1]=owner.bind(indices,DataType::Int32); call.bindings[2]=owner.bind(output,DataType::Float32);
    submit(owner,call);
}
void fill_bytes(Out storage,size_t bytes,uint8_t value) {
    if(!bytes) return; extent(storage,bytes);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_SH_FILL,0,bytes); call.byte_elements=value; call.bindings[0]=raw(owner,storage);
    submit(owner,call);
}
Tensor concatenate_into_arena(std::span<const Tensor> parts,char* data,core::TensorShape shape,DataType dtype,core::TensorExecutionTarget target) {
    require(target.backend()==core::GpuBackend::HIP && target.is_default_queue(),"HIP SH arena needs native default queue");
    const ExecutionScope scope(target);
    auto result=Tensor::from_blob(data,shape,Device::GPU,dtype,target);
    size_t bytes=0;
    for(const auto& part:parts) {
        require(part.is_valid() && part.is_contiguous() && core::gpu_backend_of(part)==core::GpuBackend::HIP && part.dtype()==dtype &&
                part.bytes()<=result.bytes()-bytes,"HIP SH arena parts exceed destination or have incompatible dtype/backend");
        bytes+=part.bytes();
    }
    require(bytes==result.bytes(),"HIP SH arena concatenation byte count mismatch");
    // Copy directly into the validated borrowed arena; no duplicate joined
    // model/scratch allocation is required.
    auto& ops=core::internal::backend_ops(core::GpuBackend::HIP);
    auto destination=core::internal::storage_ref(result); size_t offset=0;
    for(const auto& part:parts) {
        (void)const_cast<Tensor&>(part).data_ptr(); core::pin_operands({&part,&result});
        auto dst=destination; dst.byte_offset+=offset;
        ops.copy_device_to_device({core::internal::storage_ref(part),dst,part.bytes(),false,{},"hip.sh.arena_concat"});
        offset+=part.bytes();
    }
    return result;
}
}
const gpu_ops::ShOps& hip_sh_ops() {
    static const gpu_ops::ShOps table{create_run_scratch,encode_q16,decode_q16,block_ids,block_runs,reencode_touched,decode_range,
        zero_rows,gather_swizzled,gather_canonical,append_canonical,scatter_canonical,fill_bytes,concatenate_into_arena};
    return table;
}
}
