/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/photometric_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_photo_api.h"
#include <core/tensor_execution.hpp>
#include <array>
#include <climits>
#include <cmath>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
using core::Device;
using ExecutionScope=core::TensorExecutionTarget::Scope;
void require(bool condition,const char* message) { if(!condition) throw std::invalid_argument(message); }
Tensor image(In input,bool target) {
    require(input.is_valid() && core::gpu_backend_of(input)==core::GpuBackend::HIP && (input.ndim()==3 || input.ndim()==4),"HIP loss image needs native CHW/NCHW input");
    require(input.dtype()==DataType::Float32 || (target && input.dtype()==DataType::UInt8),"HIP loss image dtype unsupported");
    size_t count=1;
    for(auto extent:input.shape().dims()) {
        require(extent>0 && extent<=INT_MAX && count<=size_t(INT_MAX)/extent,"HIP loss image exceeds signed kernel index budget");
        count*=extent;
    }
    auto result=input.contiguous();
    return result.ndim()==3 ? result.unsqueeze(0) : result;
}
struct Images {
    Tensor corrected,raw,target,mask;
};
Images prepare(In corrected,In raw,In target,In mask,bool decoupled,bool masked) {
    Images result{image(corrected,false),{},image(target,true),{}};
    require(result.corrected.shape()==result.target.shape(),"HIP prediction/target dimensions differ");
    if(decoupled) {
        result.raw=image(raw,false);
        require(result.raw.shape()==result.corrected.shape(),"HIP raw/corrected image dimensions differ");
    }
    if(masked) {
        require(mask.is_valid() && core::gpu_backend_of(mask)==core::GpuBackend::HIP &&
                (mask.dtype()==DataType::Float32 || mask.dtype()==DataType::Bool || mask.dtype()==DataType::UInt8) &&
                (mask.ndim()==2 || (mask.ndim()==3 && mask.shape()[0]==1)),"HIP loss mask needs native HW/1HW input");
        result.mask=mask.contiguous();
        if(result.mask.ndim()==3) result.mask=result.mask.squeeze(0);
        require(result.mask.shape()[0]==result.corrected.shape()[2] && result.mask.shape()[1]==result.corrected.shape()[3],"HIP mask/image dimensions differ");
    }
    return result;
}
struct Workspace {
    std::array<Tensor,13> slots;
    core::TensorShape shape;
    unsigned path=UINT_MAX;
    void ensure(const core::TensorShape& dimensions,unsigned operation) {
        if(path==operation && shape==dimensions) return;
        Workspace next; next.shape=dimensions; next.path=operation;
        const size_t n=dimensions[0],c=dimensions[1],h=dimensions[2],w=dimensions[3],count=n*c*h*w;
        auto& s=next.slots;
        const bool decoupled=operation==3 || operation==5;
        if(operation>=1 && operation<=5) s[4]=Tensor::empty({count*(decoupled ? 4u : 3u)},Device::GPU,DataType::Float16);
        if(operation) {
            const bool channel_map=operation==1 || operation>=6;
            s[5]=Tensor::empty({n,channel_map ? c : 1,h,w},Device::GPU);
            if(operation!=6) s[6]=Tensor::empty(s[5].shape(),Device::GPU);
        }
        if(operation<=5) s[7]=Tensor::empty(dimensions,Device::GPU);
        if(decoupled) s[8]=Tensor::empty(dimensions,Device::GPU);
        if(operation!=6 && operation!=9) {
            s[9]=Tensor::empty({1},Device::GPU); s[10]=Tensor::empty({2048},Device::GPU);
        }
        if(operation==4 || operation==5) s[11]=Tensor::empty({1},Device::GPU);
        if(operation==1) s[12]=Tensor::empty(dimensions,Device::GPU);
        *this=std::move(next);
    }
    size_t bytes() const {
        size_t result=0;
        for(size_t i=4;i<slots.size();++i) if(slots[i].is_valid()) result+=slots[i].bytes();
        return result;
    }
};
struct HipPhotoState:BackendState { Workspace evaluate,metric,error; };
HipPhotoState& state(PhotoSaved& saved) {
    require(bool(saved.backend),"HIP photometric state has not been created");
    return static_cast<HipPhotoState&>(*saved.backend);
}
LfsHipPhotoCall call_for(hip::ApplicationCall& owner,Workspace& workspace,const Images& images,unsigned path,const PhotoParams& params) {
    LfsHipPhotoCall result{};
    result.struct_bytes=sizeof(result); result.path=path; result.device=owner.queue.device; result.queue=owner.queue.native_queue;
    result.n=int(images.corrected.shape()[0]); result.c=int(images.corrected.shape()[1]); result.h=int(images.corrected.shape()[2]); result.w=int(images.corrected.shape()[3]);
    result.byte_target=images.target.dtype()==DataType::UInt8; result.byte_mask=images.mask.is_valid() && images.mask.dtype()!=DataType::Float32;
    result.valid_padding=params.valid_padding; result.weight=params.ssim_weight; result.denominator=params.denominator;
    result.bindings[0]=owner.bind(images.corrected,DataType::Float32); result.bindings[1]=owner.bind(images.raw,DataType::Float32);
    result.bindings[2]=owner.bind(images.target,images.target.dtype()); result.bindings[3]=owner.bind(images.mask,images.mask.dtype());
    for(size_t i=4;i<workspace.slots.size();++i) result.bindings[i]=owner.bind(workspace.slots[i],i==4 ? DataType::Float16 : DataType::Float32);
    return result;
}
void submit(hip::ApplicationCall& owner,const LfsHipPhotoCall& call) {
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_photo)>("lfs_hip_application_photo");
    LfsHipError error{};
    if(int code=fn(&call,&error)) throw hip::TrainingError(code,error);
}
State create() { return std::make_unique<HipPhotoState>(); }
void evaluate(PhotoSaved& saved,In corrected,In raw,In target,In mask,const PhotoParams& params,Out loss,Out grad_corrected,Out grad_raw) {
    require(std::isfinite(params.ssim_weight) && params.ssim_weight>=0 && params.ssim_weight<=1 &&
            std::isfinite(params.denominator) && params.denominator>=0,"Invalid HIP photometric weight/normalizer");
    const auto path=static_cast<unsigned>(params.path); require(path<=5,"Invalid HIP photometric path");
    const ExecutionScope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    const auto inputs=prepare(corrected,raw,target,mask,path==3 || path==5,path==4 || path==5);
    auto& workspace=state(saved).evaluate; workspace.ensure(inputs.corrected.shape(),path);
    auto& s=workspace.slots;
    // Accumulating raw backward consumes a zeroed destination. Pure SSIM uses
    // the same padded map initialization as the reference trainer.
    if(s[8].is_valid()) s[8].zero_();
    if(path==1) {
        s[4].zero_(); s[5].zero_(); s[12].zero_(); s[7].zero_();
        auto cropped=s[12];
        if(params.valid_padding && inputs.corrected.shape()[2]>10) cropped=cropped.slice(2,5,inputs.corrected.shape()[2]-5);
        if(params.valid_padding && inputs.corrected.shape()[3]>10) cropped=cropped.slice(3,5,inputs.corrected.shape()[3]-5);
        cropped.fill_(-1.f/float(cropped.numel()));
    }
    hip::ApplicationCall owner; auto call=call_for(owner,workspace,inputs,path,params);
    submit(owner,call);
    saved.empty_mask=false;
    if(path==4 || path==5) {
        const float denominator=params.denominator>0 ? params.denominator : s[11].item();
        require(std::isfinite(denominator) && denominator>0,"HIP masked loss normalizer is invalid");
        saved.empty_mask=denominator<=1e-8f;
        call.normalization=1.f/denominator;
    }
    if(path!=0) { call.backward=1; submit(owner,call); }
    loss=path==1 ? Tensor::full({1},1.f,Device::GPU).sub(s[9]) : s[9];
    grad_corrected=corrected.ndim()==3 ? s[7].squeeze(0) : s[7];
    grad_raw=s[8].is_valid() ? (raw.ndim()==3 ? s[8].squeeze(0) : s[8]) : Tensor{};
    saved.ssim_map=s[5]; saved.cs_map=s[6];
}
Tensor metric(PhotoSaved& saved,In predicted,In target,bool maps,bool valid_padding) {
    const ExecutionScope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    const auto inputs=prepare(predicted,{},target,{},false,false);
    auto& workspace=state(saved).metric; workspace.ensure(inputs.corrected.shape(),8);
    hip::ApplicationCall owner; PhotoParams params; params.valid_padding=valid_padding;
    submit(owner,call_for(owner,workspace,inputs,8,params));
    if(maps) { saved.ssim_map=workspace.slots[5]; saved.cs_map=workspace.slots[6]; }
    return workspace.slots[9];
}
Tensor masked_metric_map(In predicted,In target,In mask) {
    const ExecutionScope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    const auto inputs=prepare(predicted,{},target,mask,false,true);
    Workspace workspace; workspace.ensure(inputs.corrected.shape(),6);
    hip::ApplicationCall owner; submit(owner,call_for(owner,workspace,inputs,6,{}));
    return workspace.slots[5];
}
void map_to_error(In map,Out error) {
    const ExecutionScope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    require(map.is_valid() && core::gpu_backend_of(map)==core::GpuBackend::HIP && map.is_contiguous() && map.dtype()==DataType::Float32 &&
            map.ndim()==4 && map.shape()[0]==1 && map.numel()>0 && map.numel()<=INT_MAX,"HIP error map requires native 1CHW map");
    const core::TensorShape shape{map.shape()[2],map.shape()[3]};
    if(!error.is_valid() || error.shape()!=shape || error.dtype()!=DataType::Float32 || core::gpu_backend_of(error)!=core::GpuBackend::HIP || !error.is_contiguous()) error=Tensor::empty(shape,Device::GPU);
    hip::ApplicationCall owner; LfsHipPhotoCall call{}; call.struct_bytes=sizeof(call); call.path=7;
    call.device=owner.queue.device; call.queue=owner.queue.native_queue;
    call.n=1; call.c=int(map.shape()[1]); call.h=int(map.shape()[2]); call.w=int(map.shape()[3]);
    call.bindings[5]=owner.bind(map,DataType::Float32); call.bindings[9]=owner.bind(error,DataType::Float32);
    submit(owner,call);
}
void error_map(PhotoSaved& saved,In predicted,In target,Out error,bool cs_only) {
    const ExecutionScope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    const auto inputs=prepare(predicted,{},target,{},false,false);
    require(inputs.corrected.shape()[0]==1,"HIP error map supports a single image");
    auto& workspace=state(saved).error; workspace.ensure(inputs.corrected.shape(),9);
    hip::ApplicationCall owner; submit(owner,call_for(owner,workspace,inputs,9,{}));
    map_to_error(workspace.slots[cs_only ? 6 : 5],error);
}
PhotoWorkspaceBytes workspace_bytes(const PhotoSaved& saved) {
    if(!saved.backend) return {};
    const auto& value=static_cast<const HipPhotoState&>(*saved.backend);
    return {value.evaluate.bytes()+value.metric.bytes(),value.evaluate.bytes()+value.metric.bytes(),value.error.bytes()};
}
void shrink(PhotoSaved&) { /* Exact current-shape allocations; no arena high-water storage. */ }
void reset(PhotoSaved& saved) {
    if(saved.backend) {
        auto& value=state(saved); value.evaluate={}; value.metric={}; value.error={};
    }
    saved.ssim_map={}; saved.cs_map={}; saved.empty_mask=false;
}
}
const gpu_ops::PhotometricOps& hip_photometric_ops() {
    static const gpu_ops::PhotometricOps table{create,evaluate,metric,masked_metric_map,error_map,map_to_error,workspace_bytes,shrink,reset};
    return table;
}
}
