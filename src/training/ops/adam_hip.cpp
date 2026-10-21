/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/adam_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_adam_api.h"
#include <core/tensor_execution.hpp>
#include <climits>
#include <cmath>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
void require(bool condition,const char* message) { if(!condition) throw std::invalid_argument(message); }
int narrow(size_t count) { require(count<=INT_MAX,"HIP Adam tensor exceeds int32 indexing"); return int(count); }
void extent(In tensor,size_t bytes) { require(tensor.is_valid() && tensor.bytes()>=bytes,"HIP Adam tensor storage is too small"); }
LfsHipAdamCall common(hip::ApplicationCall& owner,unsigned operation,const AdamMasks& masks,const AdamHyper& hyper,const AdamModifiers& modifiers) {
    require(std::isfinite(hyper.beta1) && hyper.beta1>=0 && hyper.beta1<1 && std::isfinite(hyper.beta2) && hyper.beta2>=0 && hyper.beta2<1 &&
            std::isfinite(hyper.eps) && hyper.eps>0,"Invalid HIP Adam hyperparameters");
    require(std::isfinite(modifiers.frozen_lr_scale) && std::isfinite(modifiers.cropbox_lr_scale) &&
            std::isfinite(modifiers.mean_step_median_extent) && std::isfinite(modifiers.screen_share_limit) && std::isfinite(modifiers.screen_share_penalty),"Invalid HIP Adam modifiers");
    LfsHipAdamCall call{}; call.struct_bytes=sizeof(call); call.operation=operation;
    call.device=owner.queue.device; call.queue=owner.queue.native_queue;
    const Tensor* inputs[]{&masks.frozen,&masks.crop_damping,&masks.mean_step_scales,&masks.mean_step_far,&masks.screen_share};
    for(unsigned i=0;i<5;++i) {
        call.masks[i]=owner.bind(*inputs[i],i==2 || i==4 ? DataType::Float32 : DataType::Bool);
        call.lengths[i]=inputs[i]->is_valid() ? narrow(inputs[i]->numel()) : 0;
    }
    call.beta1=hyper.beta1; call.beta2=hyper.beta2; call.eps=hyper.eps;
    call.frozen_lr=modifiers.frozen_lr_scale; call.crop_lr=modifiers.cropbox_lr_scale;
    call.median_extent=modifiers.mean_step_median_extent; call.screen_limit=modifiers.screen_share_limit; call.screen_penalty=modifiers.screen_share_penalty;
    return call;
}
void submit(hip::ApplicationCall& owner,const LfsHipAdamCall& call) {
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_adam)>("lfs_hip_application_adam");
    LfsHipError error{};
    if(int code=fn(&call,&error)) throw hip::TrainingError(code,error);
}
void step_batch(std::span<const JointStep> steps,const AdamMasks& masks,const AdamHyper& hyper,const AdamModifiers& modifiers) {
    hip::ApplicationCall owner; auto call=common(owner,0,masks,hyper,modifiers);
    for(const auto& step:steps) {
        if(!step.parameter.is_valid()) continue;
        require(call.entries<6 && step.bits==16 && step.primitives>0 && step.attributes>0 && step.attributes<=16,"Invalid HIP contiguous Adam layout/batch");
        const size_t elements=size_t(step.primitives)*step.attributes;
        require(elements<=INT_MAX && std::isfinite(step.lr) && std::isfinite(step.bc1_rcp) && std::isfinite(step.bc2_sqrt_rcp),"Invalid HIP Adam step scalars/index budget");
        extent(step.parameter,elements*4); extent(step.gradient,elements*4); extent(step.packed,elements*4); extent(step.bounds,((size_t(step.primitives)+255)/256)*16);
        auto& entry=call.batch[call.entries++];
        entry.parameter=owner.bind(step.parameter,DataType::Float32); entry.gradient=owner.bind(step.gradient,DataType::Float32);
        entry.packed=owner.bind(step.packed,DataType::UInt8); entry.bounds=owner.bind(step.bounds,DataType::Float32);
        entry.primitives=step.primitives; entry.attributes=step.attributes; entry.mean_step=step.apply_mean_step; entry.screen_share=step.apply_screen_share;
        entry.lr=step.lr; entry.bc1_rcp=step.bc1_rcp; entry.bc2_sqrt_rcp=step.bc2_sqrt_rcp;
    }
    submit(owner,call);
}
void step_sh(Out parameter,Out packed,Out bounds,Out value_bounds,In gradient,const AdamMasks& masks,const AdamHyper& hyper,const AdamModifiers& modifiers,const ShStepParams& params) {
    if(params.primitives<=0 || params.layout_slots<=0) return;
    require((params.active_bases==4 || params.active_bases==9 || params.active_bases==16) &&
            params.layout_slots>=(3*(params.active_bases-1)+3)/4 && params.layout_slots<=12 &&
            (params.value_bits==0 || params.value_bits==16) && std::isfinite(params.step_size) && std::isfinite(params.bc2_sqrt_rcp),"Invalid HIP SH Adam layout/scalars");
    const size_t cells=((size_t(params.primitives)+31)/32)*32*params.layout_slots*4;
    require(cells<=INT_MAX,"HIP SH Adam exceeds int32 indexing");
    const bool quantized=value_bounds.is_valid();
    const size_t rows=((size_t(params.primitives)+31)/32)*32;
    if(quantized) require(params.value_bits==16 && params.value_cells>=3*(params.active_bases-1) && params.value_cells<=params.layout_slots*4 && params.value_cells<=45,"Invalid HIP SH quantization cells");
    extent(parameter,quantized ? rows*size_t(params.value_cells)*2 : cells*(params.value_bits==0 ? 4 : 2)); extent(gradient,cells*4);
    extent(packed,cells*2); extent(bounds,((size_t(params.primitives)+255)/256)*16);
    if(value_bounds.is_valid()) {
        extent(value_bounds,((size_t(params.primitives)+255)/256)*8);
    }
    hip::ApplicationCall owner; auto call=common(owner,1,masks,hyper,modifiers);
    call.primitives=params.primitives; call.active_bases=params.active_bases; call.layout_slots=params.layout_slots;
    call.gradient=owner.bind(gradient,DataType::Float32);
    auto& sh=call.sh;
    require(params.value_bits==16 || parameter.dtype()==DataType::Float32,"FP32 SH Adam requires Float32 values");
    sh.parameter=owner.bind(parameter,parameter.dtype()); sh.packed=owner.bind(packed,DataType::UInt8); sh.bounds=owner.bind(bounds,DataType::Float32);
    sh.value_bounds=owner.bind(value_bounds,DataType::Float32); sh.enabled=1; sh.joint_bits=8;
    sh.value_bits=params.value_bits; sh.value_cells=params.value_cells; sh.primitives=params.primitives;
    sh.frozen=call.masks[0]; sh.crop=call.masks[1]; sh.frozen_count=call.lengths[0]; sh.crop_count=call.lengths[1];
    sh.frozen_lr=modifiers.frozen_lr_scale; sh.crop_lr=modifiers.cropbox_lr_scale;
    sh.step_size=params.step_size; sh.bc2_sqrt_rcp=params.bc2_sqrt_rcp;
    submit(owner,call);
}
void encode_zero(Out packed,Out bounds,In indices,const JointCodecParams& params) {
    if(!indices.numel()) return;
    require(params.primitives>0 && params.attributes_or_slots>0 && (params.bits==8 || params.bits==16) &&
            (params.layout==JointLayout::Rows || params.layout==JointLayout::SwizzledSH),"Invalid HIP Adam reset layout");
    const size_t n=params.primitives,blocks=(n+255)/256;
    const size_t cells=params.layout==JointLayout::Rows ? n*params.attributes_or_slots : ((n+31)/32)*32*params.attributes_or_slots*4;
    require(cells<=INT_MAX,"HIP Adam reset exceeds int32 indexing");
    extent(packed,cells*(params.bits==16 ? 4 : 2)); extent(bounds,blocks*16);
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    auto flags=Tensor::zeros({n},core::Device::GPU,DataType::UInt8);
    auto touched=Tensor::zeros({(blocks+3)&~size_t(3)},core::Device::GPU,DataType::UInt8);
    Tensor absent; AdamMasks masks{absent,absent,absent,absent,absent};
    hip::ApplicationCall owner; auto call=common(owner,2,masks,{},{});
    call.count=narrow(indices.numel()); call.primitives=params.primitives; call.attributes=params.attributes_or_slots; call.bits=params.bits;
    call.layout=params.layout==JointLayout::SwizzledSH ? 1 : 0;
    call.indices=owner.bind(indices,DataType::Int64); call.flags=owner.bind(flags,DataType::UInt8); call.touched=owner.bind(touched,DataType::UInt8);
    call.sh.packed=owner.bind(packed,DataType::UInt8); call.sh.bounds=owner.bind(bounds,DataType::Float32);
    submit(owner,call);
}
}
const gpu_ops::AdamOps& hip_adam_ops() {
    static const gpu_ops::AdamOps table{step_batch,step_sh,encode_zero};
    return table;
}
}
