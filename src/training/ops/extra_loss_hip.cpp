/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/extra_loss_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_extra_api.h"
#include <algorithm>
#include <cmath>
#include <climits>
namespace lfs::training {
namespace {
using namespace gpu_ops;
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
void submit(hip::ApplicationCall& owner,const LfsHipExtraCall& call){
    auto fn=owner.module->symbol<decltype(&lfs_hip_application_extra)>("lfs_hip_application_extra");LfsHipError error{};
    if(int code=fn(&call,&error))throw hip::TrainingError(code,error);
}
LfsHipExtraCall descriptor(hip::ApplicationCall& owner,unsigned operation,size_t count,float weight){
    require(count<=INT_MAX && std::isfinite(weight),"Invalid HIP regularization count/weight");
    LfsHipExtraCall call{};call.struct_bytes=sizeof(call);call.operation=operation;call.device=owner.queue.device;
    call.queue=owner.queue.native_queue;call.count=count;call.weight=weight;return call;
}
void regularize(In raw,Out gradient,Out loss,Out temporary,Regularizer kind,float weight){
    require(raw.is_valid() && std::isfinite(weight) && (kind==Regularizer::Scale || kind==Regularizer::Opacity),"Invalid HIP regularization input/kind/weight");
    if(!raw.numel() || weight==0)return;
    require(loss.numel()==1 && temporary.numel()>=std::min<size_t>((raw.numel()+255)/256,1024) &&
            (!gradient.is_valid() || gradient.numel()==raw.numel()),"HIP regularization output/workspace mismatch");
    hip::ApplicationCall owner;auto call=descriptor(owner,kind==Regularizer::Scale?0:1,raw.numel(),weight);
    call.bindings[0]=owner.bind(raw,core::DataType::Float32);call.bindings[1]=owner.bind(gradient,core::DataType::Float32);
    call.bindings[2]=owner.bind(loss,core::DataType::Float32);call.bindings[3]=owner.bind(temporary,core::DataType::Float32);submit(owner,call);
}

}
const gpu_ops::ExtraLossOps& hip_extra_loss_ops(){static const gpu_ops::ExtraLossOps table{regularize};return table;}
}
