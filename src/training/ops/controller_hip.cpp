/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/controller_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_controller_api.h"
#include <climits>
#include <cmath>
namespace lfs::training {
namespace {
using namespace gpu_ops;
static void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
static void submit(hip::ApplicationCall& owner,LfsHipControllerCall& call) {
    call.struct_bytes=sizeof(call);call.device=owner.queue.device;call.queue=owner.queue.native_queue;
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_controller)>("lfs_hip_application_controller");
    LfsHipError error{};if(int code=fn(&call,&error))throw hip::TrainingError(code,error);
}
void prepare(In features,Out input,float prior) {
    require(input.numel()>=1601&&(!features.is_valid()||features.numel()>=1600)&&std::isfinite(prior),"Invalid HIP controller input shape or prior");
    hip::ApplicationCall owner;LfsHipControllerCall p{};
    p.input=owner.bind(input,core::DataType::Float32);
    p.features=owner.bind(features,core::DataType::Float32);
    p.write_prior=!features.is_valid()||prior!=1.f;p.prior=prior;
    submit(owner,p);
    if(!features.is_valid())core::internal::backend_ops(core::GpuBackend::HIP).synchronize_stream({});
}
void backward(In grad,In activation,In weight,Out wg,Out bg,Out gi) {
    const auto m=grad.numel(),n=activation.numel();
    require(m&&n&&m<=INT_MAX&&n<=INT_MAX&&m<=size_t(INT_MAX)/n&&weight.numel()==m*n&&wg.numel()==m*n&&bg.numel()==m&&(!gi.is_valid()||gi.numel()==n),"Invalid HIP controller backward shape");
    hip::ApplicationCall owner;LfsHipControllerCall p{};p.operation=1;p.m=uint32_t(m);p.n=uint32_t(n);
    auto bind=[&](const core::Tensor& t){return owner.bind(t,core::DataType::Float32);};
    p.grad=bind(grad);p.activation=bind(activation);p.weight=bind(weight);p.weight_gradient=bind(wg);p.bias_gradient=bind(bg);p.grad_input=bind(gi);
    require(p.weight_gradient!=p.grad&&p.weight_gradient!=p.activation&&p.weight_gradient!=p.weight&&p.bias_gradient!=p.grad&&p.bias_gradient!=p.activation&&p.bias_gradient!=p.weight&&(!p.grad_input||(p.grad_input!=p.grad&&p.grad_input!=p.activation&&p.grad_input!=p.weight)),"HIP controller output aliases an input");
    submit(owner,p);
}
}
const gpu_ops::ControllerOps& hip_controller_ops(){static const gpu_ops::ControllerOps ops{prepare,backward};return ops;}
}
