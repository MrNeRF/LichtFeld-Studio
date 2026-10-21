/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/training_image_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_image_api.h"
#include <core/tensor_execution.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
using core::Device;
void require(bool condition,const char* message) { if(!condition) throw std::invalid_argument(message); }
void size(In tensor,size_t count) { require(tensor.is_valid() && tensor.numel()==count,"HIP training-image dimensions mismatch"); }
size_t image(In tensor,size_t channels=0) {
    require(tensor.is_valid() && tensor.ndim()==3 && tensor.shape()[0]>0 && tensor.shape()[1]>0 && tensor.shape()[2]>0 &&
            tensor.numel()<=INT_MAX && (!channels || tensor.shape()[0]==channels),"HIP training-image needs bounded CHW storage");
    return tensor.shape()[1]*tensor.shape()[2];
}
LfsHipImageCall call_for(hip::ApplicationCall& owner,unsigned operation,size_t count) {
    require(count<=INT_MAX,"HIP training-image count exceeds int32");
    LfsHipImageCall result{}; result.struct_bytes=sizeof(result); result.operation=operation; result.device=owner.queue.device;
    result.queue=owner.queue.native_queue; result.count=count;
    return result;
}
void submit(hip::ApplicationCall& owner,const LfsHipImageCall& call) {
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_image)>("lfs_hip_application_image");
    LfsHipError error{};
    if(int code=fn(&call,&error)) throw hip::TrainingError(code,error);
}
void heatmap(In loss,Out latest,Out ema,int slot,float alpha) {
    require(std::isfinite(alpha) && alpha>=0 && alpha<=1,"Invalid HIP camera-loss EMA alpha");
    size(loss,1); size(ema,latest.numel());
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_IMAGE_HEATMAP,latest.numel()); call.slot=slot; call.value=alpha;
    call.bindings[0]=owner.bind(loss,DataType::Float32); call.bindings[1]=owner.bind(latest,DataType::Float32); call.bindings[2]=owner.bind(ema,DataType::Float32);
    submit(owner,call);
}
void roi(In view,In camera_position,In camera_rays,In surface_depth,In surface_alpha,Out weights,const RoiParams& params) {
    require(params.image.h>0 && params.image.w>0 && size_t(params.image.h)*params.image.w<=INT_MAX &&
            std::isfinite(params.intrinsics.fx) && params.intrinsics.fx>0 && std::isfinite(params.intrinsics.fy) && params.intrinsics.fy>0 &&
            std::isfinite(params.intrinsics.cx) && std::isfinite(params.intrinsics.cy) && std::isfinite(params.outside_weight) && params.outside_weight>=0 && params.outside_weight<=1,"Invalid HIP ROI camera/weights");
    size(view,16); size(camera_position,3); size(weights,size_t(params.image.h)*params.image.w);
    const size_t area=weights.numel();
    const bool rays=camera_rays.is_valid() && camera_rays.numel();
    const bool surface=surface_depth.is_valid() && surface_depth.numel() && surface_alpha.is_valid() && surface_alpha.numel();
    if(rays) size(camera_rays,area*3);
    if(surface) { size(surface_depth,area); size(surface_alpha,area); }
    for(size_t i=0;i<3;++i) require(std::isfinite(params.minimum[i]) && std::isfinite(params.maximum[i]) && params.minimum[i]<=params.maximum[i],"Invalid HIP ROI box");
    for(float value:params.world_to_cropbox) require(std::isfinite(value),"Invalid HIP ROI transform");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_IMAGE_ROI,weights.numel());
    call.h=params.image.h; call.w=params.image.w; call.fx=params.intrinsics.fx; call.fy=params.intrinsics.fy; call.cx=params.intrinsics.cx; call.cy=params.intrinsics.cy;
    call.value=params.outside_weight; call.inverse=params.inverse;
    for(unsigned row=0;row<3;++row) for(unsigned column=0;column<4;++column) call.affine[row*4+column]=params.world_to_cropbox[column*4+row];
    std::copy(params.minimum.begin(),params.minimum.end(),call.minimum); std::copy(params.maximum.begin(),params.maximum.end(),call.maximum);
    call.bindings[0]=owner.bind(view,DataType::Float32); call.bindings[1]=owner.bind(camera_position,DataType::Float32); call.bindings[2]=owner.bind(weights,DataType::Float32);
    if(rays) call.bindings[3]=owner.bind(camera_rays,DataType::Float32);
    if(surface) { call.bindings[4]=owner.bind(surface_depth,DataType::Float32); call.bindings[5]=owner.bind(surface_alpha,DataType::Float32); }
    submit(owner,call);
}
void resize_background(In input,Out output) {
    image(input); const auto count=image(output); require(input.shape()[0]==output.shape()[0],"HIP background channel count mismatch");
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_IMAGE_RESIZE,count);
    call.c=int(input.shape()[0]); call.source_h=int(input.shape()[1]); call.source_w=int(input.shape()[2]); call.h=int(output.shape()[1]); call.w=int(output.shape()[2]);
    call.bindings[0]=owner.bind(input,DataType::Float32); call.bindings[1]=owner.bind(output,DataType::Float32);
    submit(owner,call);
}
void random_background(Out output,uint64_t seed) {
    const auto count=image(output,3);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_IMAGE_RANDOM,count); call.seed=seed;
    call.bindings[0]=owner.bind(output,DataType::Float32); submit(owner,call);
}
void canny(In input,In weight,Out edges) {
    const auto count=image(input,3); size(edges,count);
    require(input.dtype()==DataType::Float32 || input.dtype()==DataType::UInt8,"HIP Canny input must be Float32/UInt8");
    if(weight.is_valid()) size(weight,count);
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_IMAGE_CANNY,count);
    call.h=int(input.shape()[1]); call.w=int(input.shape()[2]); call.byte_input=input.dtype()==DataType::UInt8;
    call.bindings[0]=owner.bind(input,input.dtype()); call.bindings[1]=owner.bind(weight,DataType::Float32); call.bindings[2]=owner.bind(edges,DataType::Float32);
    submit(owner,call);
}
void normalize_scalar(Out values,In scalar,float skip) {
    require(std::isfinite(skip),"Invalid HIP scalar normalization threshold"); size(scalar,1);
    if(!values.numel()) return;
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_IMAGE_NORMALIZE,values.numel()); call.value=skip;
    call.bindings[0]=owner.bind(values,DataType::Float32); call.bindings[1]=owner.bind(scalar,DataType::Float32);
    submit(owner,call);
}
Tensor quantize_to_grid(In input,float levels) {
    require(input.is_valid() && core::gpu_backend_of(input)==core::GpuBackend::HIP && input.dtype()==DataType::Float32 && std::isfinite(levels) && levels>0,"Invalid HIP image quantization input/grid");
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    auto contiguous=input.contiguous(); auto result=Tensor::empty(input.shape(),Device::GPU);
    if(!input.numel()) return result;
    hip::ApplicationCall owner; auto call=call_for(owner,LFS_HIP_IMAGE_QUANTIZE,input.numel()); call.value=levels;
    call.bindings[0]=owner.bind(contiguous,DataType::Float32); call.bindings[1]=owner.bind(result,DataType::Float32);
    submit(owner,call); return result;
}
Tensor quantize_to_8bit_grid(In input) { return quantize_to_grid(input,255.f); }
}
const gpu_ops::TrainingImageOps& hip_training_image_ops() {
    static const gpu_ops::TrainingImageOps table{
        .heatmap=heatmap,.roi=roi,.resize_background=resize_background,.random_background=random_background,
        .canny=canny,.normalize_scalar=normalize_scalar,.quantize_to_8bit_grid=quantize_to_8bit_grid,.quantize_to_grid=quantize_to_grid,
    };
    return table;
}
}
