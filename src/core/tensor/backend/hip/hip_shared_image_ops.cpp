/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_backend_ops.hpp"
#include "../../internal/tensor_impl.hpp"
#include <core/shared_image_ops.hpp>
#include <core/tensor_execution.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lfs::core {
namespace {
using namespace gpu_ops;
using namespace internal;
void require(bool value,const char* message) { if(!value) throw std::invalid_argument(message); }
void tensor(const Tensor& value,DataType dtype,size_t bytes) {
    require(value.is_valid() && value.device()==Device::GPU && gpu_backend_of(value)==GpuBackend::HIP &&
            value.is_contiguous() && value.dtype()==dtype && value.bytes()==bytes,"HIP shared image requires matching contiguous native Tensor storage");
}
LfsHipSharedImage descriptor(unsigned op,size_t h,size_t w,size_t c) {
    require(h && w && c && c<=4 && h<=INT32_MAX/w && h*w<=INT32_MAX/c,"HIP shared image extent exceeds indexing");
    LfsHipSharedImage result{};result.struct_bytes=sizeof(result);result.operation=op;
    result.height=h;result.width=w;result.channels=c;return result;
}
void submit(const LfsHipSharedImage& call,const Tensor* source,Tensor& destination,Tensor* auxiliary=nullptr) {
    const TensorExecutionTarget::Scope scope(TensorExecutionTarget::default_queue(GpuBackend::HIP));
    if(source)pin_operands({source});
    auto& backend=static_cast<HipBackendOps&>(backend_ops(GpuBackend::HIP));
    backend.shared_image(call,source?storage_ref(*source):StorageRef{},storage_ref(destination),auxiliary?storage_ref(*auxiliary):StorageRef{},{});
}
void sentinel_fill(Tensor& value,uint32_t seed) {
    tensor(value,DataType::UInt8,value.numel());
    if(!value.numel())return;
    auto call=descriptor(16,1,value.numel(),1);call.seed=seed;submit(call,nullptr,value);
}
void sentinel_check(const Tensor& value,Tensor& unchanged,uint32_t seed) {
    tensor(value,DataType::UInt8,value.numel());tensor(unchanged,DataType::UInt32,4);
    if(!value.numel())return;
    auto call=descriptor(17,1,value.numel(),1);call.seed=seed;submit(call,&value,unchanged);
}
void convert(const Tensor& source,Tensor& destination,ImageConversion conversion,size_t h,size_t w,size_t c,const NormalPriorTransform& transform) {
    auto call=descriptor(static_cast<unsigned>(conversion),h,w,c);const size_t total=h*w*c;
    DataType input=DataType::UInt8,output=DataType::Float32;size_t input_bytes=total,output_bytes=total*4;
    switch(conversion) {
    case ImageConversion::U8HWCToF32CHW:case ImageConversion::U8HWToF32HW:case ImageConversion::MaskU8HWToF32HW:
    case ImageConversion::NormalPriorU8:case ImageConversion::U8HWCToF32CHWExactMaskAlpha:break;
    case ImageConversion::U16HWCToF32CHW:case ImageConversion::U16HWCToF32HWC:case ImageConversion::NormalPriorU16:
    case ImageConversion::U16HWCToF32CHWExactMaskAlpha:
        // Shared CPU image uploads carry raw uint16 samples in Float16
        // storage (there is no UInt16 tensor dtype). This is a byte-preserving
        // container, not a numeric half-to-float conversion.
        input=source.dtype()==DataType::Float16?DataType::Float16:DataType::UInt8;
        input_bytes=total*2;break;
    case ImageConversion::F32HWCToU16HWC:input=DataType::Float32;input_bytes=total*4;output=DataType::UInt8;output_bytes=total*2;break;
    case ImageConversion::NormalCHWToJ2KHWC:case ImageConversion::J2KHWCToNormalCHW:case ImageConversion::NormalPriorF32:
        input=DataType::Float32;input_bytes=total*4;break;
    case ImageConversion::U8HWCToU8CHW:output=DataType::UInt8;output_bytes=total;break;
    case ImageConversion::U16HWCToU8CHW:
        input=source.dtype()==DataType::Float16?DataType::Float16:DataType::UInt8;
        input_bytes=total*2;output=DataType::UInt8;output_bytes=total;break;
    case ImageConversion::F32CHWToU8CHW:input=DataType::Float32;input_bytes=total*4;output=DataType::UInt8;output_bytes=total;break;
    default:throw std::invalid_argument("Unknown HIP image conversion");
    }
    tensor(source,input,input_bytes);tensor(destination,output,output_bytes);
    call.srgb=transform.srgb;call.flip_yz=transform.flip_yz;call.world_to_camera=transform.world_to_camera;
    std::copy(std::begin(transform.w2c),std::end(transform.w2c),call.w2c);submit(call,&source,destination);
}
void rgba_split(const Tensor& rgba,Tensor& rgb,Tensor& alpha) {
    require(rgba.ndim()==3 && rgba.size(2)==4,"HIP RGBA split needs HWC input");
    const size_t h=rgba.size(0),w=rgba.size(1);auto call=descriptor(rgb.dtype()==DataType::UInt8?19:18,h,w,4);
    tensor(rgba,DataType::UInt8,h*w*4);require(rgb.dtype()==DataType::UInt8 || rgb.dtype()==DataType::Float32,"HIP RGB output dtype mismatch");
    tensor(rgb,rgb.dtype(),h*w*3*(rgb.dtype()==DataType::UInt8?1:4));tensor(alpha,DataType::Float32,h*w*4);submit(call,&rgba,rgb,&alpha);
}
void mask(Tensor& value,MaskTransform transform,float threshold) {
    require(value.ndim()==2 && std::isfinite(threshold),"HIP mask transform needs finite threshold and HW image");
    unsigned op=0;switch(transform){case MaskTransform::Invert:op=20;break;case MaskTransform::Threshold:op=21;break;case MaskTransform::QuantizeByteGrid:op=22;break;default:throw std::invalid_argument("Unknown HIP mask transform");}
    auto call=descriptor(op,value.size(0),value.size(1),1);tensor(value,DataType::Float32,value.numel()*4);call.threshold=threshold;submit(call,nullptr,value);
}
Tensor resize(const Tensor& source,int height,int width,Resample mode,int kernel_size) {
    const TensorExecutionTarget::Scope scope(TensorExecutionTarget::default_queue(GpuBackend::HIP));
    auto& backend=static_cast<HipBackendOps&>(backend_ops(GpuBackend::HIP));
    return backend.shared_image_resize(source,height,width,static_cast<unsigned>(mode),kernel_size,{});
}
Tensor undistort(const Tensor& source,const UndistortParams& params,bool is_mask) {
    const TensorExecutionTarget::Scope scope(TensorExecutionTarget::default_queue(GpuBackend::HIP));
    auto& backend=static_cast<HipBackendOps&>(backend_ops(GpuBackend::HIP));
    return backend.image_undistort(source,params,is_mask,{});
}
const SharedImageOps table{sentinel_fill,sentinel_check,convert,rgba_split,mask,resize,undistort};
}
const gpu_ops::SharedImageOps* hip_shared_image_ops() { return &table; }
}
