/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/lpips_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_lpips_api.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
uint32_t narrow(size_t value){require(value<=INT_MAX,"HIP LPIPS dimension exceeds int32");return uint32_t(value);}
uint32_t pair(int lo,int hi){require(lo>=0 && hi>=0 && lo<=65535 && hi<=65535,"HIP LPIPS packed dimension exceeds uint16");return uint32_t(lo)|(uint32_t(hi)<<16);}
void operand(In tensor,DataType type){require(tensor.is_valid() && tensor.numel() && tensor.is_contiguous() && tensor.dtype()==type && core::gpu_backend_of(tensor)==core::GpuBackend::HIP,"HIP LPIPS needs contiguous nonempty native tensors of the declared type");}
void shape(In tensor,std::initializer_list<size_t> dimensions){require(tensor.shape()==core::TensorShape(dimensions),"HIP LPIPS tensor shape mismatch");}
void separate(uint64_t source,size_t source_bytes,uint64_t destination,size_t destination_bytes){
    require(!source || !destination || source+source_bytes<=destination || destination+destination_bytes<=source,"HIP LPIPS input and output storage overlap");
}
LfsHipLpipsCall descriptor(hip::ApplicationCall& owner,unsigned operation,size_t count){
    LfsHipLpipsCall call{};call.struct_bytes=sizeof(call);call.operation=operation;call.device=owner.queue.device;call.queue=owner.queue.native_queue;call.dispatch_count=narrow(count);return call;
}
void submit(hip::ApplicationCall& owner,const LfsHipLpipsCall& call){
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_lpips)>("lfs_hip_application_lpips");LfsHipError error{};
    if(int code=fn(&call,&error))throw hip::TrainingError(code,error);
}
void weight_taps(In weight,Out taps){
    operand(weight,DataType::Float16);operand(taps,DataType::Float16);
    require(weight.ndim()==4 && weight.size(2)==3 && weight.size(3)==3,"HIP LPIPS taps require 3x3 weights");
    shape(taps,{9,weight.shape()[0],weight.shape()[1]});
    hip::ApplicationCall owner;auto call=descriptor(owner,0,weight.numel());auto& p=call.parameters;
    p.a=owner.bind(weight,DataType::Float16);p.d=owner.bind(taps,DataType::Float16);p.count=narrow(weight.numel());
    p.channels=narrow(weight.size(1));p.out_channels=narrow(weight.size(0));
    separate(p.a,weight.bytes(),p.d,taps.bytes());submit(owner,call);
}
void rgb_conv(In input,In weight,In bias,Out output,const RGBConvParams& params){
    operand(input,DataType::Float32);operand(weight,DataType::Float16);operand(output,DataType::Float16);
    require(input.ndim()==4 && input.size(1)==3 && weight.ndim()==4 && weight.size(1)==3 && weight.size(2)==3 && weight.size(3)==3,"HIP LPIPS RGB convolution requires NCHW RGB and 3x3 weights");
    shape(output,{input.shape()[0],weight.shape()[0],input.shape()[2],input.shape()[3]});
    if(bias.is_valid()){operand(bias,DataType::Float16);require(bias.numel()==weight.shape()[0],"HIP LPIPS RGB bias channel mismatch");}
    for(unsigned i=0;i<3;++i)require(std::isfinite(params.shift[i]) && std::isfinite(params.scale[i]) && params.scale[i]!=0,"HIP LPIPS RGB normalization is invalid");
    hip::ApplicationCall owner;auto call=descriptor(owner,1,output.numel());auto& p=call.parameters;
    p.a=owner.bind(input,DataType::Float32);p.b=owner.bind(weight,DataType::Float16);p.c=owner.bind(bias,DataType::Float16);p.d=owner.bind(output,DataType::Float16);
    p.count=narrow(output.numel());p.channels=3;p.height=narrow(input.size(2));p.width=narrow(input.size(3));p.out_channels=narrow(weight.size(0));p.out_height=p.height;p.out_width=p.width;
    p.stride=pair(1,1);p.padding=pair(1,1);p.dilation=pair(1,1);p.flags=1u|(params.official_scaling?1u<<16:0u);
    p.shift0=params.shift[0];p.shift1=params.shift[1];p.shift2=params.shift[2];p.scale0=params.scale[0];p.scale1=params.scale[1];p.scale2=params.scale[2];
    separate(p.a,input.bytes(),p.d,output.bytes());separate(p.b,weight.bytes(),p.d,output.bytes());if(bias.is_valid())separate(p.c,bias.bytes(),p.d,output.bytes());
    submit(owner,call);
}
void convolution(In input,In weight,In taps,In bias,Out output,Out scratch,const ConvParams& params){
    require(input.dtype()==DataType::Float16 || input.dtype()==DataType::Float32,"HIP LPIPS convolution requires Float16 or Float32");
    operand(input,input.dtype());operand(weight,input.dtype());operand(output,input.dtype());
    require(input.ndim()==4 && weight.ndim()==4 && weight.size(1)==input.size(1) && weight.size(2)==3 && weight.size(3)==3 && params.groups==1 && params.output_pad_h==0 && params.output_pad_w==0,"HIP LPIPS convolution requires NCHW, 3x3 weights and one group");
    require(params.stride_h>0 && params.stride_w>0 && params.dilation_h>0 && params.dilation_w>0 && params.pad_h>=0 && params.pad_w>=0,"HIP LPIPS convolution geometry is invalid");
    const int64_t height_extent=int64_t(input.size(2))+2ll*params.pad_h-2ll*params.dilation_h-1;
    const int64_t width_extent=int64_t(input.size(3))+2ll*params.pad_w-2ll*params.dilation_w-1;
    require(height_extent>=0 && width_extent>=0,"HIP LPIPS convolution kernel exceeds padded input");
    const int64_t height=height_extent/params.stride_h+1;
    const int64_t width=width_extent/params.stride_w+1;
    require(height>0 && width>0 && height<=INT_MAX && width<=INT_MAX,"HIP LPIPS convolution output dimensions are invalid");
    shape(output,{input.shape()[0],weight.shape()[0],size_t(height),size_t(width)});
    require(unsigned(params.activation)<=4 && (params.pad_mode==core::nn::ConvPadMode::Zeros || params.pad_mode==core::nn::ConvPadMode::Replicate),"HIP LPIPS convolution activation or padding mode is invalid");
    if(bias.is_valid()){operand(bias,input.dtype());require(bias.numel()==weight.size(0),"HIP LPIPS bias channel mismatch");}
    if(taps.is_valid()){operand(taps,DataType::Float16);shape(taps,{9,weight.shape()[0],weight.shape()[1]});}
    // The native tiled kernel reuses OIHW weights directly. Optional prepacked
    // taps and caller scratch are not required, and no replacement is allocated.
    (void)scratch;
    hip::ApplicationCall owner;auto call=descriptor(owner,4,output.numel());auto& p=call.parameters;
    p.a=owner.bind(input,input.dtype());p.b=owner.bind(weight,weight.dtype());p.c=owner.bind(bias,input.dtype());p.d=owner.bind(output,output.dtype());
    p.count=narrow(output.numel());p.channels=narrow(input.size(1));p.height=narrow(input.size(2));p.width=narrow(input.size(3));p.out_channels=narrow(weight.size(0));p.out_height=uint32_t(height);p.out_width=uint32_t(width);
    p.stride=pair(params.stride_h,params.stride_w);p.padding=pair(params.pad_h,params.pad_w);p.dilation=pair(params.dilation_h,params.dilation_w);
    p.flags=uint32_t(params.activation)|(params.pad_mode==core::nn::ConvPadMode::Replicate?1u<<8:0u)|(input.dtype()==DataType::Float16?1u<<9:0u);
    call.dispatch_count=narrow(((size_t(p.out_height)*p.out_width+63)/64)*((size_t(p.out_channels)+15)/16)*input.shape()[0]*256);
    separate(p.a,input.bytes(),p.d,output.bytes());separate(p.b,weight.bytes(),p.d,output.bytes());if(bias.is_valid())separate(p.c,bias.bytes(),p.d,output.bytes());
    submit(owner,call);
}
void pool_reduce(In x,In y,In weight,Out score,Out pooled_x,Out pooled_y,const PoolReduceParams& params){
    operand(x,DataType::Float16);operand(y,DataType::Float16);operand(weight,DataType::Float16);operand(score,DataType::Float32);
    require(x.ndim()==4 && y.shape()==x.shape() && x.size(1)%8==0 && weight.numel()==x.shape()[1] && score.numel()==1,"HIP LPIPS pooling shape/channel mismatch");
    require(std::isfinite(params.inverse_count) && params.inverse_count>=0,"HIP LPIPS pool normalization is invalid");
    if(auto error=pool_reduce_region_error(x,params);!error.empty())throw std::invalid_argument(error);
    require(pooled_x.is_valid()==pooled_y.is_valid(),"HIP LPIPS needs both pooling outputs or neither");
    const bool pool=pooled_x.is_valid();
    if(pool){
        operand(pooled_x,DataType::Float16);operand(pooled_y,DataType::Float16);
        shape(pooled_x,{x.shape()[0],x.shape()[1],x.shape()[2]/2,x.shape()[3]/2});shape(pooled_y,{x.shape()[0],x.shape()[1],x.shape()[2]/2,x.shape()[3]/2});
    }
    if(params.weights)operand(*params.weights,DataType::Float32);
    const size_t pixels=x.numel()/x.shape()[1],items=pool?std::max(pixels,pooled_x.numel()):pixels;
    hip::ApplicationCall owner;auto call=descriptor(owner,3,items);auto& p=call.parameters;
    p.a=owner.bind(x,DataType::Float16);p.b=owner.bind(y,DataType::Float16);p.c=owner.bind(weight,DataType::Float16);p.d=owner.bind(score,DataType::Float32);
    p.e=owner.bind(pooled_x,DataType::Float16);p.f=owner.bind(pooled_y,DataType::Float16);p.count=narrow(pixels);p.channels=narrow(x.size(1));p.height=narrow(x.size(2));p.width=narrow(x.size(3));
    p.stride=pair(params.y0,params.y1);p.padding=pair(params.x0,params.x1);p.inverse_count=params.inverse_count;
    if(params.weights)p.g=owner.bind(*params.weights,DataType::Float32);
    p.out_width=narrow(params.weights_width);p.out_height=narrow(params.weights_y0);p.out_channels=narrow(params.weights_x0);
    separate(p.a,x.bytes(),p.d,score.bytes());separate(p.b,y.bytes(),p.d,score.bytes());separate(p.c,weight.bytes(),p.d,score.bytes());
    if(pool){separate(p.a,x.bytes(),p.e,pooled_x.bytes());separate(p.b,y.bytes(),p.e,pooled_x.bytes());separate(p.a,x.bytes(),p.f,pooled_y.bytes());separate(p.b,y.bytes(),p.f,pooled_y.bytes());separate(p.e,pooled_x.bytes(),p.f,pooled_y.bytes());}
    submit(owner,call);
}
}
const gpu_ops::LpipsOps& hip_lpips_ops(){
    static const gpu_ops::LpipsOps table{{.weight_taps=weight_taps,.rgb_conv=rgb_conv,.convolution=convolution,.pool_reduce=pool_reduce,.prefer_independent_queue=false}};
    return table;
}
}
