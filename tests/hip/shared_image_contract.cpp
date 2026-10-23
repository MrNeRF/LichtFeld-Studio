/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_completion.hpp>
#include <core/tensor_execution.hpp>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <core/shared_image_ops.hpp>
#include <core/tensor_image.hpp>
#include <stdexcept>
using namespace lfs;
using namespace core;
namespace lfs::training {
unsigned other_tables=0;
const TrainingOps& cuda_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::CUDA}; return t; }
const TrainingOps& metal_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Metal}; return t; }
const TrainingOps& vulkan_training_ops_table() { ++other_tables; static const TrainingOps t{.backend=GpuBackend::Vulkan}; return t; }
}
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
void compare(const Tensor& a,const Tensor& b,float tolerance,const char* message) {
    require(a.shape()==b.shape(),"HIP image shape differs from CPU");auto x=a.cpu().to_vector(),y=b.cpu().to_vector();
    float maximum=0;for(size_t i=0;i<x.size();++i){if(!std::isfinite(x[i]) || !std::isfinite(y[i])){require((std::isnan(x[i]) && std::isnan(y[i])) || x[i]==y[i],message);continue;}maximum=std::max(maximum,std::abs(x[i]-y[i]));}
    if(maximum>tolerance)std::fprintf(stderr,"Image maximum error %.9g > %.9g: %s\n",maximum,tolerance,message);require(maximum<=tolerance,message);
}
int main() try {
    const auto configured=configured_gpu_backend();const auto& table=training::training_ops(GpuBackend::HIP);
    const auto* ops=shared_image_ops(GpuBackend::HIP);
    require(ops && table.shared_image==ops && !table.fast && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"Shared-image discovery initialized GPU or another table");
    require(ops->sentinel_fill && ops->sentinel_check && ops->convert && ops->rgba_split && ops->mask && ops->resize && ops->undistort,"Incomplete HIP shared image table");
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        constexpr size_t h=7,w=9,n=h*w;
        for(unsigned code=0;code<16;++code) for(bool typed_u16:{false,true}) {
            const unsigned c=(code==11 || code==13)?1:(code==14 || code==15)?4:3;const size_t count=n*c;
            const bool in16=code==1 || code==3 || code==7 || code==9 || code==15;
            if(typed_u16 && !in16)continue;
            const bool in32=code==2 || code==4 || code==5 || code==10 || code==12;
            const bool out16=code==2,out8=code==8 || code==9 || code==10;
            std::vector<uint8_t> bytes(count);std::vector<uint16_t> shorts(count);std::vector<float> floats(count);
            for(size_t i=0;i<count;++i){bytes[i]=uint8_t(i*71);shorts[i]=uint16_t(i*577);floats[i]=float(int((i*73)%313)-20)/271;}
            auto source=in32?Tensor::from_vector(floats,{count},Device::GPU):in16?Tensor::from_blob(shorts.data(),{count*2},Device::CPU,DataType::UInt8).to(Device::GPU):Tensor::from_blob(bytes.data(),{count},Device::CPU,DataType::UInt8).to(Device::GPU);
            if(typed_u16)source=Tensor::from_blob(shorts.data(),{count},Device::CPU,DataType::Float16).to(Device::GPU);
            auto destination=Tensor::empty({count*(out16?2:1)},Device::GPU,out16 || out8?DataType::UInt8:DataType::Float32);
            ops->convert(source,destination,static_cast<gpu_ops::ImageConversion>(code),h,w,c,{});
            auto cpu=destination.cpu();const auto* b=static_cast<const uint8_t*>(cpu.data_ptr());const auto* f=static_cast<const float*>(cpu.data_ptr());
            for(size_t i=0;i<count;++i){size_t from=(i%n)*c+i/n;float expected=0;
                if(code==2){uint16_t got=0;std::memcpy(&got,b+i*2,2);require(got==uint16_t(std::clamp(floats[i]*65535.f,0.f,65535.f)+.5f),"Core U16 encoding differs");continue;}
                if(code==8 || code==9 || code==10){uint8_t expected_byte=code==8?bytes[from]:code==9?uint8_t((uint32_t(shorts[from])*255+32767)/65535):uint8_t(std::clamp(floats[i],0.f,1.f)*255.f+.5f);require(b[i]==expected_byte,"Core byte conversion/layout differs");continue;}
                if(code==0 || code==14)expected=code==14 && i/n==3?float(bytes[from])/255:float(bytes[from])*(1.f/255);
                if(code==1 || code==15)expected=code==15 && i/n==3?float(shorts[from])/65535:float(shorts[from])*(1.f/65535);
                if(code==3)expected=float(shorts[i])*(1.f/65535);
                if(code==4)expected=std::clamp(floats[(i%3)*n+i/3]*.5f+.5f,0.f,1.f);
                if(code==5 || code==12)expected=floats[from]*2-1;
                if(code==6)expected=float(bytes[from])*(1.f/255)*2-1;
                if(code==7)expected=float(shorts[from])*(1.f/65535)*2-1;
                if(code==11)expected=float(bytes[i])*(1.f/255);
                if(code==13)expected=float(bytes[i])/255;
                require(std::abs(f[i]-expected)<3e-7,"Core Float32 conversion/layout differs");
            }
        }
        auto sentinel=Tensor::empty({4099},Device::GPU,DataType::UInt8);auto flag=Tensor::ones({1},Device::GPU,DataType::UInt32);
        ops->sentinel_fill(sentinel,123);ops->sentinel_check(sentinel,flag,123);require(flag.cpu().ptr<uint32_t>()[0]==1,"Core sentinel check rejected original");
        auto changed=Tensor::zeros({4099},Device::GPU,DataType::UInt8);ops->sentinel_check(changed,flag,123);require(flag.cpu().ptr<uint32_t>()[0]==0,"Core sentinel check missed change");
        for(unsigned model=0;model<=4;++model)for(unsigned channels:{1u,3u,4u})for(bool byte_input:{false,true}) {
            std::vector<float> values(channels*n);std::vector<uint8_t> bytes(values.size());for(size_t i=0;i<values.size();++i){values[i]=float((i*73)%257)/256;bytes[i]=uint8_t(i*71);}
            auto cpu=byte_input?Tensor::from_blob(bytes.data(),{channels,h,w},Device::CPU,DataType::UInt8):Tensor::from_vector(values,{channels,h,w},Device::CPU);
            auto input=cpu.to(Device::GPU);UndistortParams p{12,13,4.1f,3.7f,9,10,3.2f,2.2f,9,7,7,5,static_cast<CameraModelType>(model),{},10};p.distortion[0]=.03f;p.distortion[1]=-.004f;p.distortion[4]=.0003f;p.distortion[6]=-.0001f;
            auto reference=internal::warp_image_tensor(cpu,p,0,false,nullptr),result=ops->undistort(input,p,false);
            compare(result,reference,1e-5f,"Camera image warp differs from CPU");
            if(model==0){p.dst_width=p.src_width;p.dst_height=p.src_height;p.dst_fx=p.src_fx;p.dst_fy=p.src_fy;p.dst_cx=p.src_cx;p.dst_cy=p.src_cy;p.num_distortion=0;
                auto identity=ops->undistort(input,p,false);auto expected=byte_input?cpu.to(DataType::Float32).div(255.f):cpu;compare(identity,expected,0,"Identity camera warp changed input");}
        }
        for(unsigned model=0;model<=4;++model)for(int mode:{1,2,3}) {
            const size_t channels=mode==3?3:1;std::vector<float> values(channels*n);for(size_t i=0;i<values.size();++i)values[i]=float((i*71)%157)/100;
            if(mode>1){values[2]=std::numeric_limits<float>::quiet_NaN();values[7]=0;}
            auto cpu=Tensor::from_vector(values,mode==3?TensorShape({3,h,w}):TensorShape({h,w}),Device::CPU),gpu=cpu.to(Device::GPU);
            UndistortParams p{12,13,4.1f,3.7f,9,10,3.2f,17.2f,9,7,5,37,static_cast<CameraModelType>(model),{},3};p.distortion[0]=.03f;p.distortion[1]=-.004f;
            auto reference=internal::warp_image_tensor(cpu,p,mode,false,nullptr);
            auto result=mode==1?ops->undistort(gpu,p,true):internal::warp_image_tensor(gpu,p,mode,false,nullptr);
            compare(result,reference,1e-5f,"Camera mask/prior warp differs from CPU");
        }
        for(auto mode:{gpu_ops::Resample::LanczosRGB,gpu_ops::Resample::LanczosGray,gpu_ops::Resample::LanczosFloatCHW,gpu_ops::Resample::DepthPrior,gpu_ops::Resample::NormalPrior}) {
            auto input=Tensor::full(mode==gpu_ops::Resample::LanczosGray || mode==gpu_ops::Resample::DepthPrior?TensorShape({h,w}):mode==gpu_ops::Resample::LanczosRGB?TensorShape({h,w,3}):TensorShape({3,h,w}),1.f,Device::GPU);
            auto output=ops->resize(input,13,17,mode,3);for(float value:output.cpu().to_vector())require(std::abs(value-(mode==gpu_ops::Resample::NormalPrior?1.f/std::sqrt(3.f):1.f))<2e-6,"Core shared resize changed constant image/prior");
        }
        auto cpu=Tensor::ones({h,w,3},Device::CPU),destination=Tensor::empty({3,h,w},Device::GPU);
        bool rejected=false;try{ops->convert(cpu,destination,gpu_ops::ImageConversion::U8HWCToF32CHW,h,w,3,{});}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Shared image accepted CPU/foreign storage");
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"Shared image changed default or leaked resources");
    std::puts("PASS HIP application shared image: lazy complete table, 16 conversions, sentinel, five lens models/all color dtypes, identity, mask/depth/normal row tiling, five resizes, foreign input rejection and teardown");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"SHARED IMAGE CONTRACT FAILED: %s\n",e.what());return 1;}
