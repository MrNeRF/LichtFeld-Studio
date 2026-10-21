/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/detail/tensor_half.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
using namespace lfs;
using namespace core;
namespace lfs::training {
unsigned other_tables=0;
const TrainingOps& cuda_training_ops_table(){++other_tables;static const TrainingOps t{.backend=GpuBackend::CUDA};return t;}
const TrainingOps& vulkan_training_ops_table(){++other_tables;static const TrainingOps t{.backend=GpuBackend::Vulkan};return t;}
const TrainingOps& metal_training_ops_table(){++other_tables;static const TrainingOps t{.backend=GpuBackend::Metal};return t;}
}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
float half(float value){return core::detail::tensor_half_to_float(core::detail::tensor_float_to_half(value));}
std::vector<float> values(const Tensor& tensor){return tensor.cpu().to(DataType::Float32).to_vector();}
float activate(float x,nn::Activation mode){
    const double v=x;
    switch(mode){
    case nn::Activation::None:return x;
    case nn::Activation::Relu:return std::max(0.f,x);
    case nn::Activation::Silu:return float(v/(1+std::exp(-v)));
    case nn::Activation::GeluTanh:return float(.5*v*(1+std::tanh(std::sqrt(2./3.14159265358979323846)*(v+.044715*v*v*v))));
    case nn::Activation::GeluErf:return float(.5*v*(1+std::erf(v/std::sqrt(2.))));
    }
    throw std::runtime_error("Unknown CPU activation");
}
int half_order(float x){const auto code=core::detail::tensor_float_to_half_bits(x);return code&0x8000?int(uint16_t(~code)):int(code|0x8000);}
void convolution_case(const gpu_ops::LpipsOps& ops,DataType type,bool replicate,int stride,int dilation,nn::Activation activation){
    constexpr size_t batches=2,ci=5,co=17,h=7,w=11;
    const int padding=dilation;
    const size_t oh=(h+2*padding-2*dilation-1)/stride+1,ow=(w+2*padding-2*dilation-1)/stride+1;
    std::vector<float> x(batches*ci*h*w),weight(co*ci*9),bias(co);
    for(size_t i=0;i<x.size();++i)x[i]=float(int((i*17)%31)-15)*.03125f;
    for(size_t i=0;i<weight.size();++i)weight[i]=float(int((i*13)%19)-9)*.0078125f;
    for(size_t i=0;i<bias.size();++i)bias[i]=float(int(i%7)-3)*.015625f;
    auto input=Tensor::from_vector(x,{batches,ci,h,w},Device::GPU).to(type);
    auto weights=Tensor::from_vector(weight,{co,ci,3,3},Device::GPU).to(type);
    auto biases=Tensor::from_vector(bias,{co},Device::GPU).to(type);
    auto output=Tensor::empty({batches,co,oh,ow},Device::GPU,type);Tensor absent,scratch;
    gpu_ops::ConvParams p;p.pad_h=p.pad_w=padding;p.stride_h=p.stride_w=stride;p.dilation_h=p.dilation_w=dilation;
    p.pad_mode=replicate?nn::ConvPadMode::Replicate:nn::ConvPadMode::Zeros;p.activation=activation;
    ops.convolution(input,weights,absent,biases,output,scratch,p);
    require(!scratch.is_valid(),"HIP LPIPS allocated an unexpected full convolution scratch");
    const auto actual=values(output);
    for(size_t n=0;n<batches;++n)for(size_t c=0;c<co;++c)for(size_t y=0;y<oh;++y)for(size_t xx=0;xx<ow;++xx){
        float total=0;
        for(size_t ic=0;ic<ci;++ic)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
            int iy=int(y*stride)+ky*dilation-padding,ix=int(xx*stride)+kx*dilation-padding;
            if(replicate){iy=std::clamp(iy,0,int(h)-1);ix=std::clamp(ix,0,int(w)-1);}
            if(iy<0 || ix<0 || iy>=int(h) || ix>=int(w))continue;
            total=std::fma(x[((n*ci+ic)*h+iy)*w+ix],weight[(c*ci+ic)*9+ky*3+kx],total);
        }
        const float expected=activate(total+bias[c],activation),got=actual[((n*co+c)*oh+y)*ow+xx];
        if(type==DataType::Float16)require(std::abs(half_order(got)-half_order(expected))<=1,"HIP LPIPS tiled half convolution differs by more than one half ULP");
        else require(std::abs(got-expected)<=2e-6f,"HIP LPIPS tiled float convolution differs from independent CPU reference");
    }
    if(type==DataType::Float16){
        auto taps=Tensor::empty({9,co,ci},Device::GPU,DataType::Float16);ops.weight_taps(weights,taps);
        const auto packed=values(taps);
        for(size_t t=0;t<9;++t)for(size_t oc=0;oc<co;++oc)for(size_t ic=0;ic<ci;++ic)
            require(packed[t*co*ci+oc*ci+ic]==weight[(oc*ci+ic)*9+t],"HIP LPIPS weight tap permutation differs");
    }
}
void rgb_case(const gpu_ops::LpipsOps& ops,bool official){
    constexpr size_t n=2,ci=3,co=7,h=5,w=7;
    std::vector<float> x(n*ci*h*w),weights(co*ci*9),bias(co);
    for(size_t i=0;i<x.size();++i)x[i]=float((i*11)%33)/32.f;
    for(size_t i=0;i<weights.size();++i)weights[i]=float(int(i%13)-6)*.015625f;
    for(size_t i=0;i<bias.size();++i)bias[i]=float(int(i)-3)*.03125f;
    auto input=Tensor::from_vector(x,{n,ci,h,w},Device::GPU),weight=Tensor::from_vector(weights,{co,ci,3,3},Device::GPU).to(DataType::Float16),b=Tensor::from_vector(bias,{co},Device::GPU).to(DataType::Float16),output=Tensor::empty({n,co,h,w},Device::GPU,DataType::Float16);
    const gpu_ops::RGBConvParams p{{.125f,-.25f,.5f},{.5f,1.f,2.f},official};ops.rgb_conv(input,weight,b,output,p);
    const auto got=values(output);
    for(size_t batch=0;batch<n;++batch)for(size_t oc=0;oc<co;++oc)for(size_t y=0;y<h;++y)for(size_t xx=0;xx<w;++xx){
        float reference=0;
        for(size_t channel=0;channel<ci;++channel)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
            const int iy=int(y)+ky-1,ix=int(xx)+kx-1;if(iy<0 || ix<0 || iy>=int(h) || ix>=int(w))continue;
            float value=x[((batch*ci+channel)*h+iy)*w+ix];if(official)value=2.f*value-1.f;
            value=half((value-p.shift[channel])/p.scale[channel]);
            reference=std::fma(value,weights[(oc*ci+channel)*9+ky*3+kx],reference);
        }
        reference=half(std::max(0.f,reference+bias[oc]));
        require(got[((batch*co+oc)*h+y)*w+xx]==reference,"HIP LPIPS RGB normalization/convolution differs from exact CPU binary-fraction reference");
    }
}
void pool_case(const gpu_ops::LpipsOps& ops,bool weighted){
    constexpr size_t n=2,c=8,h=5,w=7,oh=h/2,ow=w/2,ww=11;
    std::vector<float> x(n*c*h*w),y(x.size()),linear(c),mask(9*ww);
    for(size_t i=0;i<x.size();++i){x[i]=float(int(i%23)-11)*.0625f;y[i]=float(int((i*7)%29)-14)*.03125f;}
    for(size_t i=0;i<c;++i)linear[i]=float(i+1)*.03125f;
    for(size_t i=0;i<mask.size();++i)mask[i]=float(i%5)*.25f;
    auto a=Tensor::from_vector(x,{n,c,h,w},Device::GPU).to(DataType::Float16),b=Tensor::from_vector(y,{n,c,h,w},Device::GPU).to(DataType::Float16),weights=Tensor::from_vector(linear,{c},Device::GPU).to(DataType::Float16),pixel=Tensor::from_vector(mask,{9,ww},Device::GPU);
    auto score=Tensor::full({1},.125f,Device::GPU),px=Tensor::empty({n,c,oh,ow},Device::GPU,DataType::Float16),py=Tensor::empty({n,c,oh,ow},Device::GPU,DataType::Float16);
    const gpu_ops::PoolReduceParams p{1,4,2,6,1.f/24,weighted?&pixel:nullptr,int(ww),2,1};
    ops.pool_reduce(a,b,weights,score,px,py,p);
    double reference=.125;
    for(size_t batch=0;batch<n;++batch)for(int yy=1;yy<4;++yy)for(int xx=2;xx<6;++xx){
        double sx=1e-10,sy=1e-10;
        for(size_t channel=0;channel<c;++channel){const size_t at=((batch*c+channel)*h+yy)*w+xx;sx+=double(x[at])*x[at];sy+=double(y[at])*y[at];}
        double contribution=0;
        for(size_t channel=0;channel<c;++channel){const size_t at=((batch*c+channel)*h+yy)*w+xx;const double difference=x[at]/std::sqrt(sx)-y[at]/std::sqrt(sy);contribution+=difference*difference*linear[channel];}
        reference+=contribution*p.inverse_count*(weighted?mask[(yy+2)*ww+xx+1]:1);
    }
    require(std::abs(values(score)[0]-reference)<2e-6,"HIP LPIPS weighted normalized score differs from independent CPU reference");
    const auto gx=values(px),gy=values(py);
    for(size_t batch=0;batch<n;++batch)for(size_t channel=0;channel<c;++channel)for(size_t yy=0;yy<oh;++yy)for(size_t xx=0;xx<ow;++xx){
        const size_t start=((batch*c+channel)*h+yy*2)*w+xx*2,dst=((batch*c+channel)*oh+yy)*ow+xx;
        require(gx[dst]==std::max({x[start],x[start+1],x[start+w],x[start+w+1]}) && gy[dst]==std::max({y[start],y[start+1],y[start+w],y[start+w+1]}),"HIP LPIPS odd-size pooling differs from CPU");
    }
    Tensor absent;auto same=Tensor::full({1},.125f,Device::GPU);ops.pool_reduce(a,a,weights,same,absent,absent,p);
    require(values(same)[0]==.125f,"HIP LPIPS identical inputs changed an accumulated score");
}
int main(int argc,char** argv)try{
    require(argc==2,"Pass absolute HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot configure module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot configure module");
#endif
    const auto configured=configured_gpu_backend();const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.lpips && !table.fast && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"HIP LPIPS discovery initialized another backend");
    {
        const GpuBackendScope scope(GpuBackend::HIP);const auto& ops=*table.lpips;
        require(!ops.prefer_independent_queue,"HIP LPIPS advertised unsupported independent GPU submission");
        for(auto type:{DataType::Float32,DataType::Float16})for(bool replicate:{false,true})
            for(auto activation:{nn::Activation::None,nn::Activation::Relu,nn::Activation::GeluTanh,nn::Activation::GeluErf,nn::Activation::Silu})
                convolution_case(ops,type,replicate,replicate?2:1,replicate?2:1,activation);
        rgb_case(ops,false);rgb_case(ops,true);pool_case(ops,false);pool_case(ops,true);
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP LPIPS changed backend selection or leaked storage");
    std::puts("PASS HIP LPIPS native table: tiled Float32/Float16 convolutions, five activations, RGB normalization, exact taps, weighted/unweighted pooling and score accumulation against CPU");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP LPIPS CONTRACT FAILED: %s\n",e.what());return 1;}
