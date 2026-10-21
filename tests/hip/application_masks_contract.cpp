/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/masks.hpp>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;using namespace lfs::training;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void near(double a,double b,double tolerance=3e-6){if(!std::isfinite(a)||std::abs(a-b)>tolerance*(1+std::abs(b))){std::fprintf(stderr,"actual %.12g reference %.12g\n",a,b);throw std::runtime_error("Mask differs from reference");}}
static void core_fill() {
    constexpr size_t h=3,w=19;std::vector<float> values(h*w),predicate(w);for(size_t i=0;i<values.size();++i)values[i]=float(i%5);for(size_t x=0;x<w;++x)predicate[x]=float(x%3==0);
    auto mask=Tensor::from_vector(predicate,{1,w},Device::GPU).gt(0.f);
    for(auto dtype:{DataType::Float32,DataType::Float16,DataType::Int32,DataType::Int64,DataType::UInt8,DataType::UInt32,DataType::Bool}) {
        auto source=Tensor::from_vector(values,{h,w},Device::GPU).to(dtype);const auto before=source.to(DataType::Float32).cpu().to_vector();const float value=dtype==DataType::Bool?1.f:7.f;
        auto filled=source.masked_fill(mask,value);const auto result=filled.to(DataType::Float32).cpu().to_vector();for(size_t i=0;i<result.size();++i)near(result[i],predicate[i%w]?value:before[i],0);
        const auto original=source.to(DataType::Float32).cpu().to_vector();require(original==before,"Non-mutating masked fill changed its source");
    }
    std::vector<int64_t> exact(h*w);for(size_t i=0;i<exact.size();++i)exact[i]=(int64_t{1}<<54)+int64_t(i)*17;
    auto source=Tensor::from_blob(exact.data(),{h,w},Device::CPU,DataType::Int64).clone().to(Device::GPU);auto result=source.masked_fill(mask,7.f).cpu();for(size_t i=0;i<exact.size();++i)require(result.ptr<int64_t>()[i]==(predicate[i%w]?7:exact[i]),"Masked fill rounded an unchanged Int64 lane");
}
static void weights() {
    constexpr size_t h=17,w=19,n=h*w;
    const auto* ops=training_ops(GpuBackend::HIP).masks;require(ops,"Missing HIP mask family");
    std::vector<float> values(n),roi(n),alphas(n);const unsigned levels[]={0,1,127,128,250,251,255};
    for(size_t i=0;i<n;++i){values[i]=float(levels[i%7])/255;roi[i]=float(i%5)*.2f;alphas[i]=.2f+float(i%11)*.05f;}
    for(bool bytes:{false,true})for(bool weighted:{false,true})for(int mode=0;mode<2;++mode) {
        auto m=Tensor::from_vector(values,{h,w},Device::GPU);if(bytes){std::vector<uint8_t> b(n);for(size_t i=0;i<n;++i)b[i]=levels[i%7];m=Tensor::from_blob(b.data(),{h,w},Device::CPU,DataType::UInt8).clone().to(Device::GPU);}
        auto a=Tensor::from_vector(alphas,{h,w},Device::GPU);Tensor r;if(weighted)r=Tensor::from_vector(roi,{h,w},Device::GPU);
        auto output=Tensor::empty({h,w},Device::GPU),temp=Tensor::empty({(n+255)/256},Device::GPU),loss=Tensor::empty({1},Device::GPU);
        ops->photometric_weight(m,r,output,static_cast<lfs::gpu_ops::MaskPhotoMode>(mode));const auto photo=output.cpu().to_vector();
        for(size_t i=0;i<n;++i){double expected=mode?(levels[i%7]>250?1:0):(bytes?(levels[i%7]>0?1:0):values[i]);near(photo[i],expected*(weighted?roi[i]:1));}
        ops->opacity_penalty(a,m,r,output,temp,loss,static_cast<lfs::gpu_ops::MaskOpacityMode>(mode),2.f,.7f);const auto gradient=output.cpu().to_vector();double expected_loss=0;
        for(size_t i=0;i<n;++i){double bg=mode?(levels[i%7]>=128&&levels[i%7]<=250?1:0):1-(bytes?(levels[i%7]>0?1:0):values[i]);double g=bg*bg*.7/n*(weighted?roi[i]:1);near(gradient[i],g);expected_loss+=alphas[i]*g;}near(loss.cpu().item<float>(),expected_loss);
        ops->alpha_consistency(a,m,r,output,temp,loss,.8f);const auto consistent=output.cpu().to_vector();expected_loss=0;
        for(size_t i=0;i<n;++i){double target=bytes?(levels[i%7]>0?1:0):values[i],diff=alphas[i]-target,rw=weighted?roi[i]:1;near(consistent[i],(diff>0?1:diff<0?-1:0)*.8/n*rw);expected_loss+=std::abs(diff)*.8/n*rw;}near(loss.cpu().item<float>(),expected_loss);
    }
}
struct Coverage {std::vector<uint8_t> mesh,points;};
static Coverage coverage(GpuBackend backend,bool distorted) {
    GpuBackendScope scope(backend);const auto* ops=training_ops(backend).masks;
    constexpr size_t h=33,w=35;lfs::gpu_ops::MeshMaskCamera k;k.width=w;k.height=h;k.fx=20;k.fy=21;k.cx=17.5;k.cy=16.5;
    std::vector<float> xyz{-1,-1,2,1,-1,2,0,1,2, .1f,.1f,1.f,.15f,.1f,1.f,.1f,.15f,1.f, 0,0,-1};
    std::vector<int32_t> faces{0,1,2,3,4,5,0,1,6,-1,1,2};
    auto v=Tensor::from_vector(xyz,{7,3},Device::GPU),idx=Tensor::from_blob(faces.data(),{4,3},Device::CPU,DataType::Int32).clone().to(Device::GPU);
    Tensor samples;UndistortParams d{};if(distorted) {
        d.src_fx=k.fx;d.src_fy=k.fy;d.src_cx=k.cx;d.src_cy=k.cy;d.dst_fx=k.fx;d.dst_fy=k.fy;d.dst_cx=k.cx;d.dst_cy=k.cy;d.src_width=d.dst_width=w;d.src_height=d.dst_height=h;d.model_type=CameraModelType::PINHOLE;
        std::vector<float> rays(h*w*2);for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){rays[(y*w+x)*2]=(float(x)+.5f-k.cx)/k.fx;rays[(y*w+x)*2+1]=(float(y)+.5f-k.cy)/k.fy;}samples=Tensor::from_vector(rays,{h,w,2},Device::GPU);
    }
    auto mesh=ops->mesh_coverage(v,idx,k,samples,distorted?&d:nullptr,.01f),points=ops->point_coverage(v,k,2,distorted?&d:nullptr);
    Coverage result;auto mc=mesh.cpu(),pc=points.cpu();result.mesh.assign(mc.ptr<uint8_t>(),mc.ptr<uint8_t>()+h*w);result.points.assign(pc.ptr<uint8_t>(),pc.ptr<uint8_t>()+h*w);
    require(result.mesh[16*w+17]!=0,"Mesh coverage missing interior");require(result.mesh[0]==0,"Mesh coverage outside triangle");return result;
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");core_fill();weights();
    for(bool distortion:{false,true}){auto a=coverage(GpuBackend::HIP,distortion),b=coverage(GpuBackend::Vulkan,distortion);require(a.mesh==b.mesh,"Mesh coverage differs from Vulkan");require(a.points==b.points,"Point coverage differs from Vulkan");}
    require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"Vulkan masks retained storage");require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP masks retained storage");std::puts("PASS HIP masks: CPU photo/opacity/alpha references, all mask bands, Float32/UInt8, ROI, large/small/near-clipped/invalid faces, point disks, inverse samples and Vulkan coverage parity");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());(void)shutdown_gpu_backend(GpuBackend::HIP);return 1;}
