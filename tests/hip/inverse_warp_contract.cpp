/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_image.hpp>
#include "core/tensor/internal/undistort_resample.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void near(const Tensor& a,const Tensor& b,float tolerance,const char* message) {
    require(a.shape()==b.shape() && a.dtype()==b.dtype(),"Inverse warp shape/dtype mismatch");
    const auto x=a.cpu().to_vector(),y=b.cpu().to_vector();
    float error=0;
    for(size_t i=0;i<x.size();++i){require(std::isfinite(x[i])&&std::isfinite(y[i]),"Nonfinite inverse warp result");error=std::max(error,std::abs(x[i]-y[i]));}
    if(error>tolerance)std::fprintf(stderr,"Inverse warp max error %.9g > %.9g\n",error,tolerance);
    require(error<=tolerance,message);
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        constexpr size_t w=17,h=39;
        std::vector<float> values(3*w*h);
        for(size_t i=0;i<values.size();++i)values[i]=float(i*19%251)/251.f;
        const auto cpu=Tensor::from_vector(values,{3,h,w},Device::CPU);
        UndistortParams p{18,19,7.8f,18.1f,20,21,8.4f,19.3f,17,37,17,39,CameraModelType::PINHOLE,{},0};
        auto input=cpu.to(Device::GPU);Tensor mask;
        auto output=distort_image_to_source(input,p,mask,nullptr);
        const auto actual=output.cpu().to_vector();
        const auto valid=mask.cpu().to_vector_uint8();
        require(mask.dtype()==DataType::UInt8 && mask.shape()==TensorShape{37,17},"Inverse warp validity layout changed");
        // No distortion: inverse rays are analytical; the CPU sampler provides
        // an independent footprint/coverage and Lanczos reference.
        size_t valid_count=0;
        for(int y=0;y<p.src_height;++y)for(int x=0;x<p.src_width;++x) {
            float sum[4]{},inside=0,full=0;bool converged=true;
            for(int qy=0;qy<8 && converged;++qy)for(int qx=0;qx<8;++qx) {
                const float ux=(float(x)+(float(qx)+.5f)/8-p.src_cx)/p.src_fx;
                const float uy=(float(y)+(float(qy)+.5f)/8-p.src_cy)/p.src_fy;
                float sample[4]{},a=0,b=0;
                if(!internal::warp_math::lanczos3_sample(values.data(),p.dst_width,p.dst_height,3,
                    std::fma(ux,p.dst_fx,p.dst_cx)-.5f,std::fma(uy,p.dst_fy,p.dst_cy)-.5f,sample,a,b)){converged=false;break;}
                inside+=a;full+=b;for(int c=0;c<3;++c)sum[c]+=sample[c];
            }
            const bool keep=converged && full>0 && inside/full>=.999f;
            const size_t i=size_t(y)*p.src_width+x;
            require((valid[i]!=0)==keep,"Inverse warp validity differs from CPU coverage");
            valid_count+=keep;
            for(size_t c=0;c<3;++c)require(std::abs(actual[c*p.src_width*p.src_height+i]-(keep?sum[c]/64:0))<2e-5,"Inverse warp differs from analytical CPU rays");
        }
        require(valid_count>0 && valid_count<valid.size(),"Inverse warp reference did not exercise both coverage outcomes");
        auto depth_hw=Tensor::full({h,w},2.f,Device::GPU);
        auto depth_chw=depth_hw.unsqueeze(0);
        const auto hw_depth=distort_depth_to_source_area(depth_hw,p,nullptr);
        const auto chw_depth=distort_depth_to_source_area(depth_chw,p,nullptr);
        near(hw_depth,chw_depth.squeeze(0),0,"Inverse depth changed between HW and renderer CHW layouts");
#ifdef LFS_TENSOR_VULKAN
        for(int model=0;model<5;++model)for(int mode=0;mode<4;++mode) {
            p.model_type=static_cast<CameraModelType>(model);p.num_distortion=4;p.distortion[0]=.015f;p.distortion[1]=-.002f;
            const auto shape=mode==1||mode==2?TensorShape{h,w}:TensorShape{3,h,w};
            auto data=Tensor::full(shape,mode==3?.577350269f:.25f,Device::CPU);
            Tensor native,reference,native_mask,reference_mask;
            {
                const GpuBackendScope scope(GpuBackend::HIP);
                native=internal::warp_image_tensor(data.to(Device::GPU),p,mode,true,mode==0?&native_mask:nullptr);
            }
            {
                const GpuBackendScope scope(GpuBackend::Vulkan);
                reference=internal::warp_image_tensor(data.to(Device::GPU),p,mode,true,mode==0?&reference_mask:nullptr);
            }
            near(native,reference,3e-5,"HIP inverse warp differs from established Vulkan implementation");
            if(mode==0)require(native_mask.cpu().to_vector_uint8()==reference_mask.cpu().to_vector_uint8(),"HIP/Vulkan inverse validity mismatch");
        }
#endif
    }
#ifdef LFS_TENSOR_VULKAN
    require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"Inverse warp retained Vulkan reference storage");
#endif
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Inverse warp retained HIP storage");
    std::puts("PASS HIP inverse image warp: analytical CPU rays/coverage, odd dimensions and row seams, five lens models and color/mask/depth/normal against Vulkan");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP INVERSE WARP FAILED: %s\n",e.what());return 1;}
