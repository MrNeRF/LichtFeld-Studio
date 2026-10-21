/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/structure.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
using namespace lfs::training;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void close(double a,double b,double tolerance,const char* m){require(std::isfinite(a)&&std::abs(a-b)<=tolerance,m);}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        const auto* ops=training_ops(GpuBackend::HIP).structure;
        require(ops,"HIP Structure family is unavailable");
        constexpr size_t h=19,w=37,n=h*w;
        std::vector<float> image(3*n),target(3*n),confidence(n),structure(n),error(n);
        for(size_t i=0;i<3*n;++i){image[i]=float((i*31)%251)/251.f;target[i]=float((i*17)%257)/257.f;}
        for(size_t i=0;i<n;++i){confidence[i]=i%13?0.25f:0.f;structure[i]=float(i%19)/3.f;error[i]=float(i%7)/8.f;}
        auto a=Tensor::from_vector(image,{3,h,w},Device::GPU),b=Tensor::from_vector(target,{3,h,w},Device::GPU);
        auto mask=Tensor::from_vector(confidence,{h,w},Device::GPU),s=Tensor::from_vector(structure,{h,w},Device::GPU);
        for(bool padding:{false,true}) {
            Tensor result;ops->photometric_weight(s,mask,result,0.75f,padding);
            const auto actual=result.cpu().to_vector();
            for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x) {
                const bool inside=!padding || (x>=5 && x<w-5 && y>=5 && y<h-5);
                close(actual[y*w+x],inside?confidence[y*w+x]*(1+0.75f*std::min(structure[y*w+x],4.f)):0,2e-7,"Structure photometric weight differs from CPU");
            }
        }
        auto e=Tensor::from_vector(error,{h,w},Device::GPU);ops->densification_weight(e,s,0.5f);
        const auto actual_error=e.cpu().to_vector();
        for(size_t i=0;i<n;++i)close(actual_error[i],error[i]*(1+0.5f*std::clamp(structure[i],0.f,4.f)),2e-7,"Structure growth weight differs from CPU");
        // Independent double-precision Sobel loss and its analytical transpose.
        std::vector<double> residual(n),expected(3*n,0.125),gx(n),gy(n),weight(n);
        for(size_t i=0;i<n;++i)residual[i]=.2126*(image[i]-target[i])+.7152*(image[n+i]-target[n+i])+.0722*(image[2*n+i]-target[2*n+i]);
        auto sx=[](int x,int y){return x*(y==0?2.:1.)*.125;};
        auto sy=[](int x,int y){return y*(x==0?2.:1.)*.125;};
        double loss=0,denominator=0;
        for(size_t y=1;y<h-1;++y)for(size_t x=1;x<w-1;++x) {
            const auto i=y*w+x;bool valid=true;
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){const auto j=size_t(ptrdiff_t(i)+dy*ptrdiff_t(w)+dx);valid=valid&&confidence[j]>0;gx[i]+=sx(dx,dy)*residual[j];gy[i]+=sy(dx,dy)*residual[j];}
            weight[i]=valid?confidence[i]:0;
            loss+=weight[i]*(std::sqrt(gx[i]*gx[i]+1e-6)+std::sqrt(gy[i]*gy[i]+1e-6)-.002);
            denominator+=weight[i];
        }
        const double scale=denominator>0?1.25/(2*denominator):0;
        for(size_t y=1;y<h-1;++y)for(size_t x=1;x<w-1;++x) {
            const auto i=y*w+x;
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
                const auto j=size_t(ptrdiff_t(i)+dy*ptrdiff_t(w)+dx);
                const auto d=scale*weight[i]*(sx(dx,dy)*gx[i]/std::sqrt(gx[i]*gx[i]+1e-6)+sy(dx,dy)*gy[i]/std::sqrt(gy[i]*gy[i]+1e-6));
                expected[j]+=.2126*d;expected[n+j]+=.7152*d;expected[2*n+j]+=.0722*d;
            }
        }
        auto gradient=Tensor::full({3,h,w},.125f,Device::GPU);
        kernels::GradientResidualWorkspace workspace;
        const auto actual_loss=ops->gradient_residual(a,b,mask,gradient,1.25f,workspace).cpu().item<float>();
        close(actual_loss,loss*scale,2e-6,"Structure gradient loss differs from CPU");
        const auto actual_gradient=gradient.cpu().to_vector();
        for(size_t i=0;i<3*n;++i)close(actual_gradient[i],expected[i],2e-6,"Structure gradient differs from analytical CPU reference");
        kernels::RidgeWorkspace full,banded;
        banded.band_bytes=3*w*sizeof(float)*28;
        auto ridge=Tensor::empty({h,w},Device::GPU),other=Tensor::empty({h,w},Device::GPU);
        ops->ridge(a,ridge,full,{},lfs::gpu_ops::MaskPhotoMode::BinaryGt0);
        ops->ridge(a,other,banded,{},lfs::gpu_ops::MaskPhotoMode::BinaryGt0);
        const auto r=ridge.cpu().to_vector(),q=other.cpu().to_vector();double total=0;
        for(size_t i=0;i<n;++i){close(r[i],q[i],1e-5,"Ridge row-band seam changed output");require(r[i]>=0,"Negative ridge response");total+=r[i];}
        close(total/n,1,1e-5,"Ridge normalization changed");
        auto keep=Tensor::ones({h,w},Device::GPU);
        ops->ridge(a,other,banded,keep,lfs::gpu_ops::MaskPhotoMode::BinaryGt0);
        const auto masked=other.cpu().to_vector();
        for(size_t i=0;i<n;++i)close(masked[i],r[i],1e-5,"All-valid ridge mask changed normalization");
        std::vector<uint8_t> bytes(3*n);std::vector<float> normalized(3*n);
        for(size_t i=0;i<3*n;++i){bytes[i]=uint8_t(i*31%251);normalized[i]=bytes[i]*(1.f/255.f);}
        auto byte_image=Tensor::from_blob(bytes.data(),{3,h,w},Device::CPU,DataType::UInt8).to(Device::GPU);
        auto normalized_image=Tensor::from_vector(normalized,{3,h,w},Device::GPU);
        ops->ridge(byte_image,ridge,full,{},lfs::gpu_ops::MaskPhotoMode::BinaryGt0);
        ops->ridge(normalized_image,other,banded,{},lfs::gpu_ops::MaskPhotoMode::BinaryGt0);
        const auto byte_ridge=ridge.cpu().to_vector(),float_ridge=other.cpu().to_vector();
        for(size_t i=0;i<n;++i)close(byte_ridge[i],float_ridge[i],3e-5,"U8 ridge differs from normalized RGB reference");
        auto zero=Tensor::zeros({h,w},Device::GPU);
        ops->ridge(a,other,banded,zero,lfs::gpu_ops::MaskPhotoMode::BinaryGt0);
        for(float v:other.cpu().to_vector())require(v==0,"Fully masked ridge must be zero");
        gradient.fill_(.125f);
        close(ops->gradient_residual(a,b,zero,gradient,1.25f,workspace).cpu().item<float>(),0,0,"Fully masked gradient loss must be zero");
        for(float v:gradient.cpu().to_vector())require(v==.125f,"Fully masked gradient changed accumulated input");
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Structure contract retained native storage");
    std::puts("PASS HIP Structure: CPU photometric/growth/analytical Sobel references, odd tiles, gradient accumulation, ridge normalization, bounded row bands and empty masks");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP STRUCTURE FAILED: %s\n",e.what());return 1;}
