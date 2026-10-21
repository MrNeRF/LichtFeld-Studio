/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "training_session_fixture.hpp"
#include <core/tensor_backend.hpp>
#include <core/sh_layout.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/gsplat.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs;using namespace lfs::core;using namespace lfs::training;
static void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
static void near(double a,double b,double tolerance=4e-4){if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tolerance*(1+std::abs(b))){std::fprintf(stderr,"actual %.12g reference %.12g\n",a,b);throw std::runtime_error("Gsplat differs from reference");}}
struct Gradients {
    std::array<Tensor,6> values;
    static Tensor& get(void* p,gpu_ops::AdamSlot slot){return static_cast<Gradients*>(p)->values[size_t(slot)];}
};
struct RasterComparison {std::vector<float> image,alpha,depth,normal,density,shares,rays;std::array<std::vector<float>,6> gradients;};
static RasterComparison evaluate(GpuBackend backend,int mode,unsigned degree=0,int camera=0,bool background=false,bool tile=false,bool camera_rays=false) {
    GpuBackendScope scope(backend);const auto* ops=training_ops(backend).gsplat;require(ops,"Missing Gsplat family");
    hip_fixture::Fixture fixture;fixture.perturb();constexpr size_t n=hip_fixture::n,h=17,w=19;std::array<Tensor,5> params;constexpr size_t attrs[]{3,4,3,3,1};for(size_t i=0;i<5;++i)params[i]=Tensor::from_vector(fixture.values[i],{n,attrs[i]},Device::GPU);
    Tensor rest;if(degree){const auto size=sh_swizzled_float_count(n,15);std::vector<float> coefficients(size);for(size_t i=0;i<size;++i)coefficients[i]=float(int(i%23)-11)*.0001f;rest=Tensor::from_vector(coefficients,{size},Device::GPU);}
    auto view=Tensor::eye(4);Tensor none,image,alpha,depth,normal,bg,bgi;if(background){bg=Tensor::from_vector(std::vector<float>{.1f,.2f,.3f},{3},Device::GPU);bgi=Tensor::full({3,h,w},.15f,Device::GPU);}
    gpu_ops::SplatInputs model{params[0],params[2],params[1],params[4],params[3],rest,none};gpu_ops::RenderOutputs out{image,alpha,depth,normal};
    gpu_ops::GsplatParams p{};p.full_image={int(h)+(tile?6:0),int(w)+(tile?4:0)};p.intrinsics={32,33,p.full_image.w*.5f,p.full_image.h*.5f};if(tile){p.tile_x=2;p.tile_y=3;p.tile_w=w;p.tile_h=h;}p.camera_rays=camera_rays;p.render_mode=static_cast<gpu_ops::GsplatRenderMode>(mode);p.camera_model=static_cast<CameraModelType>(camera);p.sh.active_bases=(degree+1)*(degree+1);p.sh.layout_bases=degree?16:1;
    gpu_ops::GsplatSaved saved{ops->create()};const auto status=ops->forward(saved,model,view,none,none,bg,bgi,p,out);
    if(status.code!=gpu_ops::RasterResult::Code::Success){std::fprintf(stderr,"Gsplat forward: %.*s\n",int(status.message.size()),status.message.data());throw std::runtime_error("Gsplat forward failed");}
    RasterComparison r;if(image.is_valid())r.image=image.cpu().to_vector();r.alpha=alpha.cpu().to_vector();if(depth.is_valid())r.depth=depth.cpu().to_vector();if(normal.is_valid())r.normal=normal.cpu().to_vector();
    Gradients gradients;const size_t counts[]{n*3,n*3,n*4,n,n*3,sh_swizzled_float_count(n,degree?15:0)};for(size_t i=0;i<6;++i)if(counts[i])gradients.values[i]=Tensor::full({counts[i]},.013f,Device::GPU);
    Tensor gi,gd,gn;if(image.is_valid())gi=Tensor::full(image.shape(),.001f,Device::GPU);if(depth.is_valid())gd=Tensor::full(depth.shape(),-.0007f,Device::GPU);if(normal.is_valid())gn=Tensor::full(normal.shape(),.0003f,Device::GPU);
    auto ga=Tensor::full(alpha.shape(),-.0002f,Device::GPU),density=Tensor::zeros({2,n},Device::GPU),error=Tensor::full({h,w},.1f,Device::GPU),edge=Tensor::full({h,w},.2f,Device::GPU),scores=Tensor::zeros({n},Device::GPU),shares=Tensor::zeros({n},Device::GPU);
    gpu_ops::GsplatGradients g{&gradients,Gradients::get,gd,gn,.002f};ops->backward(saved,gi,ga,g,density,error,edge,scores,shares);
    for(size_t i=0;i<6;++i)if(gradients.values[i].is_valid())r.gradients[i]=gradients.values[i].cpu().to_vector();r.density=density.cpu().to_vector();r.shares=shares.cpu().to_vector();
    if(mode==5||camera_rays){require(saved.camera_rays.is_valid(),"Gsplat failed to expose camera rays");const auto rays=saved.camera_rays.cpu().to_vector();r.rays=rays;if(camera==0)for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){near(rays[(y*w+x)*3],(double(x)+.5+p.tile_x-p.intrinsics.cx)/32,1e-6);near(rays[(y*w+x)*3+1],(double(y)+.5+p.tile_y-p.intrinsics.cy)/33,1e-6);near(rays[(y*w+x)*3+2],1,1e-6);}}
    if(mode!=5&&!camera_rays)require(!saved.camera_rays.is_valid(),"Gsplat exposed unrequested camera rays");
    ops->release(saved);require(ops->release_caches(saved),"Gsplat caches did not release");return r;
}
static void compare(const std::vector<float>& a,const std::vector<float>& b,double tolerance=4e-4){require(a.size()==b.size(),"Gsplat result shape mismatch");for(size_t i=0;i<a.size();++i)try{near(a[i],b[i],tolerance);}catch(...){std::fprintf(stderr,"element %zu of %zu\n",i,a.size());throw;}}
static void compare_gradient(const std::vector<float>& a,const std::vector<float>& b) {
    require(a.size()==b.size(),"Gsplat gradient shape mismatch");
    for(size_t i=0;i<a.size();++i){const double reference=double(b[i])-.013f,actual=double(a[i])-.013f;const double tolerance=3e-6+.003*std::abs(reference);if(!std::isfinite(actual)||std::abs(actual-reference)>tolerance){std::fprintf(stderr,"gradient element %zu actual %.12g reference %.12g tolerance %.12g\n",i,actual,reference,tolerance);throw std::runtime_error("Gsplat gradient differs from reference");}}
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    for(int mode=0;mode<6;++mode){const auto a=evaluate(GpuBackend::HIP,mode,0,0,true),b=evaluate(GpuBackend::Vulkan,mode,0,0,true);std::fprintf(stderr,"mode %d image\n",mode);compare(a.image,b.image);std::fprintf(stderr,"alpha\n");compare(a.alpha,b.alpha);compare(a.depth,b.depth,2e-3);compare(a.normal,b.normal,2e-3);for(size_t i=0;i<6;++i){std::fprintf(stderr,"gradient %zu\n",i);compare_gradient(a.gradients[i],b.gradients[i]);}std::fprintf(stderr,"density\n");compare(a.density,b.density,3e-5);std::fprintf(stderr,"shares\n");compare(a.shares,b.shares,3e-5);std::printf("PASS Gsplat render mode %d\n",mode);}
    for(unsigned degree=0;degree<=3;++degree){std::fprintf(stderr,"SH degree %u image\n",degree);const auto a=evaluate(GpuBackend::HIP,0,degree),b=evaluate(GpuBackend::Vulkan,0,degree);compare(a.image,b.image);for(size_t i=0;i<6;++i)compare_gradient(a.gradients[i],b.gradients[i]);std::printf("PASS Gsplat SH %u\n",degree);}
    for(int camera:{2,3,4}){std::fprintf(stderr,"camera %d image\n",camera);const auto a=evaluate(GpuBackend::HIP,5,0,camera),b=evaluate(GpuBackend::Vulkan,5,0,camera);compare(a.image,b.image,2e-3);compare(a.depth,b.depth,2e-3);compare(a.normal,b.normal,2e-3);for(size_t i=0;i<6;++i)compare_gradient(a.gradients[i],b.gradients[i]);std::printf("PASS Gsplat camera %d\n",camera);}
    for(int camera:{0,2,3,4}){const auto a=evaluate(GpuBackend::HIP,5,2,camera,true,true),b=evaluate(GpuBackend::Vulkan,5,2,camera,true,true);compare(a.image,b.image,2e-3);compare(a.alpha,b.alpha);compare(a.depth,b.depth,2e-3);compare(a.normal,b.normal,2e-3);for(size_t i=0;i<6;++i)compare_gradient(a.gradients[i],b.gradients[i]);compare(a.density,b.density,3e-5);compare(a.shares,b.shares,3e-5);std::printf("PASS Gsplat cropped camera %d SH2\n",camera);}
    for(int camera:{0,2,3,4}) {
        const auto plain=evaluate(GpuBackend::HIP,0,0,camera,true,true);
        const auto requested=evaluate(GpuBackend::HIP,0,0,camera,true,true,true);
        const auto reference=evaluate(GpuBackend::Vulkan,0,0,camera,true,true,true);
        const auto geometry=evaluate(GpuBackend::HIP,5,0,camera,true,true);
        compare(requested.image,plain.image,1e-7);compare(requested.alpha,plain.alpha,1e-7);
        require(requested.depth.empty()&&requested.normal.empty(),"Ray-only request unexpectedly enabled geometry outputs");
        require(requested.rays.size()==17*19*3,"Ray-only request has incorrect shape");
        compare(requested.rays,reference.rays,2e-5);compare(requested.rays,geometry.rays,1e-7);
        for(size_t i=0;i<6;++i)compare(requested.gradients[i],plain.gradients[i],1e-7);
        std::printf("PASS Gsplat RGB camera rays %d without geometry outputs\n",camera);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"Vulkan Gsplat retained storage");require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP Gsplat retained storage");std::puts("PASS HIP Gsplat: six render modes, RGB/depth/normal/alpha gradients, SH0-3, backgrounds, density/edge/share paths, camera models and CPU ray references");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());(void)shutdown_gpu_backend(GpuBackend::HIP);return 1;}
