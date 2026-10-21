/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <lfs/training/ops/geometry.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;using namespace lfs::training;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void near(double a,double b,double tolerance=2e-4){if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tolerance*(1+std::abs(b))){std::fprintf(stderr,"actual %.12g reference %.12g\n",a,b);throw std::runtime_error("Geometry differs from reference");}}
struct Result{std::vector<float> gn,gd,ga,diagnostics;float loss;};
static Result evaluate(GpuBackend backend,unsigned operation,bool weighted,bool rays,bool wrap,bool inactive=false) {
    GpuBackendScope scope(backend);const auto* ops=training_ops(backend).geometry;require(ops,"Missing geometry family");
    constexpr size_t h=23,w=31,n=h*w;std::vector<float> a(n),d(n),t(n),norm(n*3),target(n*3),weights(n),camera(n*3);
    for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){size_t i=y*w+x;a[i]=inactive?0.f:.75f+float(i%11)*.01f;d[i]=a[i]*(2.f+.002f*x+.003f*y);t[i]=.2f+.002f*x+.001f*y;weights[i]=i%23==0?0.f:.5f+float(i%7)*.1f;norm[i]=.15f;norm[n+i]=-.05f;norm[2*n+i]=-.9f;target[i]=.1f;target[n+i]=.02f;target[2*n+i]=-.99f;camera[i*3]=(float(x)+.5f-15.5f)/40;camera[i*3+1]=(float(y)+.5f-11.5f)/41;camera[i*3+2]=1;}
    auto alpha=Tensor::from_vector(a,{h,w},Device::GPU),depth=Tensor::from_vector(d,{h,w},Device::GPU),prior=Tensor::from_vector(t,{h,w},Device::GPU);
    auto normal=Tensor::from_vector(norm,{3,h,w},Device::GPU),nt=Tensor::from_vector(target,{3,h,w},Device::GPU);
    Tensor weight,cr;if(weighted)weight=Tensor::from_vector(weights,{h,w},Device::GPU);if(rays)cr=Tensor::from_vector(camera,{h,w,3},Device::GPU);
    auto gn=Tensor::full({3,h,w},.013f,Device::GPU),gd=Tensor::full({h,w},.017f,Device::GPU),ga=Tensor::full({h,w},.019f,Device::GPU),loss=Tensor::full({1},99.f,Device::GPU);
    auto scratch=Tensor::zeros({kernels::depth_loss_partial_count(n)},Device::GPU);
    kernels::DepthAnchor anchor;anchor.valid=true;anchor.model=weighted?1:0;anchor.scale=.7f;anchor.shift=.1f;anchor.floor=.05f;
    const lfs::gpu_ops::DepthParams dp{.8f,.2f,weighted?1.f/255:0.f,&anchor};const lfs::gpu_ops::GeometryCamera k{40,41,15.5f,11.5f,cr,wrap};
    switch(operation){case 0:ops->depth(depth,alpha,prior,weight,gd,ga,loss,scratch,dp);break;case 1:ops->normal(normal,alpha,nt,weight,gn,loss,scratch,.8f);break;case 2:ops->consistency(normal,depth,alpha,weight,gn,gd,ga,loss,scratch,k,.8f);break;case 3:ops->prior_depth(nt,depth,alpha,weight,gd,ga,loss,scratch,k,.8f);break;}
    auto diagnostic=scratch.cpu().to_vector();diagnostic.resize(operation==0?10:6);
    Result result{gn.cpu().to_vector(),gd.cpu().to_vector(),ga.cpu().to_vector(),diagnostic,loss.cpu().item<float>()};
    if(inactive)near(result.loss,0,0);
    if(operation==1&&!inactive){double total=0,numerator=0;const double cosine=(.15*.1-.05*.02+.9*.99)/(std::sqrt(.15*.15+.05*.05+.9*.9)*std::sqrt(.1*.1+.02*.02+.99*.99));for(size_t i=0;i<n;++i){double aw=a[i]*(weighted?weights[i]:1);total+=aw;numerator+=aw*(1-cosine);}near(result.loss,.8*numerator/std::max(total,1.),2e-6);}
    return result;
}
static void anchor_test() {
    GpuBackendScope scope(GpuBackend::HIP);const auto* ops=training_ops(GpuBackend::HIP).geometry;
    constexpr size_t h=32,w=32;std::vector<float> xyz,prior(h*w);for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){prior[y*w+x]=.2f+float(x)*.003f;float z=2.f+float(y)*.01f;xyz.push_back((float(x)+.25f-16)*z/20);xyz.push_back((float(y)+.25f-16)*z/20);xyz.push_back(z);}
    std::vector<float> identity(16);identity[0]=identity[5]=identity[10]=identity[15]=1;
    auto points=Tensor::from_vector(xyz,{h*w,3},Device::GPU),view=Tensor::from_vector(identity,{4,4},Device::GPU),image=Tensor::from_vector(prior,{h,w},Device::GPU);
    lfs::gpu_ops::AnchorParams p{{20,20,16,16},.01f,{-100,-100,-100},{100,100,100},{}};
    auto samples=ops->collect_anchor_samples(points,view,image,p);require(samples.size()==h*w,"Anchor collector lost visible points");
    std::vector<std::pair<float,float>> expected,actual;for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x)expected.emplace_back(prior[y*w+x],xyz[(y*w+x)*3+2]);for(auto s:samples)actual.emplace_back(s.x,s.y);std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());for(size_t i=0;i<actual.size();++i){near(actual[i].first,expected[i].first,1e-6);near(actual[i].second,expected[i].second,1e-6);}
    p.aabb_hi={-90,-90,-90};require(ops->collect_anchor_samples(points,view,image,p).empty(),"Invisible anchors should be empty");
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    for(unsigned op=0;op<4;++op)for(bool weighted:{false,true})for(bool inactive:{false,true}) {
        const auto a=evaluate(GpuBackend::HIP,op,weighted,op>=2,op>=2,inactive),b=evaluate(GpuBackend::Vulkan,op,weighted,op>=2,op>=2,inactive);
        near(a.loss,b.loss);for(size_t i=0;i<a.gn.size();++i)near(a.gn[i],b.gn[i]);for(size_t i=0;i<a.gd.size();++i){near(a.gd[i],b.gd[i]);near(a.ga[i],b.ga[i]);}for(size_t i=0;i<a.diagnostics.size();++i)near(a.diagnostics[i],b.diagnostics[i],1e-3);
    }
    anchor_test();require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"Vulkan geometry retained storage");require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP geometry retained storage");std::puts("PASS HIP geometry: four loss families, weighted/unweighted, inactive coverage, rays/wrap, CPU cosine and anchor projection references, Vulkan gradient/diagnostic parity");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());(void)shutdown_gpu_backend(GpuBackend::HIP);return 1;}
