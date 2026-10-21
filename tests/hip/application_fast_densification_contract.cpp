/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "training_session_fixture.hpp"
#include <core/tensor_backend.hpp>
#include <lfs/training/ops/registry.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
using namespace lfs;
using namespace lfs::core;
using namespace lfs::training;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
struct RasterComparison {std::vector<float> image,densification;std::array<std::vector<float>,5> parameters;};
static RasterComparison evaluate(const char* module,int width,int height,float gain,bool optimized) {
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",module)==0,"Cannot select HIP module");
    require(_putenv_s("LFS_HIP_FAST_DENSIFICATION_PIXELS",optimized?"1":"0")==0,"Cannot select backward specialization");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",module,1)==0,"Cannot select HIP module");
    require(setenv("LFS_HIP_FAST_DENSIFICATION_PIXELS",optimized?"1":"0",1)==0,"Cannot select backward specialization");
#endif
    const GpuBackendScope scope(GpuBackend::HIP);
    hip_fixture::Fixture fixture;fixture.perturb();constexpr size_t n=hip_fixture::n;
    std::array<Tensor,5> parameters;constexpr size_t attributes[]{3,4,3,3,1};
    for(size_t i=0;i<5;++i)parameters[i]=Tensor::from_vector(fixture.values[i],{n,attributes[i]},Device::GPU);
    auto view=Tensor::eye(4),camera=Tensor::zeros({3},Device::GPU);
    Tensor absent,image,alpha,depth,normal,screen_share;
    gpu_ops::SplatInputs inputs{parameters[0],parameters[2],parameters[1],parameters[4],parameters[3],absent,absent};
    gpu_ops::RenderOutputs outputs{image,alpha,depth,normal};
    gpu_ops::FastParams options{};options.full_image={height,width};options.intrinsics={32,32,width*.5f,height*.5f};options.render_depth=false;
    const auto* ops=training_ops(GpuBackend::HIP).fast;require(ops,"HIP Fast family is unavailable");
    gpu_ops::FastSaved saved{ops->create()};
    std::vector<float> error(size_t(width)*height),gradient(size_t(width)*height*3);
    for(size_t i=0;i<error.size();++i)error[i]=.1f+float(i%13)/13;
    for(size_t i=0;i<gradient.size();++i)gradient[i]=gain*float(int(i%19)-9);
    auto error_tensor=Tensor::from_vector(error,{size_t(height),size_t(width)},Device::GPU);
    auto gradient_tensor=Tensor::from_vector(gradient,{3,size_t(height),size_t(width)},Device::GPU);
    auto density=Tensor::zeros({2,n},Device::GPU);
    std::array<Tensor,6> packed,bounds;
    constexpr unsigned mapping[]{0,2,1,4,3};
    constexpr float rates[]{.00016f,.005f,.001f,.05f,.0025f};
    std::array<gpu_ops::BackwardAdamParam,6> groups{
        gpu_ops::BackwardAdamParam{parameters[0],packed[0],bounds[0],absent,absent,absent,absent},
        gpu_ops::BackwardAdamParam{parameters[2],packed[1],bounds[1],absent,absent,absent,absent},
        gpu_ops::BackwardAdamParam{parameters[1],packed[2],bounds[2],absent,absent,absent,absent},
        gpu_ops::BackwardAdamParam{parameters[4],packed[3],bounds[3],absent,absent,absent,absent},
        gpu_ops::BackwardAdamParam{parameters[3],packed[4],bounds[4],absent,absent,absent,absent},
        gpu_ops::BackwardAdamParam{absent,packed[5],bounds[5],absent,absent,absent,absent}};
    for(size_t i=0;i<5;++i) {
        const auto attr=attributes[mapping[i]];packed[i]=Tensor::zeros({n*attr*4},Device::GPU,DataType::UInt8);bounds[i]=Tensor::zeros({(n+255)/256,4},Device::GPU);
        auto& g=groups[i];g.joint_bits=16;g.primitives=n;g.elements=n*attr;g.attributes=attr;g.enabled=true;
    }
    gpu_ops::BackwardAdam adam{groups,absent,absent};adam.eps=1e-8f;
    RasterComparison result;
    for(int step=1;step<=3;++step) {
        const auto forward=ops->forward(saved,inputs,view,camera,absent,absent,options,outputs,screen_share);
        require(forward.code==gpu_ops::RasterResult::Code::Success&&forward.has_work,"HIP densification forward failed");
        if(step==1)result.image=image.cpu().to_vector();
        for(size_t i=0;i<5;++i){groups[i].step_size=rates[i]/(1-std::pow(.9f,float(step)));groups[i].bc2_sqrt_rcp=1/std::sqrt(1-std::pow(.999f,float(step)));}
        ops->backward(saved,{gradient_tensor,absent,absent,absent},density,error_tensor,absent,absent,adam,DensificationType::MCMC);
    }
    result.densification=density.cpu().to_vector();
    for(size_t i=0;i<5;++i)result.parameters[i]=parameters[i].cpu().to_vector();
    ops->release(saved);ops->release_caches(saved);return result;
}
int main(int argc,char** argv)try {
    require(argc==2,"Argument: absolute HIP training module");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    for(const auto size:{std::array<int,2>{19,17},std::array<int,2>{64,48},std::array<int,2>{1,1}})for(float gain:{.0001f,100000.f}) {
        const auto reference=evaluate(argv[1],size[0],size[1],gain,false),candidate=evaluate(argv[1],size[0],size[1],gain,true);
        for(size_t i=0;i<reference.image.size();++i)require(std::abs(reference.image[i]-candidate.image[i])<2e-6f,"Forward changed during backward specialization");
        double max_error=0;
        for(size_t group=0;group<5;++group)for(size_t i=0;i<reference.parameters[group].size();++i)max_error=std::max(max_error,double(std::abs(reference.parameters[group][i]-candidate.parameters[group][i])));
        require(max_error<3e-5,"MCMC optimized backward changed fused Adam parameters");
        bool nonzero=false;
        for(size_t i=0;i<reference.densification.size();++i) {
            const auto a=reference.densification[i],b=candidate.densification[i];nonzero|=a!=0;
            require(std::isfinite(b)&&std::abs(a-b)<=2e-5f+2e-5f*std::abs(a),"MCMC optimized backward changed density/error statistics");
        }
        require(nonzero,"MCMC comparison did not exercise density writes");
        std::printf("PASS MCMC %dx%d gain %.4g parameter max error %.9g\n",size[0],size[1],gain,max_error);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Densification comparison retained HIP storage");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
