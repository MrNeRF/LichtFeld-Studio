/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/nn/models/lpips.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs;
using namespace core;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
std::array<float,3> evaluate(GpuBackend backend,const std::filesystem::path& weights){
    const GpuBackendScope scope(backend);
    constexpr size_t h=32,w=40;
    std::vector<float> a(3*h*w),b(a.size());
    for(size_t i=0;i<a.size();++i){a[i]=float((i*17+i/31)%257)/256.f;b[i]=float((i*11+i/47+13)%257)/256.f;}
    auto x=Tensor::from_vector(a,{3,h,w},Device::GPU),y=Tensor::from_vector(b,{3,h,w},Device::GPU);
    auto loaded=nn::models::Lpips::load(weights,Device::GPU,DataType::Float16,nn::models::InputScaling::Identity,64ull<<20);
    if(!loaded)throw std::runtime_error(format_for_developer(loaded.error()));
    const auto* ops=training::training_ops(backend).lpips;require(ops!=nullptr,"LPIPS application family is missing");loaded->set_dispatch(*ops);
    const auto check=[](const Result<float>& result){if(!result)throw std::runtime_error(format_for_developer(result.error()));require(std::isfinite(*result),"LPIPS full model produced a nonfinite score");return *result;};
    const float self=check(loaded->forward(x,x)),different=check(loaded->forward(x,y)),reverse=check(loaded->forward(y,x)),warm=check(loaded->forward(x,y));
    require(std::abs(self)<1e-7f && different>1e-5f,"LPIPS full model identity/contrast invariant failed");
    require(std::abs(different-reverse)<2e-6f && std::abs(different-warm)<2e-6f,"LPIPS full model symmetry or warm-cache consistency failed");
    if(backend==GpuBackend::HIP) {
        auto mask=Tensor::full({h,w},1.f,Device::GPU);
        const float full_mask=check(loaded->forward(x,y,mask));
        mask.fill_(.25f);
        const float scaled_mask=check(loaded->forward(x,y,mask));
        require(std::abs(full_mask-different)<2e-6f && std::abs(scaled_mask-full_mask)<2e-6f,
                "HIP LPIPS mask pyramid changed full-image or normalized constant-mask scores");
        std::vector<float> spatial(h*w);for(size_t i=0;i<spatial.size();++i)spatial[i]=float(i%17)/16.f;
        mask=Tensor::from_vector(spatial,{h,w},Device::GPU);
        const float masked=check(loaded->forward(x,y,mask)),masked_reverse=check(loaded->forward(y,x,mask));
        require(masked>0 && std::abs(masked-masked_reverse)<2e-6f,"HIP masked LPIPS symmetry failed");
        std::printf("HIP masked VGG: full=%.9g scaled=%.9g spatial=%.9g reverse=%.9g\n",full_mask,scaled_mask,masked,masked_reverse);
    }
    std::printf("%s full VGG: self=%.9g different=%.9g reverse=%.9g warm=%.9g weights=%zu arena=%zu workspace=%zu\n",gpu_backend_name(backend),self,different,reverse,warm,loaded->weights_bytes(),loaded->arena_bytes(),loaded->workspace_bytes());
    loaded->release_activations();return {self,different,warm};
}
int main(int argc,char** argv)try{
    require(argc==2 || argc==3,"Arguments: LPIPS weights [compare-vulkan]");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select explicit HIP backend");
    const auto hip=evaluate(GpuBackend::HIP,argv[1]);
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP full LPIPS model retained live storage");
    if(argc==3){
        require(std::string_view(argv[2])=="compare-vulkan","Unknown reference mode");
        const auto reference=evaluate(GpuBackend::Vulkan,argv[1]);
        // Different half-precision convolution reduction orders may round at
        // intermediate layers. Bound final score drift to 0.1% plus 1e-5.
        require(std::abs(hip[1]-reference[1])<=1e-5f+1e-3f*std::abs(reference[1]),"HIP full LPIPS score drift exceeds the reference bound");
        require(bool(shutdown_gpu_backend(GpuBackend::Vulkan)),"Reference LPIPS model retained live storage");
    }
    require(!gpu_backend_live(GpuBackend::CUDA),"HIP LPIPS model initialized a CUDA Tensor context");
    std::puts("PASS full application HIP LPIPS model: real weights, identity, contrast, symmetry and warm-cache consistency");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP LPIPS MODEL FAILED: %s\n",e.what());return 1;}
