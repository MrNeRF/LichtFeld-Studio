/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/nn/models/moge2.hpp>
#include <core/tensor_backend.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
using namespace lfs::core;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv)try {
    require(argc==3,"Arguments: MoGe weights, existing reference fixture");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    std::ifstream file(argv[2]);require(bool(file),"Cannot open reference fixture");
    const auto fixture=nlohmann::json::parse(file);
    const size_t h=fixture.at("input_shape")[2],w=fixture.at("input_shape")[3],plane=h*w;
    std::vector<float> image(3*plane);
    for(size_t y=0;y<h;++y)for(size_t x=0;x<w;++x){const size_t i=y*w+x;
        image[i]=float(x)/float(std::max<size_t>(w-1,1));image[plane+i]=float(y)/float(std::max<size_t>(h-1,1));
        image[2*plane+i]=float(x+y)/float(std::max<size_t>(w+h-2,1));}
    {
        auto input=Tensor::from_vector(image,{1,3,h,w},Device::GPU);
        auto model=nn::models::Moge2::load(argv[1],Device::GPU,DataType::Float16);
        if(!model)throw std::runtime_error(lfs::format_for_developer(model.error()));
        auto output=model->forward(input,fixture.value("num_tokens",400));
        if(!output)throw std::runtime_error(lfs::format_for_developer(output.error()));
        auto normal=output->normal.to(DataType::Float32).cpu();
        const auto expected=fixture.at("full").at("normal").at("values").get<std::vector<float>>();
        require(normal.numel()==expected.size(),"MoGe normal shape differs from fixture");
        float maximum=0;double squares=0;
        for(size_t i=0;i<expected.size();++i){const float value=normal.ptr<float>()[i];require(std::isfinite(value),"MoGe normal is nonfinite");
            const float delta=value-expected[i];maximum=std::max(maximum,std::abs(delta));squares+=double(delta)*delta;}
        std::printf("HIP MoGe real weights: normal max=%.9g RMS=%.9g weights=%zu arena=%zu workspace=%zu\n",maximum,std::sqrt(squares/expected.size()),model->weights_bytes(),model->arena_bytes(),model->workspace_bytes());
        // Same Half-vs-reference bound as the existing MoGe model test.
        require(maximum<=5e-3f,"HIP MoGe normal exceeds existing FP16 reference tolerance");
        for(const auto* tensor:{&output->points,&output->mask,&output->metric_scale}) {
            auto observed=tensor->to(DataType::Float32).cpu();
            for(size_t i=0;i<observed.numel();++i)require(std::isfinite(observed.ptr<float>()[i]),"MoGe point/mask/scale is nonfinite");
        }
        auto mask=output->mask.to(DataType::Float32).cpu();
        for(size_t i=0;i<mask.numel();++i)require(mask.ptr<float>()[i]>=0 && mask.ptr<float>()[i]<=1,"MoGe mask is outside probability range");
        auto scale=output->metric_scale.to(DataType::Float32).cpu();
        for(size_t i=0;i<scale.numel();++i)require(scale.ptr<float>()[i]>0,"MoGe metric scale is not positive");
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"MoGe retained HIP model storage");
    require(!gpu_backend_live(GpuBackend::CUDA),"HIP MoGe initialized CUDA");
    std::puts("PASS HIP MoGe: real weights, existing FP16 normal reference, finite points/depth/mask/scale, native-only backend");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP MOGE FAILED: %s\n",e.what());return 1;}
