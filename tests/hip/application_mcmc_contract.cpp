/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
int main(int argc,char** argv) try {
    require(argc==2 && std::filesystem::path(argv[1]).is_absolute(),"Pass absolute HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set HIP module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set HIP module");
#endif
    const auto configured=configured_gpu_backend();
    const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.mcmc && !table.fast && !table.photometric && !table.refine && !training::other_tables,"MCMC discovery fell through to another backend");
    require(!gpu_backend_live(GpuBackend::HIP),"Discovery initialized GPU");
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        const auto& ops=*table.mcmc;
        auto integers=Tensor::ones({263},Device::GPU,DataType::Int32);
        require(integers.cpu().to_vector_int()==std::vector<int>(263,1),"Int32 ones incorrect");
        auto ratios=Tensor::from_vector(std::vector<int>{0,1,51,80,2147483647},{5},Device::GPU);
        require(ratios.clamp(1,51).cpu().to_vector_int()==std::vector<int>({1,1,51,51,51}),"Int32 clamp incorrect");
        auto flags=Tensor::ones({263},Device::GPU,DataType::Bool);
        require(flags.logical_not().cpu().to_vector_bool()==std::vector<bool>(263,false),"Bool fill/not incorrect");
        require(flags.logical_and(flags).cpu().to_vector_bool()==std::vector<bool>(263,true),"Bool and incorrect");
        ops.initialize(3); ops.initialize(51);
        auto opacity=Tensor::from_vector(std::vector<float>{.1f,.4f,.8f},{3},Device::GPU);
        auto scales=Tensor::ones({3,3},Device::GPU);
        auto one=Tensor::ones({3},Device::GPU,DataType::Int32);
        auto relocated=Tensor::zeros({3},Device::GPU), relocated_scale=Tensor::zeros({3,3},Device::GPU);
        ops.relocate(opacity,scales,one,relocated,relocated_scale,.005f);
        auto op=opacity.cpu().to_vector(), actual=relocated.cpu().to_vector();
        for(size_t i=0;i<3;++i) require(std::abs(op[i]-actual[i])<1e-6f,"Ratio-one relocation changed opacity");
        for(float x:relocated_scale.cpu().to_vector()) require(std::abs(x-1.f)<2e-6f,"Ratio-one relocation changed scale");
        auto invalid=Tensor::zeros({3},Device::GPU,DataType::Int32);
        bool rejected=false;
        try { ops.relocate(opacity,scales,invalid,relocated,relocated_scale,.005f); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"Invalid relocation ratios accepted");
        constexpr size_t draws=60000;
        auto weights=Tensor::from_vector(std::vector<float>{1,3,0},{3},Device::GPU);
        auto raw_scales=Tensor::from_vector(std::vector<float>{1,2,3,4,5,6,7,8,9},{3,3},Device::GPU);
        auto ids=Tensor::zeros({draws},Device::GPU,DataType::Int64);
        auto sampled_opacity=Tensor::zeros({draws},Device::GPU), sampled_scales=Tensor::zeros({draws,3},Device::GPU);
        Tensor absent;
        ops.sample(weights,opacity,raw_scales,absent,ids,sampled_opacity,sampled_scales,gpu_ops::SampleDomain::All,791);
        const auto selected=ids.cpu().to_vector_int64();
        const auto gathered=sampled_opacity.cpu().to_vector(), gathered_scale=sampled_scales.cpu().to_vector();
        size_t first=0;
        for(size_t i=0;i<draws;++i) {
            require(selected[i]>=0 && selected[i]<2,"Sampler selected zero-weight/out-of-range row");
            first+=selected[i]==0;
            require(gathered[i]==op[selected[i]],"Sampler opacity gather mismatch");
            for(size_t c=0;c<3;++c) {
                const float expected=std::exp(float(selected[i]*3+c+1));
                require(std::abs(gathered_scale[i*3+c]-expected)<=expected*2e-6f,"Sampler scale activation/gather mismatch");
            }
        }
        require(first>14000 && first<16000,"Sampling distribution incorrect");
        ops.sample(weights,opacity,raw_scales,absent,ids,sampled_opacity,sampled_scales,gpu_ops::SampleDomain::All,791);
        require(ids.cpu().to_vector_int64()==selected,"Sampling seed not reproducible");
        auto alive=Tensor::from_vector(std::vector<int>{1},{1},Device::GPU).to(DataType::Int64);
        ops.sample(weights,opacity,raw_scales,alive,ids,sampled_opacity,sampled_scales,gpu_ops::SampleDomain::AliveIndices,791);
        for(auto id:ids.cpu().to_vector_int64()) require(id==1,"Alive sampler selected excluded row");
        auto maximum=Tensor::from_vector(std::vector<float>{1,4,2},{3},Device::GPU);
        auto densification=Tensor::from_vector(std::vector<float>{8,9,10,3,2,7},{2,3},Device::GPU);
        ops.fold_error(maximum,densification);
        require(maximum.cpu().to_vector()==std::vector<float>({3,4,7}),"Error fold incorrect");
        require(densification.cpu().to_vector()==std::vector<float>(6,0),"Densification reset incorrect");
        auto source=Tensor::from_vector(std::vector<int>{0},{1},Device::GPU).to(DataType::Int64);
        auto destination=Tensor::from_vector(std::vector<int>{2},{1},Device::GPU).to(DataType::Int64);
        auto means=raw_scales.clone(), sh0=raw_scales.clone(), rotations=Tensor::ones({3,4},Device::GPU), raw_opacity=opacity.clone();
        auto copy_scales=raw_scales.clone();
        gpu_ops::McmcRows rows{means,sh0,copy_scales,rotations,raw_opacity};
        ops.copy_rows(source,destination,rows);
        for(const auto* tensor:{&means,&sh0,&copy_scales}) {
            const auto values=tensor->cpu().to_vector();
            require(values==std::vector<float>({1,2,3,4,5,6,1,2,3}),"MCMC base-row copy changed wrong rows");
        }
        require(raw_opacity.cpu().to_vector()==std::vector<float>({.1f,.4f,.1f}),"MCMC opacity row copy failed");
        auto changed_scales=Tensor::from_vector(std::vector<float>{-2,-3,-4},{1,3},Device::GPU);
        auto changed_opacity=Tensor::from_vector(std::vector<float>{-5},{1},Device::GPU);
        ops.update_rows(destination,changed_scales,changed_opacity,copy_scales,raw_opacity);
        require(copy_scales.cpu().to_vector()==std::vector<float>({1,2,3,4,5,6,-2,-3,-4}),"MCMC scale row update failed");
        require(raw_opacity.cpu().to_vector()==std::vector<float>({.1f,.4f,-5}),"MCMC opacity row update failed");
        std::vector<float> quaternions(draws*4,0);
        for(size_t i=0;i<draws;++i) quaternions[i*4]=1;
        auto noise_rotations=Tensor::from_vector(quaternions,{draws,4},Device::GPU);
        auto noise_scales=Tensor::zeros({draws,3},Device::GPU);
        auto noise_opacity=Tensor::full({draws},-6.f,Device::GPU);
        auto noise_means=Tensor::zeros({draws,3},Device::GPU);
        std::vector<bool> frozen_values(draws,false);
        std::fill_n(frozen_values.begin(),100,true);
        auto frozen=Tensor::from_vector(frozen_values,{draws},Device::GPU);
        ops.noise(noise_opacity,noise_scales,noise_rotations,frozen,noise_means,893,.1f);
        const auto noise=noise_means.cpu().to_vector();
        double sum=0,squares=0;
        for(size_t i=0;i<noise.size();++i) {
            require(std::isfinite(noise[i]),"Nonfinite MCMC noise");
            if(i<300) require(noise[i]==0,"MCMC noise changed frozen row");
            else { sum+=noise[i]; squares+=double(noise[i])*noise[i]; }
        }
        const double factor=.1/(1+std::exp(100/(1+std::exp(6.))-.5));
        const double variance=squares/(noise.size()-300);
        require(std::abs(sum/(noise.size()-300))<.001 && std::abs(variance-factor*factor)<factor*factor*.03,"MCMC noise statistics incorrect");
        noise_means.zero_();
        ops.noise(noise_opacity,noise_scales,noise_rotations,frozen,noise_means,893,.1f);
        require(noise_means.cpu().to_vector()==noise,"MCMC noise seed not reproducible");
        std::printf("MCMC sampling first-row proportion %.6f\n",double(first)/draws);
    }
    require(configured_gpu_backend()==configured,"MCMC changed default backend");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"MCMC retained live resources");
    std::puts("PASS HIP application MCMC: real Tensor fills/masks, relocation, invalid-ratio rejection, sampling/gather/distribution/reproducibility, alive domain, error fold and teardown");
    return 0;
} catch(const std::exception& error) { std::fprintf(stderr,"HIP MCMC CONTRACT FAILED: %s\n",error.what()); return 1; }
