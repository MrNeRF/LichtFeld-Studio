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
#include <limits>
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
void near(float actual,float expected,const char* message) { require(std::abs(actual-expected)<=2e-6f*std::max(1.f,std::abs(expected)),message); }
int main(int argc,char** argv) try {
    require(argc==2 && std::filesystem::path(argv[1]).is_absolute(),"Pass absolute HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set HIP module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set HIP module");
#endif
    const auto configured=configured_gpu_backend();
    const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.refine && !table.mcmc && !table.fast && !training::other_tables,"Refinement discovery fell through");
    require(!gpu_backend_live(GpuBackend::HIP),"Discovery initialized GPU");
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        const auto& ops=*table.refine;
        const float nan=std::numeric_limits<float>::quiet_NaN();
        for(const auto& input:std::vector<std::vector<float>>{{nan,0,-2,1,3,5,7},{0,-2,nan},{2,4,6,8},{4}}) {
            auto values=Tensor::from_vector(input,{input.size()},Device::GPU);
            auto positive=input;
            std::erase_if(positive,[](float v){ return !(v>0); });
            std::sort(positive.begin(),positive.end());
            const float median=positive.empty() ? 0 : positive[positive.size()/2];
            ops.normalize_positive_median(values);
            const auto result=values.cpu().to_vector();
            for(size_t i=0;i<input.size();++i) {
                const float expected=median==0 ? 0 : (std::isnan(input[i]) ? 0 : input[i]/std::max(median,1e-9f));
                near(result[i],expected,"Positive median normalization differs from exact sort");
            }
        }
        std::vector<float> large(10007);
        for(size_t i=0;i<large.size();++i) large[i]=i%11==0 ? -1.f : float((i*7919)%173)+.25f;
        auto values=Tensor::from_vector(large,{large.size()},Device::GPU);
        auto sorted=large; std::erase_if(sorted,[](float x){return !(x>0);}); std::sort(sorted.begin(),sorted.end());
        const float median=sorted[sorted.size()/2];
        ops.normalize_positive_median(values);
        const auto normalized=values.cpu().to_vector();
        for(size_t i=0;i<large.size();++i) near(normalized[i],large[i]/median,"Multi-block radix median incorrect");
        auto boolean=Tensor::from_vector(std::vector<bool>{true,false,true},{3},Device::GPU);
        auto floats=Tensor::from_vector(std::vector<float>{-1,0,2,3,nan},{5},Device::GPU);
        auto counts=Tensor::zeros({4},Device::GPU,DataType::Int64); Tensor absent;
        ops.counts(boolean,absent,floats,absent,counts);
        require(counts.cpu().to_vector_int64()==std::vector<int64_t>({2,0,2,0}),"Packed refinement counts incorrect");
        auto rotations=Tensor::from_vector(std::vector<float>{1,0,0,0,0,0,0,0,1,0,0,0},{3,4},Device::GPU);
        auto opacity=Tensor::from_vector(std::vector<float>{.8f,.8f,.01f},{3},Device::GPU);
        auto mask=Tensor::zeros({3},Device::GPU,DataType::Bool);
        ops.dead_mask(opacity,rotations,mask,.05f);
        require(mask.cpu().to_vector_bool()==std::vector<bool>({false,true,true}),"Dead mask incorrect");
        ops.rotation_mask(rotations,mask);
        require(mask.cpu().to_vector_bool()==std::vector<bool>({false,true,false}),"Rotation mask incorrect");
        auto shares=Tensor::from_vector(std::vector<float>{.8f,.8f,.2f},{3},Device::GPU);
        auto error=Tensor::from_vector(std::vector<float>{4,4,4},{3},Device::GPU);
        auto frozen=Tensor::from_vector(std::vector<bool>{true,false,false},{3},Device::GPU);
        auto scores=Tensor::zeros({3},Device::GPU);
        ops.oversize_scores(error,shares,frozen,scores,.5f);
        const auto score=scores.cpu().to_vector(); near(score[0],0,"Frozen score changed"); near(score[1],3.2f,"Oversize score incorrect"); near(score[2],0,"Undersize score incorrect");
        auto scales=Tensor::from_vector(std::vector<float>{0,1,2,0,1,2,0,1,2},{3,3},Device::GPU);
        ops.clip_scales(scales,shares,frozen,.5f);
        const auto clipped=scales.cpu().to_vector();
        for(size_t i=0;i<9;++i) near(clipped[i],float(i%3)-(i==5 ? std::log(1.5f) : 0.f),"Scale clipping changed wrong axis/frozen row");
        auto means=Tensor::zeros({3,3},Device::GPU), sh0=Tensor::ones({3,3},Device::GPU), raw_opacity=Tensor::zeros({3},Device::GPU);
        rotations=Tensor::from_vector(std::vector<float>{1,0,0,0,1,0,0,0,1,0,0,0},{3,4},Device::GPU);
        scales=Tensor::zeros({3,3},Device::GPU);
        auto child_means=Tensor::zeros({1,3},Device::GPU), child_rotations=Tensor::zeros({1,4},Device::GPU), child_scales=Tensor::zeros({1,3},Device::GPU);
        auto child_sh0=Tensor::zeros({1,3},Device::GPU), child_opacity=Tensor::zeros({1},Device::GPU);
        auto index=Tensor::from_vector(std::vector<int>{1},{1},Device::GPU).to(DataType::Int64);
        gpu_ops::RefineOutputs parents{means,rotations,scales,sh0,raw_opacity}, children{child_means,child_rotations,child_scales,child_sh0,child_opacity};
        ops.split(parents,children,index);
        near(means.cpu().to_vector()[3],.5f,"Split parent offset incorrect"); near(child_means.cpu().to_vector()[0],-.5f,"Split child offset incorrect");
        require(child_rotations.cpu().to_vector()==std::vector<float>({1,0,0,0}) && child_sh0.cpu().to_vector()==std::vector<float>(3,1),"Split attributes not copied");
        near(child_scales.cpu().to_vector()[0],std::log(.5f),"Split scale incorrect");
        near(child_opacity.cpu().to_vector()[0],std::log(.3f/.7f),"Split opacity incorrect");
        auto target=Tensor::from_vector(std::vector<int>{2},{1},Device::GPU).to(DataType::Int64);
        auto free=Tensor::ones({3},Device::GPU,DataType::Bool);
        const gpu_ops::RefineInputs source{child_means,child_rotations,child_scales,child_sh0,child_opacity};
        ops.fill_slots(target,source,parents,free);
        near(means.cpu().to_vector()[6],-.5f,"Slot fill means incorrect");
        require(free.cpu().to_vector_bool()==std::vector<bool>({true,true,false}),"Slot fill changed wrong free-mask rows");
    }
    require(configured_gpu_backend()==configured,"Refinement changed default backend");
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Refinement leaked resources");
    std::puts("PASS HIP application refinement: exact positive median, packed counts, masks, frozen clipping/scores, split, slot reuse and complete teardown");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"HIP REFINEMENT CONTRACT FAILED: %s\n",e.what()); return 1; }
