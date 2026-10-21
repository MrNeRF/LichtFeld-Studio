/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/sh_layout.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_execution.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    require(argc==2,"Pass absolute HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set module");
#endif
    const auto configured=configured_gpu_backend();const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.extra_loss && !table.fast && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"Extra-loss discovery initialized GPU");
    {
        GpuBackendScope scope(GpuBackend::HIP);const auto& ops=*table.extra_loss;
        for(size_t n:{size_t(263),size_t(10007),size_t(300001)})for(auto kind:{gpu_ops::Regularizer::Scale,gpu_ops::Regularizer::Opacity}){
            std::vector<float> raw(n);for(size_t i=0;i<n;++i)raw[i]=float(int((i*73)%107)-80)*.1f;
            auto input=Tensor::from_vector(raw,{n},Device::GPU),gradient=Tensor::full({n},.125f,Device::GPU),loss=Tensor::full({1},123.f,Device::GPU),scratch=Tensor::empty({1024},Device::GPU);
            ops.regularize(input,gradient,loss,scratch,kind,.37f);
            double sum=0;auto actual=gradient.cpu().to_vector();float max_gradient=0;
            for(size_t i=0;i<n;++i){double activation=kind==gpu_ops::Regularizer::Scale?std::exp(double(raw[i])):1/(1+std::exp(-double(raw[i])));sum+=activation;double derivative=kind==gpu_ops::Regularizer::Scale?activation:activation*(1-activation);double expected=.125+double(.37f)*derivative/n;max_gradient=std::max(max_gradient,float(std::abs(actual[i]-expected)));}
            const double expected=double(.37f)*sum/n;const double error=std::abs(loss.cpu().to_vector()[0]-expected);
            require(error<3e-7 && max_gradient<3e-8,"Regularizer loss/gradient differs from CPU derivative");
            auto before=actual;ops.regularize(input,gradient,loss,scratch,kind,0);require(gradient.cpu().to_vector()==before,"Zero regularizer changed gradient");
            Tensor no_gradient;ops.regularize(input,no_gradient,loss,scratch,kind,.37f);require(std::abs(loss.cpu().to_vector()[0]-expected)<3e-7,"Value-only regularizer differs");
            bool rejected=false;auto small=Tensor::empty({1},Device::GPU);if(n>256)try{ops.regularize(input,gradient,loss,small,kind,.37f);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Undersized reduction workspace accepted");
            std::printf("Extra loss n=%zu kind=%d: loss error %.9g, gradient error %.9g\n",n,int(kind),error,max_gradient);
        }

    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"Extra loss changed backend/default or leaked buffers");
    std::puts("PASS HIP application extra loss: scale/opacity CPU loss and derivatives, grid-stride reduction, value-only/zero-weight, workspace guard");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"EXTRA LOSS CONTRACT FAILED: %s\n",e.what());return 1;}
