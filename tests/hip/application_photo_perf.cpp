/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <core/gpu_elapsed.hpp>
#include <array>
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

static void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
int main(int argc,char** argv) try {
    check((argc==2 || argc==3) && std::filesystem::path(argv[1]).is_absolute(),"Pass absolute training module");
#ifdef _WIN32
    check(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot select module");
#else
    check(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot select module");
#endif
    {
        GpuBackendScope scope(GpuBackend::HIP);
        const auto& ops=*training::training_ops(GpuBackend::HIP).photometric;
        constexpr size_t h=1440,w=1920,count=3*h*w;
        std::vector<float> input(count),target(count);
        for(size_t i=0;i<count;++i) {
            input[i]=.15f+.7f*float((i*19+3)%997)/997;
            target[i]=.1f+.8f*float((i*31+11)%991)/991;
        }
        auto image=Tensor::from_vector(input,{1,3,h,w},Device::GPU);
        auto truth=Tensor::from_vector(target,{1,3,h,w},Device::GPU);
        if(argc==3) {
            check(std::string(argv[2])=="byte","Unknown target format");
            truth=truth.mul(255.f).to(DataType::UInt8);
        }
        gpu_ops::PhotoSaved saved; saved.backend=ops.create();
        gpu_ops::PhotoParams params; params.path=gpu_ops::PhotoPath::Fused; params.ssim_weight=.2f;
        Tensor loss,gradient,raw_gradient;
        const auto before=gpu_backend_memory_info(GpuBackend::HIP);
        const size_t events=std::getenv("LFS_HIP_PHOTO_EVENT_COUNT") ? std::stoul(std::getenv("LFS_HIP_PHOTO_EVENT_COUNT")) : 2;
        check(events>=2 && events<=65536,"Invalid event count");
        GpuElapsed timer(GpuBackend::HIP,events);
        const auto after=gpu_backend_memory_info(GpuBackend::HIP);
        std::printf("TIMING_POOL events=%zu before_free=%zu after_free=%zu\n",events,size_t(before.free_bytes),size_t(after.free_bytes));
        check(timer.ready(),"Missing HIP timer");
        for(int i=0;i<20;++i) ops.evaluate(saved,image,{},truth,{},params,loss,gradient,raw_gradient);
        check(timer.wait_queue(nullptr),"Warmup wait failed");
        std::vector<double> samples;
        for(int i=0;i<25;++i) {
            check(timer.mark(0,nullptr),"Start mark failed");
            ops.evaluate(saved,image,{},truth,{},params,loss,gradient,raw_gradient);
            check(timer.mark(1,nullptr) && timer.wait_event(1),"End mark failed");
            const auto ms=timer.milliseconds(0,1);
            check(ms.has_value() && std::isfinite(*ms),"Invalid elapsed");
            samples.push_back(*ms);
            std::printf("sample,%d,%.6f\n",i,*ms);
        }
        const auto values=gradient.cpu().to_vector();
        if(const char* path=std::getenv("LFS_HIP_PHOTO_GRADIENT_DUMP")) {
            std::ofstream file(path,std::ios::binary);
            file.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(float)));
            check(bool(file),"Cannot save gradient");
        }
        double sum=0,abs_sum=0; float maximum=0;
        for(float value:values) { check(std::isfinite(value),"Nonfinite gradient"); sum+=value; abs_sum+=std::abs(value); maximum=std::max(maximum,std::abs(value)); }
        std::sort(samples.begin(),samples.end());
        double mean=0; for(double v:samples) mean+=v/samples.size();
        std::printf("SUMMARY mean_ms=%.6f median_ms=%.6f min_ms=%.6f loss=%.9g gradient_sum=%.12g gradient_abs=%.12g gradient_max=%.9g workspace=%zu\n",mean,samples[samples.size()/2],samples.front(),loss.item(),sum,abs_sum,maximum,size_t(ops.workspace_bytes(saved).allocated));
        ops.reset(saved);
    }
    check(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Resources retained");
    std::puts("PASS isolated HIP application photo benchmark");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"FAIL photo benchmark: %s\n",e.what()); return 1; }
