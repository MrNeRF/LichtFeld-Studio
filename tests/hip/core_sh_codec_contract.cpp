/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_completion.hpp>
#include <core/tensor_execution.hpp>
#include <core/gpu_device_runtime.hpp>
#include <cstdio>
#include <core/tensor_sh.hpp>
#include <core/sh_layout.hpp>
#include <core/detail/tensor_half.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
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
size_t cells(size_t n,unsigned rest,ShFormat format){return format==ShFormat::Canonical?n*rest*3:((n+31)/32*32)*(format==ShFormat::Q16?rest*3:(rest*3+3)/4*4);}
Tensor storage(size_t n,unsigned rest,ShFormat format,Device device,float fill=123.f){return Tensor::full({cells(n,rest,format)},fill,device,format==ShFormat::Float16 || format==ShFormat::Q16?DataType::Float16:DataType::Float32);}
void same(const Tensor& gpu,const Tensor& cpu,float tolerance,const char* message){
    auto a=gpu.cpu();require(a.shape()==cpu.shape() && a.dtype()==cpu.dtype(),"Core SH shape/type mismatch");
    if(a.dtype()==DataType::Float32){auto av=a.to_vector(),bv=cpu.to_vector();float error=0;for(size_t i=0;i<av.size();++i)error=std::max(error,std::abs(av[i]-bv[i]));if(error>tolerance)std::fprintf(stderr,"Core SH maximum error %.9g: %s\n",error,message);require(error<=tolerance,message);}
    else require(std::memcmp(a.data_ptr(),cpu.data_ptr(),a.bytes())==0,message);
}
int main() try {
    const auto configured=configured_gpu_backend();
    require(!gpu_backend_live(GpuBackend::HIP), "HIP was initialized before inventory");
    const int devices=gpu_device_count(GpuBackend::HIP);
    require(devices>0 && gpu_backend_available(GpuBackend::HIP) && gpu_device_count(GpuBackend::HIP)==devices,
            "HIP lazy hardware inventory returned inconsistent availability");
    require(!gpu_backend_live(GpuBackend::HIP) && configured_gpu_backend()==configured,
            "HIP inventory initialized a Tensor context or changed the configured backend");
    {
        const GpuBackendScope scope(GpuBackend::HIP);const auto target=TensorExecutionTarget::default_queue(GpuBackend::HIP);
        constexpr size_t n=777;
        for(unsigned rest:{3u,8u,15u}){
            std::vector<float> values(n*rest*3);for(size_t i=0;i<values.size();++i)values[i]=float(int((i*71)%997)-498)*.0003f;
            auto canonical=Tensor::from_vector(values,{n,rest,3},Device::CPU),gpu_canonical=canonical.to(Device::GPU);
            for(auto format:{ShFormat::Float32,ShFormat::Float16,ShFormat::Q16}){
                auto cpu=storage(n,rest,format,Device::CPU),gpu=storage(n,rest,format,Device::GPU);
                auto cb=Tensor::zeros({(n+255)/256*2},Device::CPU),gb=cb.to(Device::GPU);
                ShCodec encode{.source_format=ShFormat::Canonical,.destination_format=format,.source_rows=n,.destination_rows=n,.count=n,.source_rest=rest,.destination_rest=rest};
                sh_codec(canonical,cpu,encode,nullptr,nullptr,format==ShFormat::Q16?&cb:nullptr);
                sh_codec(gpu_canonical,gpu,encode,nullptr,nullptr,format==ShFormat::Q16?&gb:nullptr);
                if(format!=ShFormat::Q16)same(gpu,cpu,0,"Core SH encode differs from CPU");
                else {same(gb,cb,0,"Core SH Q16 bounds differ");auto raw=gpu.cpu();const auto* a=static_cast<const uint16_t*>(raw.data_ptr());const auto* b=static_cast<const uint16_t*>(cpu.data_ptr());for(size_t i=0;i<raw.numel();++i)require(std::abs(int(a[i])-int(b[i]))<=1,"Core SH Q16 encoding differs by more than one code");}
                for(auto destination_format:{ShFormat::Canonical,ShFormat::Float32,ShFormat::Float16}){
                    auto expected=storage(n,rest,destination_format,Device::CPU),actual=storage(n,rest,destination_format,Device::GPU);
                    // Use the actual encoded GPU values as both inputs: isolates conversion from encoder quantization.
                    auto same_source=gpu.cpu();auto bounds=format==ShFormat::Q16?gb.cpu():Tensor{};
                    ShCodec decode{.source_format=format,.destination_format=destination_format,.source_rows=n,.destination_rows=n,.count=n,.source_rest=rest,.destination_rest=rest,.match_cpu_rounding=true};
                    sh_codec(same_source,expected,decode,nullptr,format==ShFormat::Q16?&bounds:nullptr);
                    sh_codec(gpu,actual,decode,nullptr,format==ShFormat::Q16?&gb:nullptr);
                    same(actual,expected,1e-7f,"Core SH decode/copy differs from CPU");
                }
                for(bool wide:{false,true}){
                    std::vector<int> ids{776,3,0,71,255,256,511,512,42,700,99,333,1};
                    auto cpu_ids=Tensor::from_vector(ids,{ids.size()},Device::CPU);if(wide)cpu_ids=cpu_ids.to(DataType::Int64);auto gpu_ids=cpu_ids.to(Device::GPU);
                    const unsigned dest_rest=rest==15?8:15;auto expected=storage(29,dest_rest,ShFormat::Float32,Device::CPU),actual=storage(29,dest_rest,ShFormat::Float32,Device::GPU);
                    auto same_source=gpu.cpu(),bounds=format==ShFormat::Q16?gb.cpu():Tensor{};
                    ShCodec gather{.source_format=format,.destination_format=ShFormat::Float32,.source_rows=n,.destination_rows=29,.count=ids.size(),.source_rest=rest,.destination_rest=dest_rest,.destination_offset=3,.match_cpu_rounding=true};
                    sh_codec(same_source,expected,gather,&cpu_ids,format==ShFormat::Q16?&bounds:nullptr);
                    sh_codec(gpu,actual,gather,&gpu_ids,format==ShFormat::Q16?&gb:nullptr);
                    same(actual,expected,1e-7f,"Core SH indexed partial gather/preservation differs");
                }
                // Offset ranges must preserve untouched rows and clear only final destination padding.
                auto expected=storage(29,rest,ShFormat::Float32,Device::CPU),actual=storage(29,rest,ShFormat::Float32,Device::GPU);
                auto same_source=gpu.cpu(),bounds=format==ShFormat::Q16?gb.cpu():Tensor{};
                ShCodec range{.source_format=format,.destination_format=ShFormat::Float32,.source_rows=n,.destination_rows=29,.count=13,.source_rest=rest,.destination_rest=rest,.source_offset=7,.destination_offset=16,.match_cpu_rounding=true};
                sh_codec(same_source,expected,range,nullptr,format==ShFormat::Q16?&bounds:nullptr);sh_codec(gpu,actual,range,nullptr,format==ShFormat::Q16?&gb:nullptr);same(actual,expected,1e-7f,"Core SH offset/padding differs");
            }
            auto source=storage(n,rest,ShFormat::Float32,Device::GPU);sh_codec(gpu_canonical,source,{.source_format=ShFormat::Canonical,.destination_format=ShFormat::Float32,.source_rows=n,.destination_rows=n,.count=n,.source_rest=rest,.destination_rest=rest});
            std::vector<int> ids{7,2,23};auto cpu_ids=Tensor::from_vector(ids,{3},Device::CPU),gpu_ids=cpu_ids.to(Device::GPU);
            auto expected=storage(29,rest,ShFormat::Float32,Device::CPU),actual=storage(29,rest,ShFormat::Float32,Device::GPU);
            ShCodec scatter{.source_format=ShFormat::Canonical,.destination_format=ShFormat::Float32,.source_rows=n,.destination_rows=29,.count=3,.source_rest=rest,.destination_rest=rest,.source_offset=9,.scatter=true};
            sh_codec(canonical,expected,scatter,&cpu_ids);sh_codec(gpu_canonical,actual,scatter,&gpu_ids);same(actual,expected,0,"Core SH scatter differs");
            std::vector<int> reversed(n);for(size_t i=0;i<n;++i)reversed[i]=int(n-1-i);auto cp=Tensor::from_vector(reversed,{n},Device::CPU),gp=cp.to(Device::GPU);
            auto alias_expected=source.cpu();auto snapshot=alias_expected.clone();ShCodec alias{.source_rows=n,.destination_rows=n,.count=n,.source_rest=rest,.destination_rest=rest};
            sh_codec(snapshot,alias_expected,alias,&cp);sh_codec(source,source,alias,&gp);same(source,alias_expected,0,"Core SH aliased gather lost snapshot");
            auto empty_cpu=Tensor::empty({0},Device::CPU),empty_gpu=empty_cpu.to(Device::GPU),zeros_cpu=storage(n,rest,ShFormat::Float32,Device::CPU),zeros_gpu=storage(n,rest,ShFormat::Float32,Device::GPU);
            ShCodec zero{.source_format=ShFormat::Canonical,.destination_format=ShFormat::Float32,.source_rows=n,.destination_rows=n,.count=n,.source_rest=0,.destination_rest=rest};sh_codec(empty_cpu,zeros_cpu,zero);sh_codec(empty_gpu,zeros_gpu,zero);same(zeros_gpu,zeros_cpu,0,"Core SH zero-rest expansion differs");
            target.wait();std::printf("Core SH rest %u: four formats, codecs, half, indexed/partial/scatter/alias/padding/zero-rest pass\n",rest);
        }
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"Core SH changed default or leaked ownership");
    std::puts("PASS HIP Core SH codec API: all layouts and mutation contracts against independent CPU implementation");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"CORE SH CONTRACT FAILED: %s\n",e.what());return 1;}
