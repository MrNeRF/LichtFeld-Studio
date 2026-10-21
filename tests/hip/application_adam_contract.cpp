/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_execution.hpp>
#include <algorithm>
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
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
Tensor bytes_to_gpu(std::vector<uint8_t>& values) {
    return Tensor::from_blob(values.data(),{values.size()},Device::CPU,DataType::UInt8).to(Device::GPU);
}
template<int Bits>
void reset_test(const gpu_ops::AdamOps& ops,bool swizzled) {
    using Codec=training::joint_adam::Codec<Bits>;
    constexpr size_t n=777,attributes=3,blocks=(n+255)/256;
    const size_t cells=swizzled ? ((n+31)/32)*32*attributes*4 : n*attributes;
    std::vector<uint8_t> initial(cells*(Bits==16 ? 4 : 2),0xa5);
    auto cell_for=[&](size_t row,size_t attribute) { return swizzled ? ((row/32)*(attributes*32)+(attribute/4)*32+row%32)*4+attribute%4 : row*attributes+attribute; };
    for(size_t row=0;row<n;++row) for(size_t a=0;a<(swizzled ? attributes*4 : attributes);++a)
        Codec::encode_us(initial.data(),cell_for(row,a),2.f,3.f,1.f,3.f,2.f,5.f);
    std::vector<float> bound_values(blocks*4);
    for(size_t i=0;i<blocks;++i) { bound_values[i*4]=1; bound_values[i*4+1]=3; bound_values[i*4+2]=2; bound_values[i*4+3]=5; }
    auto packed=bytes_to_gpu(initial),bounds=Tensor::from_vector(bound_values,{blocks,4},Device::GPU);
    auto indices=Tensor::from_vector(std::vector<int>{-1,0,17,int(n)-1,int(n)+1},{5},Device::GPU).to(DataType::Int64);
    gpu_ops::JointCodecParams params{swizzled ? gpu_ops::JointLayout::SwizzledSH : gpu_ops::JointLayout::Rows,int(n),int(attributes),Bits};
    ops.encode_zero(packed,bounds,indices,params);
    const auto actual=packed.cpu().to_vector_uint8(); const auto mm=bounds.cpu().to_vector();
    for(size_t row=0;row<n;++row) for(size_t a=0;a<(swizzled ? attributes*4 : attributes);++a) {
        const size_t cell=cell_for(row,a),block=row/256;
        float u=0,s=0; Codec::decode_us(actual.data(),cell,mm[block*4],mm[block*4+1],mm[block*4+2],mm[block*4+3],u,s);
        if(row==0 || row==17 || row==n-1) require(u==0 && s==0,"Moment reset failed to encode true zero");
        else {
            float old_u=0,old_s=0; Codec::decode_us(initial.data(),cell,1,3,2,5,old_u,old_s);
            require(std::abs(old_u-u)<(Bits==16 ? .0001f : .02f) && std::abs(old_s-s)<(Bits==16 ? .0001f : .02f),"Moment reset changed unrelated decoded cells");
        }
        if(block==1 || block==2)
            require(std::equal(actual.begin()+cell*Codec::kBytesPerCell,actual.begin()+(cell+1)*Codec::kBytesPerCell,initial.begin()+cell*Codec::kBytesPerCell),"Moment reset rewrote untouched block");
    }
    if(swizzled) for(size_t row=n;row<((n+31)/32)*32;++row) for(size_t a=0;a<attributes*4;++a) {
        const auto offset=cell_for(row,a)*Codec::kBytesPerCell;
        require(std::equal(actual.begin()+offset,actual.begin()+offset+Codec::kBytesPerCell,initial.begin()+offset),"Moment reset changed capacity padding");
    }
}
int main(int argc,char** argv) try {
    require(argc==2 && std::filesystem::path(argv[1]).is_absolute(),"Pass absolute HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set HIP module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set HIP module");
#endif
    const auto configured=configured_gpu_backend(); const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.adam && !table.fast && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"Adam discovery initialized another backend");
    {
        const GpuBackendScope scope(GpuBackend::HIP); const auto& ops=*table.adam;
        auto arena=Tensor::zeros({64},Device::GPU);
        const auto execution=TensorExecutionTarget::default_queue(GpuBackend::HIP);
        auto borrowed=Tensor::from_blob(static_cast<char*>(arena.data_ptr())+16,{7},Device::GPU,DataType::Float32,execution);
        borrowed.fill_(3.f);
        const auto arena_values=arena.cpu().to_vector();
        for(size_t i=0;i<arena_values.size();++i) require(arena_values[i]==(i>=4 && i<11 ? 3.f : 0.f),"HIP borrowed arena view offset incorrect");
        bool rejected=false;
        try { (void)Tensor::from_blob(static_cast<char*>(arena.data_ptr())+arena.bytes()-4,{2},Device::GPU,DataType::Float32,execution); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"HIP raw view accepted an out-of-allocation span");
        float host_pointer=0; rejected=false;
        try { (void)Tensor::from_blob(&host_pointer,{1},Device::GPU,DataType::Float32,execution); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"HIP raw view accepted a non-HIP pointer");
        constexpr size_t n=263,attributes=3,cells=n*attributes,blocks=(n+255)/256;
        std::vector<float> host(cells),grad(cells);
        std::vector<bool> frozen(n,false),crop(n,false);
        for(size_t i=0;i<cells;++i) { host[i]=float(i%97)/97; grad[i]=float(int(i%53)-26)*.01f; }
        for(size_t i=0;i<n;++i) { frozen[i]=i%7==0; crop[i]=i%11==0; }
        auto parameter=Tensor::from_vector(host,{n,attributes},Device::GPU),gradient=Tensor::from_vector(grad,{n,attributes},Device::GPU);
        auto packed=Tensor::zeros({cells*4},Device::GPU,DataType::UInt8),bounds=Tensor::zeros({blocks,4},Device::GPU);
        auto frozen_mask=Tensor::from_vector(frozen,{n},Device::GPU),crop_mask=Tensor::from_vector(crop,{n},Device::GPU); Tensor absent;
        gpu_ops::AdamMasks masks{frozen_mask,crop_mask,absent,absent,absent}; gpu_ops::AdamHyper hyper; gpu_ops::AdamModifiers modifiers; modifiers.cropbox_lr_scale=.25f;
        gpu_ops::JointStep step{parameter,packed,bounds,gradient,int(n),int(attributes),16,.01f,10.f,float(1/std::sqrt(1.-double(hyper.beta2)))};
        std::vector<uint8_t> old_packed(cells*4,0); std::vector<float> old_bounds(blocks*4,0);
        for(unsigned iteration=0;iteration<3;++iteration) {
            const auto before=parameter.cpu().to_vector(); ops.step_batch(std::span(&step,1),masks,hyper,modifiers);
            const auto after=parameter.cpu().to_vector(); double maximum=0;
            for(size_t i=0;i<cells;++i) {
                const size_t row=i/attributes,b=row/256;
                float m=0,v=0;
                training::joint_adam::Codec<16>::decode_g1g2(old_packed.data(),i,old_bounds[b*4],old_bounds[b*4+1],old_bounds[b*4+2],old_bounds[b*4+3],m,v);
                double expected=before[i];
                if(!frozen[row]) {
                    m=hyper.beta1*m+(1-hyper.beta1)*grad[i]; v=hyper.beta2*v+(1-hyper.beta2)*grad[i]*grad[i];
                    expected-=step.lr*(crop[row] ? .25f : 1.f)*step.bc1_rcp*m/(std::sqrt(v)*step.bc2_sqrt_rcp+hyper.eps);
                }
                maximum=std::max(maximum,std::abs(after[i]-expected));
                require(std::abs(after[i]-expected)<3e-6,"Standalone Adam differs from CPU moment/update reference");
                if(frozen[row]) require(after[i]==before[i],"Frozen Adam parameter changed");
            }
            old_packed=packed.cpu().to_vector_uint8(); old_bounds=bounds.cpu().to_vector();
            std::printf("Adam iteration %u max parameter error %.3g\n",iteration,maximum);
        }
        reset_test<8>(ops,false); reset_test<16>(ops,false); reset_test<8>(ops,true); reset_test<16>(ops,true);
        for(unsigned mode=0;mode<3;++mode) {
            constexpr size_t rows=37,slots=3,padded=64,value_cells=9,sh_cells=padded*slots*4;
            auto values=Tensor::full({sh_cells},.125f,Device::GPU);
            auto sh_gradient=Tensor::full({sh_cells},.1f,Device::GPU);
            auto sh_packed=Tensor::zeros({sh_cells*2},Device::GPU,DataType::UInt8),sh_bounds=Tensor::zeros({4},Device::GPU);
            Tensor value_bounds;
            if(mode==1) values=values.to(DataType::Float16);
            if(mode==2) { values=Tensor::zeros({padded*value_cells*2},Device::GPU,DataType::UInt8); value_bounds=Tensor::full({2},.125f,Device::GPU); }
            gpu_ops::AdamMasks no_masks{absent,absent,absent,absent,absent};
            gpu_ops::ShStepParams params{int(rows),int(slots),4,mode ? 16 : 0,mode==2 ? int(value_cells) : 0,.1f,step.bc2_sqrt_rcp};
            ops.step_sh(values,sh_packed,sh_bounds,value_bounds,sh_gradient,no_masks,hyper,{},params);
            if(mode==2) {
                const auto codes=values.cpu().to_vector_uint8(); const auto ranges=value_bounds.cpu().to_vector();
                for(size_t row=0;row<rows;++row) for(size_t cell=0;cell<value_cells;++cell) {
                    const size_t index=(row/32)*(value_cells*32)+cell*32+row%32;
                    const unsigned code=unsigned(codes[index*2])|(unsigned(codes[index*2+1])<<8);
                    const float actual=ranges[0]+(ranges[1]-ranges[0])*(float(code)/65535);
                    require(std::abs(actual-.115f)<3e-6,"Q16 SH Adam update differs");
                }
            } else {
                const auto actual=values.to(DataType::Float32).cpu().to_vector();
                for(size_t row=0;row<rows;++row) for(size_t cell=0;cell<value_cells;++cell) {
                    const size_t index=((row/32)*(slots*32)+(cell/4)*32+row%32)*4+cell%4;
                    require(std::abs(actual[index]-.115f)<(mode==1 ? .0001f : 3e-6f),"FP32/F16 SH Adam update differs");
                }
            }
        }
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"Adam changed default or leaked resources");
    std::puts("PASS HIP application Adam: CPU update/moment reference, frozen/crop rows, 8/16-bit row/SH reset with untouched-block and padding preservation, FP32/F16/Q16 SH and teardown");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"HIP ADAM CONTRACT FAILED: %s\n",e.what()); return 1; }
