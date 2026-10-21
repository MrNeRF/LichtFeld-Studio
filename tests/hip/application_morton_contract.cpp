/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/registry.hpp"
#include <core/sh_layout.hpp>
#include <core/splat_data.hpp>
#include <core/tensor_sh.hpp>
#include <core/detail/tensor_half.hpp>
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_morton_api.h"
#include <core/sh_value_quant.hpp>
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include <numeric>
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
uint32_t spread(uint32_t v){uint32_t result=0;for(unsigned bit=0;bit<10;++bit)result|=((v>>bit)&1)<<(bit*3);return result;}
size_t moment_cell(size_t row,size_t cell,unsigned slots){return ((row/32)*(slots*32)+(cell/4)*32+row%32)*4+cell%4;}
template<int Bits>void joint_contract(const gpu_ops::MortonOps& ops,gpu_ops::JointLayout layout){
    using Codec=training::joint_adam::Codec<Bits>;constexpr size_t n=777;const unsigned width=layout==gpu_ops::JointLayout::Rows?6:12,cells=layout==gpu_ops::JointLayout::Rows?width:width*4;
    const size_t physical=(layout==gpu_ops::JointLayout::Rows?n:sh_swizzled_padded_n(n))*cells;
    std::vector<uint8_t> packed(physical*Codec::kBytesPerCell);std::vector<float> bounds((n+255)/256*4);
    for(size_t block=0;block<(n+255)/256;++block){bounds[block*4]=-2-float(block);bounds[block*4+1]=2+float(block);bounds[block*4+2]=float(block)*.1f;bounds[block*4+3]=3+float(block)*.2f;}
    for(size_t row=0;row<n;++row)for(unsigned cell=0;cell<cells;++cell){float* mm=bounds.data()+row/256*4;float u=float(int((row*73+cell*37)%313)-156)/100,s=float((row*31+cell*7)%197)/100;size_t at=layout==gpu_ops::JointLayout::Rows?row*cells+cell:moment_cell(row,cell,width);Codec::encode_us(packed.data(),at,u,s,mm[0],mm[1],mm[2],mm[3]);}
    std::vector<int64_t> order(n);for(size_t i=0;i<n;++i)order[i]=int64_t(n-1-i);
    auto input=Tensor::from_blob(packed.data(),{packed.size()},Device::CPU,DataType::UInt8).to(Device::GPU),mm=Tensor::from_vector(bounds,{bounds.size()},Device::GPU);
    auto perm=Tensor::from_blob(order.data(),{n},Device::CPU,DataType::Int64).to(Device::GPU),output=Tensor::zeros({packed.size()},Device::GPU,DataType::UInt8),next_mm=Tensor::empty({bounds.size()},Device::GPU);
    gpu_ops::JointCodecParams p{layout,int(n),int(width),Bits};ops.permute_joint(input,mm,perm,output,next_mm,p);
    auto bytes=output.cpu();auto new_bounds=next_mm.cpu().to_vector();const auto* out=bytes.ptr<uint8_t>();
    for(size_t row=0;row<n;++row)for(unsigned cell=0;cell<cells;++cell){size_t from=size_t(order[row]);size_t src=layout==gpu_ops::JointLayout::Rows?from*cells+cell:moment_cell(from,cell,width),dst=layout==gpu_ops::JointLayout::Rows?row*cells+cell:moment_cell(row,cell,width);float a,b,x,y;auto* sm=bounds.data()+from/256*4;auto* dm=new_bounds.data()+row/256*4;Codec::decode_us(packed.data(),src,sm[0],sm[1],sm[2],sm[3],a,b);Codec::decode_us(out,dst,dm[0],dm[1],dm[2],dm[3],x,y);float step_u=(dm[1]-dm[0])/Codec::kQMax,step_s=(dm[3]-dm[2])/Codec::kQMax;require(std::abs(a-x)<=step_u*.51f+2e-6f && std::abs(b-y)<=step_s*.51f+2e-6f,"Morton moment decode/reencode changed values beyond quantization");}
    if(layout==gpu_ops::JointLayout::SwizzledSH){auto grouped=input.clone(),group_bounds=Tensor::empty({bounds.size()},Device::GPU);const size_t slot_bytes=sh_swizzled_padded_n(n)*4*Codec::kBytesPerCell;auto scratch=Tensor::empty({slot_bytes*2},Device::GPU,DataType::UInt8);ops.permute_joint_grouped(grouped,mm,perm,group_bounds,scratch,p);auto actual=grouped.cpu();require(std::memcmp(actual.data_ptr(),bytes.data_ptr(),bytes.bytes())==0,"Grouped and full Morton moments differ");require(group_bounds.cpu().to_vector()==new_bounds,"Grouped Morton bounds differ");auto small=Tensor::empty({slot_bytes-1},Device::GPU,DataType::UInt8);bool rejected=false;try{ops.permute_joint_grouped(grouped,mm,perm,group_bounds,small,p);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Morton accepted undersized group workspace");}
}
int main(int argc,char** argv) try {
    require(argc==2,"Pass absolute HIP module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set module");
#endif
    const auto configured=configured_gpu_backend();const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.morton && table.sh && table.session && !table.fast && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"Morton discovery initialized GPU");
    {
        const GpuBackendScope scope(GpuBackend::HIP);const auto execution=TensorExecutionTarget::default_queue(GpuBackend::HIP);const TensorExecutionTarget::Scope queue(execution);const auto& ops=*table.morton;
        constexpr size_t n=777;std::vector<float> means(n*3);for(size_t i=0;i<n;++i){means[i*3]=float((i*73)%1009)/100;means[i*3+1]=float((i*71)%1013)/200;means[i*3+2]=float((i*67)%1019)/300;}
        means[3]=means[0];means[4]=means[1];means[5]=means[2];
        std::array<float,3> lo{1e30f,1e30f,1e30f},hi{-1e30f,-1e30f,-1e30f};for(size_t i=0;i<n;++i)for(unsigned c=0;c<3;++c){lo[c]=std::min(lo[c],means[i*3+c]);hi[c]=std::max(hi[c],means[i*3+c]);}
        std::vector<uint32_t> codes(n);for(size_t i=0;i<n;++i){uint32_t q[3];for(unsigned c=0;c<3;++c){float t=(means[i*3+c]-lo[c])*(1024.f/(hi[c]-lo[c]));q[c]=t>0?std::min(1023u,uint32_t(t)):0;}codes[i]=spread(q[0])+(spread(q[1])<<1)+(spread(q[2])<<2);}
        std::vector<int64_t> expected(n);std::iota(expected.begin(),expected.end(),0);std::stable_sort(expected.begin(),expected.end(),[&](auto a,auto b){return codes[a]<codes[b];});
        auto positions=Tensor::from_vector(means,{n,3},Device::GPU);auto permutation=ops.permutation(positions);auto actual=permutation.cpu();require(std::equal(expected.begin(),expected.end(),actual.ptr<int64_t>()),"Morton integer key order/stable ties differ from CPU");
        auto same=Tensor::full({n,3},1.f,Device::GPU);auto tied=ops.permutation(same).cpu();for(size_t i=0;i<n;++i)require(tied.ptr<int64_t>()[i]==int64_t(i),"Degenerate Morton axes lost stable ties");
        auto tiny=Tensor::from_vector(std::vector<float>{2,0,0,0,0,0,1,0,0,1024,0,0,3,0,0},{5,3},Device::GPU);
        auto tiny_order=ops.permutation(tiny).cpu();const int64_t tiny_expected[]{1,2,0,4,3};require(std::equal(std::begin(tiny_expected),std::end(tiny_expected),tiny_order.ptr<int64_t>()),"Morton lost low/subnormal integer key bits");
        joint_contract<8>(ops,gpu_ops::JointLayout::Rows);joint_contract<16>(ops,gpu_ops::JointLayout::Rows);joint_contract<8>(ops,gpu_ops::JointLayout::SwizzledSH);joint_contract<16>(ops,gpu_ops::JointLayout::SwizzledSH);
        std::vector<int64_t> reverse(n);for(size_t i=0;i<n;++i)reverse[i]=int64_t(n-1-i);auto perm=Tensor::from_blob(reverse.data(),{n},Device::CPU,DataType::Int64).to(Device::GPU);
        for(unsigned rest:{3u,8u,15u}){
            const int degree=rest==3?1:rest==8?2:3;const size_t physical=sh_swizzled_float_count(n,rest);std::vector<float> canonical(n*rest*3);for(size_t i=0;i<canonical.size();++i)canonical[i]=float(int((i*71)%997)-498)*.0003f;
            auto source=Tensor::from_vector(canonical,{n,rest,3},Device::GPU),swizzled=Tensor::zeros({physical},Device::GPU);table.sh->append_canonical(source,swizzled,{n,n,0,rest,rest});
            auto gathered=Tensor::zeros({physical},Device::GPU);ops.gather_gradient(swizzled,perm,gathered,rest,execution);auto reference=Tensor::empty({n,rest,3},Device::GPU);table.sh->gather_canonical(swizzled,perm,reference,{n,n,0,rest,rest});
            for(bool half:{false,true}){
                core::sh_value_quant::set_enabled_for_testing(false);
                auto live=half?swizzled.to(DataType::Float16):swizzled.clone();
                SplatData model(degree,positions.clone(),Tensor::zeros({n,1,3},Device::GPU),live,Tensor::zeros({n,3},Device::GPU),Tensor::zeros({n,4},Device::GPU),Tensor::zeros({n,1},Device::GPU),1.f,SplatData::ShNLayout::Swizzled);
                training::LiveModelMutationGuard guard("HIP Morton contract");training::IdleArenaScratch scratch(0,0,execution);ops.permute_sh_fp32(model,perm,execution,scratch);
                auto actual=Tensor::empty({n,rest,3},Device::GPU);sh_codec(model.shN(),actual,{.source_format=ShFormat::Float32,.destination_format=ShFormat::Canonical,.source_rows=n,.destination_rows=n,.count=n,.source_rest=rest,.destination_rest=rest});
                auto got=actual.cpu().to_vector(),ref=reference.cpu().to_vector();for(size_t i=0;i<got.size();++i){if(half)ref[i]=core::detail::tensor_half_to_float(core::detail::tensor_float_to_half(ref[i]));require(got[i]==ref[i],"Morton FP32/half SH model differs from exact CPU conversion");}
            }
            core::sh_value_quant::set_enabled_for_testing(true);
            const size_t code_cells=sh_swizzled_padded_n(n)*rest*3;auto codes=Tensor::empty({code_cells},Device::GPU,DataType::Float16),bounds=Tensor::empty({(n+255)/256*2},Device::GPU);table.sh->encode_q16(swizzled,codes,bounds,n,rest,0,0);
            auto decoded=Tensor::zeros({physical},Device::GPU);table.sh->decode_q16(codes,bounds,decoded,n,rest);auto expected=Tensor::empty({n,rest,3},Device::GPU);table.sh->gather_canonical(decoded,perm,expected,{n,n,0,rest,rest});
            SplatData model(degree,positions.clone(),Tensor::zeros({n,1,3},Device::GPU),swizzled.clone(),Tensor::zeros({n,3},Device::GPU),Tensor::zeros({n,4},Device::GPU),Tensor::zeros({n,1},Device::GPU),1.f,SplatData::ShNLayout::Swizzled);model.shN()=codes;model.shN_value_bounds()=bounds;
            auto original_codes=codes.clone(),original_bounds=bounds.clone();
            training::LiveModelMutationGuard guard("HIP Morton Q16 contract");training::IdleArenaScratch scratch(0,0,execution);ops.permute_sh_q16(model,perm,execution,scratch);
            auto actual=Tensor::empty({n,rest,3},Device::GPU);sh_codec(model.shN(),actual,{.source_format=ShFormat::Q16,.destination_format=ShFormat::Canonical,.source_rows=n,.destination_rows=n,.count=n,.source_rest=rest,.destination_rest=rest},nullptr,&model.shN_value_bounds());
            auto got=actual.cpu().to_vector(),ref=expected.cpu().to_vector();float error=0;for(size_t i=0;i<got.size();++i)error=std::max(error,std::abs(got[i]-ref[i]));require(error<6e-6f,"Morton Q16 model changed SH beyond one quantization bin");
            {
                auto small_codes=original_codes.clone(),small_bounds=Tensor::empty(bounds.shape(),Device::GPU),small_scratch=Tensor::empty({sh_swizzled_padded_n(n)*2*2},Device::GPU,DataType::UInt8);
                training::hip::ApplicationCall owner;LfsHipMortonCall call{};call.struct_bytes=sizeof(call);call.operation=4;call.device=owner.queue.device;call.queue=owner.queue.native_queue;call.count=n;call.width=rest*3;call.scratch_bytes=small_scratch.bytes();
                call.bindings[0]=owner.bind(small_codes,DataType::Float16);call.bindings[1]=owner.bind(original_bounds,DataType::Float32);call.bindings[2]=owner.bind(perm,DataType::Int64);call.bindings[3]=owner.bind(small_bounds,DataType::Float32);call.bindings[4]=owner.bind(small_scratch,DataType::UInt8);
                auto fn=owner.module->symbol<decltype(&lfs_hip_application_morton)>("lfs_hip_application_morton");LfsHipError status{};require(fn(&call,&status)==0,status.message);
                auto a=small_codes.cpu(),b=model.shN().cpu();require(std::memcmp(a.data_ptr(),b.data_ptr(),a.bytes())==0,"Multi-group and full-group Q16 Morton codes differ");require(small_bounds.cpu().to_vector()==model.shN_value_bounds().cpu().to_vector(),"Multi-group Q16 Morton bounds differ");
            }
            std::printf("Morton SH rest %u: FP32/F16/Q16, max Q16 error %.9g\n",rest,error);
        }
        core::sh_value_quant::set_enabled_for_testing(std::nullopt);
        auto src=Tensor::full({64},7.f,Device::GPU),dst=Tensor::zeros({16},Device::GPU);auto& backend=static_cast<internal::HipBackendOps&>(internal::backend_ops(GpuBackend::HIP));auto pointer=backend.storage_address(internal::storage_ref(src),src.bytes());ops.copy_back(dst,reinterpret_cast<const void*>(pointer+16),dst.bytes(),execution);require(dst.cpu().to_vector()==std::vector<float>(16,7.f),"Morton native borrowed copy differs");
        bool rejected=false;try{ops.copy_back(dst,nullptr,0,TensorExecutionTarget::default_queue(GpuBackend::CUDA));}catch(const std::invalid_argument&){rejected=true;}require(rejected,"Morton accepted foreign queue");
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"Morton changed backend/default or leaked allocations");
    std::puts("PASS HIP application Morton: exact integer/stable order, joint 8/16-bit moments, grouped/full equality, real SH models and borrowed copies");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MORTON CONTRACT FAILED: %s\n",e.what());return 1;}
