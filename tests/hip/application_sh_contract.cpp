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
size_t at(size_t p,size_t cell,size_t slots) { return ((p/32)*(slots*32)+(cell/4)*32+p%32)*4+cell%4; }
int main(int argc,char** argv) try {
    require(argc==2 && std::filesystem::path(argv[1]).is_absolute(),"Pass absolute HIP training module");
#ifdef _WIN32
    require(_putenv_s("LFS_HIP_TRAINING_MODULE",argv[1])==0,"Cannot set HIP module");
#else
    require(setenv("LFS_HIP_TRAINING_MODULE",argv[1],1)==0,"Cannot set HIP module");
#endif
    const auto configured=configured_gpu_backend(); const auto& table=training::training_ops(GpuBackend::HIP);
    require(table.sh && !table.fast && !training::other_tables && !gpu_backend_live(GpuBackend::HIP),"SH discovery initialized another backend");
    {
        const GpuBackendScope scope(GpuBackend::HIP); const auto& ops=*table.sh;
        constexpr size_t n=777;
        for(unsigned rest:{3u,8u,15u}) {
            const size_t cells=rest*3,slots=sh_float4_slots_for_rest(rest),physical=sh_swizzled_float_count(n,rest),code_bytes=sh_swizzled_padded_n(n)*cells*2,nb=(n+255)/256;
            std::vector<float> canonical(n*cells);
            for(size_t i=0;i<canonical.size();++i) canonical[i]=float(int((i*7919)%3571)-1785)*.0002f;
            auto input=Tensor::from_vector(canonical,{n,rest,3},Device::GPU);
            auto swizzle=Tensor::full({physical},123.f,Device::GPU);
            ops.append_canonical(input,swizzle,{n,n,0,rest,rest});
            const auto resident=swizzle.cpu().to_vector();
            for(size_t p=0;p<n;++p) for(size_t cell=0;cell<slots*4;++cell)
                require(resident[at(p,cell,slots)]==(cell<cells ? canonical[p*cells+cell] : 0.f),"SH append/swizzle/padding differs");
            for(size_t p=n;p<sh_swizzled_padded_n(n);++p) for(size_t cell=0;cell<slots*4;++cell)
                require(resident[at(p,cell,slots)]==123.f,"SH append changed reserved row padding");
            auto indices=Tensor::from_vector(std::vector<int>{776,3,0},{3},Device::GPU);
            auto i64=indices.to(DataType::Int64);
            auto gathered=Tensor::zeros({3,rest,3},Device::GPU);
            ops.gather_canonical(swizzle,i64,gathered,{n,3,0,rest,rest});
            const auto gathered_values=gathered.cpu().to_vector();
            constexpr size_t rows[]{776,3,0};
            for(size_t row=0;row<3;++row) for(size_t cell=0;cell<cells;++cell)
                require(gathered_values[row*cells+cell]==canonical[rows[row]*cells+cell],"SH canonical gather differs");
            auto gathered_swizzle=Tensor::full({sh_swizzled_float_count(5,rest)},123.f,Device::GPU);
            ops.gather_swizzled(swizzle,i64,gathered_swizzle,{n,3,2,rest,rest});
            const auto gathered_resident=gathered_swizzle.cpu().to_vector();
            for(size_t row=0;row<3;++row) for(size_t cell=0;cell<cells;++cell)
                require(gathered_resident[at(row+2,cell,slots)]==canonical[rows[row]*cells+cell],"SH swizzled gather/offset differs");
            auto byte_source=swizzle.mul(100.f).add(128.f).to(DataType::UInt8);
            auto byte_destination=Tensor::full({sh_swizzled_float_count(5,rest)},123.f,Device::GPU,DataType::UInt8);
            ops.gather_swizzled(byte_source,indices,byte_destination,{n,3,2,rest,rest});
            const auto src_bytes=byte_source.cpu().to_vector_uint8(),dst_bytes=byte_destination.cpu().to_vector_uint8();
            for(size_t row=0;row<3;++row) for(size_t cell=0;cell<slots*4;++cell)
                require(dst_bytes[at(row+2,cell,slots)]==src_bytes[at(rows[row],cell,slots)],"SH byte gather differs");
            auto zeroed=swizzle.clone(); ops.zero_rows(zeroed,indices,rest);
            const auto zero_values=zeroed.cpu().to_vector();
            for(size_t p=0;p<n;++p) for(size_t cell=0;cell<slots*4;++cell)
                require(zero_values[at(p,cell,slots)]==((p==776 || p==3 || p==0) ? 0.f : resident[at(p,cell,slots)]),"SH zero rows changed unrelated coefficients");
            auto changed=Tensor::full({3,rest,3},-.7f,Device::GPU),scattered=swizzle.clone();
            ops.scatter_canonical(changed,indices,scattered,{n,3,0,rest,rest});
            const auto scattered_values=scattered.cpu().to_vector();
            for(size_t p=0;p<n;++p) for(size_t cell=0;cell<cells;++cell)
                require(scattered_values[at(p,cell,slots)]==((p==776 || p==3 || p==0) ? -.7f : resident[at(p,cell,slots)]),"SH scatter changed wrong coefficient");
            constexpr size_t code_offset=32,bound_offset=4;
            auto code_backing=Tensor::full({code_bytes+code_offset*2+64},90.f,Device::GPU,DataType::UInt8);
            auto bound_backing=Tensor::full({nb*2+bound_offset+4},123.f,Device::GPU);
            ops.encode_q16(swizzle,code_backing,bound_backing,n,rest,code_offset,bound_offset);
            const auto backing=code_backing.cpu().to_vector_uint8(); const auto bound_values=bound_backing.cpu().to_vector();
            require(std::all_of(backing.begin(),backing.begin()+code_offset*2,[](auto x){return x==90;}) && std::all_of(backing.end()-64,backing.end(),[](auto x){return x==90;}),"Chunk Q16 encode changed code sentinels");
            require(std::all_of(bound_values.begin(),bound_values.begin()+bound_offset,[](float x){return x==123;}) && std::all_of(bound_values.end()-4,bound_values.end(),[](float x){return x==123;}),"Chunk Q16 encode changed bounds sentinels");
            auto codes=code_backing.slice(0,code_offset*2,code_offset*2+code_bytes),bounds=bound_backing.slice(0,bound_offset,bound_offset+nb*2);
            auto decoded=Tensor::full({physical},123.f,Device::GPU); ops.decode_q16(codes,bounds,decoded,n,rest);
            const auto decoded_values=decoded.cpu().to_vector();
            for(size_t p=0;p<n;++p) for(size_t cell=0;cell<cells;++cell)
                require(std::abs(decoded_values[at(p,cell,slots)]-canonical[p*cells+cell])<2e-5f,"Q16 round trip exceeds quantization error");
            constexpr size_t offset=123,count=71;
            auto range=Tensor::zeros({count},Device::GPU);
            Tensor absent;
            ops.decode_range(swizzle,absent,range,{offset,count,n,rest,rest,gpu_ops::ShStorage::Float32});
            const auto exact=range.cpu().to_vector();
            for(size_t i=0;i<count;++i) require(exact[i]==canonical[offset+i],"FP32 bounded checkpoint decode differs");
            auto half=swizzle.to(DataType::Float16);
            ops.decode_range(half,absent,range,{offset,count,n,rest,rest,gpu_ops::ShStorage::IeeeFloat16});
            const auto half_result=range.cpu().to_vector(),half_source=half.to(DataType::Float32).cpu().to_vector();
            for(size_t i=0;i<count;++i) { const auto j=offset+i; require(half_result[i]==half_source[at(j/cells,j%cells,slots)],"F16 bounded checkpoint decode differs"); }
            ops.decode_range(codes,bounds,range,{offset,count,n,rest,rest,gpu_ops::ShStorage::Q16});
            const auto rounded=range.cpu().to_vector(),ranges=bounds.cpu().to_vector(); const auto code_values=codes.cpu().to_vector_uint8();
            for(size_t i=0;i<count;++i) {
                const size_t j=offset+i,p=j/cells,cell=j%cells,k=(p/32)*(cells*32)+cell*32+p%32;
                uint16_t q=0; std::memcpy(&q,code_values.data()+k*2,2);
                const float lo=ranges[(p/256)*2],hi=ranges[(p/256)*2+1],normalized=float(q)*(1.f/65535.f),delta=hi-lo;
                const float expected=std::fma(delta,normalized,lo);
                require(rounded[i]==expected,"Q16 bounded checkpoint decode is not host-rounding exact");
            }
            {
                // Snapshot capture intentionally supplies an untyped byte
                // scratch band. Exercise that real caller contract directly.
                auto backing=Tensor::empty({count*4+16},Device::GPU,DataType::UInt8);
                ops.fill_bytes(backing,backing.bytes(),0x5a);
                auto bytes=backing.slice(0,4,4+count*4);
                for(auto codec:{gpu_ops::ShStorage::Float32,gpu_ops::ShStorage::IeeeFloat16,gpu_ops::ShStorage::Q16}) {
                    const auto& source=codec==gpu_ops::ShStorage::Float32?swizzle:codec==gpu_ops::ShStorage::IeeeFloat16?half:codes;
                    ops.decode_range(source,codec==gpu_ops::ShStorage::Q16?bounds:absent,bytes,{offset,count,n,rest,rest,codec});
                    const auto actual=backing.cpu().to_vector_uint8();
                    const auto& reference=codec==gpu_ops::ShStorage::Float32?exact:codec==gpu_ops::ShStorage::IeeeFloat16?half_result:rounded;
                    for(size_t i=0;i<count;++i){float value;std::memcpy(&value,actual.data()+4+i*4,4);require(value==reference[i],"SH checkpoint byte scratch differs from typed destination");}
                    for(size_t i=0;i<4;++i)require(actual[i]==0x5a,"SH byte scratch prefix guard was overwritten");
                    for(size_t i=4+count*4;i<actual.size();++i)require(actual[i]==0x5a,"SH byte scratch suffix guard was overwritten");
                }
                auto bad=backing.slice(0,1,1+count*4);bool rejected=false;
                try{ops.decode_range(codes,bounds,bad,{offset,count,n,rest,rest,gpu_ops::ShStorage::Q16});}catch(const std::invalid_argument&){rejected=true;}
                require(rejected,"SH byte scratch accepted a misaligned Float32 destination");
            }
            auto destinations=Tensor::from_vector(std::vector<int>{0,3,260,776},{4},Device::GPU).to(DataType::Int64);
            auto ids=Tensor::zeros({4},Device::GPU),unique=Tensor::zeros({4},Device::GPU,DataType::Int32),run_offsets=unique.clone(),run_count=Tensor::zeros({1},Device::GPU,DataType::Int32);
            auto scratch=ops.create_run_scratch(); ops.block_ids(destinations,ids); ops.block_runs(*scratch,ids,unique,run_offsets,run_count);
            require(ids.cpu().to_vector()==std::vector<float>({0,0,1,3}) && run_count.cpu().to_vector_int()[0]==3,"SH touched block IDs/run count differ");
            const auto unique_values=unique.cpu().to_vector_int(),offset_values=run_offsets.cpu().to_vector_int();
            require(std::equal(unique_values.begin(),unique_values.begin()+3,std::vector<int>({0,1,3}).begin()) && std::equal(offset_values.begin(),offset_values.begin()+3,std::vector<int>({0,2,3}).begin()),"SH run compaction differs");
            std::vector<float> change_values(4*cells);
            for(size_t row=0;row<4;++row) std::fill_n(change_values.begin()+row*cells,cells,.6f+.05f*row);
            auto changes=Tensor::from_vector(change_values,{4,rest,3},Device::GPU);
            auto order=Tensor::from_vector(std::vector<int>{3,2,1,0},{4},Device::GPU).to(DataType::Int64);
            ops.reencode_touched(codes,bounds,changes,destinations,unique,run_offsets,run_count,order,{4,n,n,rest});
            ops.decode_q16(codes,bounds,decoded,n,rest);
            const auto touched=decoded.cpu().to_vector();
            const auto after_codes=codes.cpu().to_vector_uint8();
            for(size_t p=0;p<n;++p) for(size_t cell=0;cell<cells;++cell) {
                const bool changed_row=p==0 || p==3 || p==260 || p==776;
                const float expected=changed_row ? (.6f+.05f*(p==0 ? 3 : p==3 ? 2 : p==260 ? 1 : 0)) : decoded_values[at(p,cell,slots)];
                require(std::abs(touched[at(p,cell,slots)]-expected)<4e-5f,"Touched Q16 re-encode changed wrong row or missed a run");
            }
            require(std::equal(after_codes.begin()+512*cells*2,after_codes.begin()+768*cells*2,code_values.begin()+512*cells*2),"Touched Q16 rewrote an untouched block");
            auto arena=Tensor::full({2048},90.f,Device::GPU,DataType::UInt8);
            std::array parts{gathered.slice(0,0,1),gathered.slice(0,1,3)};
            auto joined=ops.concatenate_into_arena(parts,static_cast<char*>(arena.data_ptr())+64,{3,rest,3},DataType::Float32,TensorExecutionTarget::default_queue(GpuBackend::HIP));
            require(joined.cpu().to_vector()==gathered_values,"SH arena concatenation differs");
            const auto arena_bytes=arena.cpu().to_vector_uint8();
            require(std::all_of(arena_bytes.begin(),arena_bytes.begin()+64,[](auto x){return x==90;}),"SH arena copy changed prefix");
            ops.fill_bytes(arena,37,7);
            const auto filled=arena.cpu().to_vector_uint8();
            require(std::all_of(filled.begin(),filled.begin()+37,[](auto x){return x==7;}) && filled[37]==90,"SH byte fill exceeds requested region");
            std::printf("SH rest %u: layout/gather/scatter, codecs/ranges, touched blocks and arena passed\n",rest);
        }
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"SH changed default or leaked resources");
    std::puts("PASS HIP application SH: full table, three SH layouts, exact/quantized/f16 data paths, run compaction, touched-block preservation and bounded arena views");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"HIP SH CONTRACT FAILED: %s\n",e.what()); return 1; }
