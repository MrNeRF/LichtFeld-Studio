/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor_backend.hpp>
#include "core/tensor/internal/tensor_impl.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
using namespace lfs::core;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static std::vector<int64_t> read(const Tensor& t) {
    const auto cpu=t.cpu();std::vector<int64_t> result(cpu.numel());
    for(size_t i=0;i<result.size();++i) {
        if(cpu.dtype()==DataType::Int64)result[i]=cpu.ptr<int64_t>()[i];
        else if(cpu.dtype()==DataType::Int32)result[i]=cpu.ptr<int32_t>()[i];
        else if(cpu.dtype()==DataType::Bool)result[i]=cpu.ptr<uint8_t>()[i];
        else throw std::runtime_error("Unexpected integer reduction output dtype");
    }return result;
}
// Exercise the native descriptor where the shared Tensor API deliberately
// rejects a dtype/operation combination on every backend (e.g. Int64 reduce).
static Tensor native_reduce(const Tensor& input,int axis,int operation) {
    const auto dtype=operation>=5?DataType::Bool:(input.dtype()==DataType::Bool&&operation==0?DataType::Int64:input.dtype());
    std::vector<size_t> shape;
    if(axis>=0)for(size_t i=0;i<input.ndim();++i)if(i!=size_t(axis))shape.push_back(input.shape()[i]);
    auto out=Tensor::empty(TensorShape(shape),Device::GPU,dtype);
    (void)const_cast<Tensor&>(input).data_ptr();(void)out.data_ptr();
    pin_operands({&input,&out});
    internal::ReduceProgram p{};p.op=static_cast<ReduceOp>(operation);p.result_dtype=dtype;
    if(axis>=0){p.axis_count=1;p.axes[0]=axis;}
    internal::backend_ops(GpuBackend::HIP).reduce(internal::storage_ref(input),internal::storage_ref(out),internal::strided_layout(input),p,{});
    return out;
}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        const std::vector<int> conversion{0,1,-1,42,-73,std::numeric_limits<int>::lowest(),std::numeric_limits<int>::max()};
        auto converted=Tensor::from_vector(conversion,{conversion.size()},Device::GPU).to(DataType::Bool).cpu().to_vector_bool();
        for(size_t i=0;i<converted.size();++i)require(converted[i]==(conversion[i]!=0),"HIP Int32 to Bool lost nonzero semantics");
        for(auto dtype:{DataType::Int32,DataType::Int64,DataType::Bool})for(bool transpose:{false,true}) {
            std::vector<int64_t> values(30);for(size_t i=0;i<values.size();++i)values[i]=dtype==DataType::Bool?int64_t(i%4!=0):int64_t(int(i%17)-8)+(dtype==DataType::Int64?9007199254740993LL:0);
            Tensor tensor;
            if(dtype==DataType::Int64)tensor=Tensor::from_blob(values.data(),{2,3,5},Device::CPU,DataType::Int64).to(Device::GPU);
            else {std::vector<int> v(values.size());for(size_t i=0;i<v.size();++i)v[i]=static_cast<int>(values[i]);tensor=Tensor::from_vector(v,{2,3,5},Device::GPU);if(dtype==DataType::Bool)tensor=tensor.to(dtype);}
            std::array<size_t,3> dims{2,3,5};
            if(transpose) {
                tensor=tensor.transpose(0,2);dims={5,3,2};std::vector<int64_t> transposed(30);
                for(size_t a=0;a<2;++a)for(size_t b=0;b<3;++b)for(size_t c=0;c<5;++c)transposed[(c*3+b)*2+a]=values[(a*3+b)*5+c];values=std::move(transposed);
            }
            for(int axis=-1;axis<3;++axis)for(int op:{0,2,3,5,6}) {
                const auto output=native_reduce(tensor,axis,op);
                const auto actual=read(output);std::vector<int64_t> expected(actual.size(),op==6?1:0);
                if(op==2)std::fill(expected.begin(),expected.end(),std::numeric_limits<int64_t>::lowest());
                if(op==3)std::fill(expected.begin(),expected.end(),std::numeric_limits<int64_t>::max());
                for(size_t a=0;a<dims[0];++a)for(size_t b=0;b<dims[1];++b)for(size_t c=0;c<dims[2];++c) {
                    const auto v=values[(a*dims[1]+b)*dims[2]+c];size_t destination=0;
                    const size_t coordinate[]{a,b,c};if(axis>=0)for(int d=0;d<3;++d)if(d!=axis)destination=destination*dims[d]+coordinate[d];
                    auto& out=expected[destination];if(op==0)out+=v;if(op==2)out=std::max(out,v);if(op==3)out=std::min(out,v);if(op==5)out|=v!=0;if(op==6)out&=v!=0;
                }
                require(actual==expected,"HIP integer/logical reduction differs from exact CPU reference");
            }
        }
        auto empty=Tensor::empty({3,0,7},Device::GPU,DataType::Bool);
        // The public facade resolves empty identities before entering a backend.
        for(auto v:read(empty.sum(1)))require(v==0,"Empty Bool sum identity differs");
        for(auto v:read(empty.any(1)))require(v==0,"Empty Bool any identity differs");
        for(auto v:read(empty.all(1)))require(v==1,"Empty Bool all identity differs");
        auto fp=Tensor::from_vector(std::vector<float>{0,-0.f,1,std::numeric_limits<float>::quiet_NaN(),0,0},{2,3},Device::GPU);
        require(read(native_reduce(fp,1,5))==std::vector<int64_t>{1,1},"Float32 any mishandled NaN or signed zero");
        require(read(native_reduce(fp,1,6))==std::vector<int64_t>{0,0},"Float32 all mishandled zeros");
        auto mask=Tensor::from_vector(std::vector<float>{0,1,0,1,1,0},{2,3},Device::GPU).to(DataType::Bool);
        require(mask.sum_scalar()==3,"Public Bool sum_scalar differs from exact GPU count");
        require(read(mask.any(1))==std::vector<int64_t>{1,1},"Public Bool any differs");
        auto ints=Tensor::from_vector(std::vector<int>{-7,2,3,9,8,-1},{2,3},Device::GPU);
        require(read(ints.sum(1))==std::vector<int64_t>{-2,16},"Public Int32 row sum differs");
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Integer reduction contract retained HIP storage");
    std::puts("PASS HIP typed axis reductions: exact Int64 beyond 2^53, Int32/Bool, sums/extrema/any/all, transposed layouts, empty identities, nonzero conversions");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());(void)shutdown_gpu_backend(GpuBackend::HIP);return 1;}
