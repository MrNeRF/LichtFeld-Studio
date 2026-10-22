/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_fused.hpp>
#include <core/detail/tensor_half.hpp>
#include <core/tensor/internal/expression_runtime.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <bit>
#include <limits>
#include <cstdio>
#include <functional>
#include <future>
#include <stdexcept>
#include <vector>
using namespace lfs::core;
using namespace lfs::core::internal;
namespace f=lfs::core::fused;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void near(float actual,float expected) {
    if(std::isnan(expected)) require(std::isnan(actual),"HIP expression lost NaN");
    else if(std::isinf(expected)) require(actual==expected,"HIP expression changed infinity");
    else require(std::isfinite(actual) && std::abs(actual-expected)<=3e-5f*std::max(1.f,std::abs(expected)),"HIP expression differs from scalar reference");
}
void unary() {
    const std::vector<float> values{-.75f,-.5f,-.125f,-0.f,0.f,.125f,.5f,.75f,1.f};
    auto input=Tensor::from_vector(values,{values.size()},Device::GPU);
    const std::vector<std::pair<ExprOp,std::function<float(float)>>> cases{
        {ExprOp::Abs,[](float x){return std::abs(x);}}, {ExprOp::Neg,[](float x){return -x;}},
        {ExprOp::Exp,[](float x){return std::exp(x);}}, {ExprOp::Log,[](float x){return std::log(x);}},
        {ExprOp::Sqrt,[](float x){return std::sqrt(x);}}, {ExprOp::Tanh,[](float x){return std::tanh(x);}},
        {ExprOp::Round,[](float x){return std::nearbyint(x);}}, {ExprOp::Log1p,[](float x){return std::log1p(x);}},
        {ExprOp::Sin,[](float x){return std::sin(x);}}, {ExprOp::Cos,[](float x){return std::cos(x);}},
        {ExprOp::Asin,[](float x){return std::asin(x);}}, {ExprOp::Acos,[](float x){return std::acos(x);}},
        {ExprOp::Exp2,[](float x){return std::exp2(x);}}, {ExprOp::Log2,[](float x){return std::log2(x);}},
        {ExprOp::Log10,[](float x){return std::log10(x);}}, {ExprOp::Rsqrt,[](float x){return 1.f/std::sqrt(x);}},
        {ExprOp::Relu,[](float x){return std::max(0.f,x);}}, {ExprOp::Square,[](float x){return x*x;}},
        {ExprOp::Reciprocal,[](float x){return 1.f/x;}}, {ExprOp::Floor,[](float x){return std::floor(x);}},
        {ExprOp::Ceil,[](float x){return std::ceil(x);}}, {ExprOp::Tan,[](float x){return std::tan(x);}},
        {ExprOp::Atan,[](float x){return std::atan(x);}}, {ExprOp::Sinh,[](float x){return std::sinh(x);}},
        {ExprOp::Cosh,[](float x){return std::cosh(x);}}, {ExprOp::Trunc,[](float x){return std::trunc(x);}},
        {ExprOp::Sign,[](float x){return float(int(x>0)-int(x<0));}},
        {ExprOp::Gelu,[](float x){return .5f*x*(1.f+std::tanh(std::sqrt(2.f/3.14159265358979323846f)*(x+.044715f*x*x*x)));}},
        {ExprOp::Sigmoid,[](float x){return 1.f/(1.f+std::exp(-x));}}, {ExprOp::Swish,[](float x){return x/(1.f+std::exp(-x));}}
    };
    for(const auto& [op,reference] : cases) {
        ExpressionBuilder builder;
        auto x=builder.emit(ExprOp::Load);
        builder.emit(ExprOp::Store,builder.emit(op,x));
        auto output=Tensor::empty({values.size()},Device::GPU);
        const std::array<ExprInput,1> in{{{storage_ref(input),{uint32_t(values.size()),1,1,1},{1,0,0,0}}}};
        const std::array<ExprOutput,1> out{{{storage_ref(output),{1,0,0,0}}}};
        run_expression(builder.build(),in,out,{{uint32_t(values.size()),1,1,1},1});
        const auto observed=output.cpu();
        for(size_t i=0;i<values.size();++i) near(observed.ptr<float>()[i],reference(values[i]));
    }
}
void precise_bits() {
    const float tiny=std::numeric_limits<float>::denorm_min();
    const std::vector<float> a{-0.f,0.f,tiny,tiny*4.f,1.f,-1.f,INFINITY,NAN};
    const std::vector<float> b{1.f,-1.f,1.f,2.f,3.f,0.f,INFINITY,1.f};
    auto x=Tensor::from_vector(a,{a.size()},Device::GPU),y=Tensor::from_vector(b,{b.size()},Device::GPU);
    for(auto op : {ExprOp::PreciseDiv,ExprOp::Min,ExprOp::Max}) {
        ExpressionBuilder builder;
        auto lhs=builder.emit(ExprOp::Load),rhs=builder.emit(ExprOp::Load,0,0,0,1);
        builder.emit(ExprOp::Store,builder.emit(op,lhs,rhs));
        auto output=Tensor::empty({a.size()},Device::GPU);
        const std::array<ExprInput,2> input{{{storage_ref(x),{8,1,1,1},{1,0,0,0}}, {storage_ref(y),{8,1,1,1},{1,0,0,0}}}};
        const std::array<ExprOutput,1> out{{{storage_ref(output),{1,0,0,0}}}};
        run_expression(builder.build(),input,out,{{8,1,1,1},1});
        const auto actual=output.cpu();
        for(size_t i=0;i<a.size();++i) {
            if(std::isnan(a[i]) || (op==ExprOp::PreciseDiv && std::isinf(a[i]) && std::isinf(b[i]))) {
                require(std::isnan(actual.ptr<float>()[i]),"HIP precise math lost NaN");continue;
            }
            uint32_t expected;
            if(op==ExprOp::PreciseDiv) expected=std::bit_cast<uint32_t>(a[i]/b[i]);
            else if(a[i]==0.f && b[i]==0.f) expected=op==ExprOp::Min ? std::bit_cast<uint32_t>(a[i])|std::bit_cast<uint32_t>(b[i]) : std::bit_cast<uint32_t>(a[i])&std::bit_cast<uint32_t>(b[i]);
            else expected=std::bit_cast<uint32_t>(op==ExprOp::Min ? std::fmin(a[i],b[i]) : std::fmax(a[i],b[i]));
            require(std::bit_cast<uint32_t>(actual.ptr<float>()[i])==expected,"HIP precise math changed subnormal or signed-zero bits");
        }
    }
}
void gathered_and_packed() {
    const auto indices=Tensor::from_vector(std::vector<int32_t>{-8,-1,0,1,2,3,8},{7},Device::GPU);
    for(auto dtype : {DataType::Float32,DataType::Float16,DataType::UInt8,DataType::Bool}) {
        auto table=Tensor::empty_pageable_host({3},dtype);
        for(size_t i=0;i<3;++i) {
            if(dtype==DataType::Float32) table.ptr<float>()[i]=float(i)*.25f;
            else if(dtype==DataType::Float16) static_cast<detail::tensor_half_t*>(table.data_ptr())[i]=detail::tensor_float_to_half(float(i)*.25f);
            else static_cast<uint8_t*>(table.data_ptr())[i]=uint8_t(i%2);
        }
        const auto gpu_table=table.to(Device::GPU);
        for(auto bounds : {f::Bounds::Clamp,f::Bounds::Zero,f::Bounds::Wrap}) {
            f::Builder builder(1);
            const auto index=builder.input(DataType::Int32,1);
            const auto data=builder.input(dtype,1);
            builder.output(data.gather({index.load()},bounds),dtype);
            const f::Kernel kernel(builder);
            const auto output=kernel({7},{indices,gpu_table})[0].cpu();
            const int32_t raw[]{-8,-1,0,1,2,3,8};
            for(size_t i=0;i<7;++i) {
                const auto j=bounds==f::Bounds::Clamp ? std::clamp(raw[i],0,2) : (raw[i]%3+3)%3;
                const bool zero=bounds==f::Bounds::Zero && (raw[i]<0 || raw[i]>=3);
                const float expected=zero ? 0.f : dtype==DataType::Float32 || dtype==DataType::Float16 ? float(j)*.25f : float(j%2);
                const float actual=dtype==DataType::Float32 ? output.ptr<float>()[i] : dtype==DataType::Float16 ? detail::tensor_half_to_float(static_cast<const detail::tensor_half_t*>(output.data_ptr())[i]) : float(static_cast<const uint8_t*>(output.data_ptr())[i]);
                require(actual==expected,"HIP fused gather/packed output differs");
            }
        }
    }
}
void fold_and_host() {
    f::Builder builder(2);
    const auto image=builder.input(DataType::Float32,2);
    const auto gain=builder.input(DataType::Float32,1);
    const auto value=image.load()*gain.at({0});
    builder.output(builder.fold(f::Fold::Sum,value,1),DataType::Float32);
    builder.output(builder.fold(f::Fold::Min,value,1),DataType::Float32);
    builder.output(builder.fold(f::Fold::Max,value,1),DataType::Float32);
    const f::Kernel kernel(builder);
    const auto host_gain=Tensor::from_vector(std::vector<float>{.5f},{1},Device::CPU);
    for(size_t columns : {size_t(1),size_t(17),size_t(65),size_t(129)}) {
        std::vector<float> values(3*columns);
        for(size_t i=0;i<values.size();++i) values[i]=float(int(i%19)-9);
        const auto tensor=Tensor::from_vector(values,{3,columns},Device::GPU);
        const auto outputs=kernel({3,columns},{tensor,host_gain});
        for(size_t o=0;o<3;++o) {
            const auto observed=outputs[o].cpu();
            for(size_t row=0;row<3;++row) {
                float expected=o==0?0.f:o==1?INFINITY:-INFINITY;
                for(size_t j=0;j<columns;++j) {
                    const float x=values[row*columns+j]*.5f;
                    expected=o==0?expected+x:o==1?std::min(expected,x):std::max(expected,x);
                }
                near(observed.ptr<float>()[row],expected);
            }
        }
    }
}
void cache_and_threads() {
    f::Builder builder(1);
    const auto input=builder.input(DataType::Float32,1);
    builder.output(input.load()*2.f+1.f,DataType::Float32);
    const f::Kernel kernel(builder);
    auto input_a=Tensor::from_vector(std::vector<float>(17,3.f),{17},Device::GPU);
    auto output=Tensor::from_vector(std::vector<float>(17,-77.f),{17},Device::GPU);
    kernel.prepare({17},{input_a},{output});
    require(output.cpu().ptr<float>()[0]==-77.f,"HIP prepare executed the kernel");
    kernel({17},{input_a},{output});
    require(output.cpu().ptr<float>()[16]==7.f,"HIP prepared kernel launch failed");
    const auto before=expression_cache_stats(GpuBackend::HIP);
    for(size_t n : {size_t(19),size_t(257),size_t(4099)}) {
        const auto a=Tensor::from_vector(std::vector<float>(n,4.f),{n},Device::GPU);
        const auto out=kernel({n},{a})[0].cpu();
        require(out.ptr<float>()[n-1]==9.f,"HIP expression changed-shape launch failed");
    }
    const auto after=expression_cache_stats(GpuBackend::HIP);
    require(after.compilations==before.compilations && after.loads==before.loads && after.hits>=before.hits+3,
            "HIP warm expression failed to reuse layout specialization");
    std::vector<std::future<void>> jobs;
    for(int worker=0;worker<3;++worker) jobs.push_back(std::async(std::launch::async,[&,worker] {
        GpuBackendScope scope(GpuBackend::HIP);
        const auto a=Tensor::from_vector(std::vector<float>(31,float(worker)),{31},Device::GPU);
        const auto out=kernel({31},{a})[0].cpu();
        require(out.ptr<float>()[30]==float(worker*2+1),"Concurrent HIP expression launch lost arguments");
    }));
    for(auto& job:jobs) job.get();
    const auto stats=expression_cache_stats(GpuBackend::HIP);
    std::printf("HIP expression cache: entries=%zu bytes=%zu compilations=%llu disk=%llu hits=%llu\n",stats.entries,stats.bytes,
        static_cast<unsigned long long>(stats.compilations),static_cast<unsigned long long>(stats.disk_hits),static_cast<unsigned long long>(stats.hits));
}
int main() try {
    GpuBackendScope scope(GpuBackend::HIP);
    unary();precise_bits();gathered_and_packed();fold_and_host();cache_and_threads();
    std::puts("PASS HIP expressions: native math, IEEE values, gather bounds, Float16/Bool/UInt8 packed tails, host constants, short/runtime folds, prepare, shape cache and concurrent launches");
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"HIP expression failed: %s\n",e.what());return 1;}
