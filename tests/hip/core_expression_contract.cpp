/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/detail/gpu_backend_ops.hpp>
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
template<class T> void integer_arithmetic_type(DataType dtype,std::vector<T> values) {
    using U=std::make_unsigned_t<T>;
    const auto wrap=[](U value){return std::bit_cast<T>(value);};
    auto host=Tensor::empty_pageable_host({values.size()},dtype);
    std::copy(values.begin(),values.end(),static_cast<T*>(host.data_ptr()));
    auto input=host.to(Device::GPU);
    auto guards=Tensor::empty_pageable_host({values.size()+2},dtype);
    std::fill_n(static_cast<T*>(guards.data_ptr()),values.size()+2,T(77));
    auto output=guards.to(Device::GPU);
    auto destination=storage_ref(output);destination.byte_offset=sizeof(T);
    auto& gpu=backend_ops(GpuBackend::HIP);
    PointwiseProgram program{};program.in_dtype=dtype;program.out_dtype=dtype;
    const auto check=[&](const std::vector<T>& wanted) {
        auto actual=output.cpu();const auto* data=static_cast<const T*>(actual.data_ptr());
        require(data[0]==T(77) && data[values.size()+1]==T(77),"HIP integer arithmetic overwrote guards");
        for(size_t i=0;i<wanted.size();++i) require(data[i+1]==wanted[i],"HIP integer arithmetic lost exact integer values");
    };
    for(auto op : {PointwiseOp::AddTensor,PointwiseOp::SubTensor,PointwiseOp::MulTensor,PointwiseOp::MinimumTensor,PointwiseOp::MaximumTensor}) {
        program.op=op;
        gpu.binary(program,storage_ref(input),storage_ref(input),destination,values.size(),{});
        std::vector<T> wanted(values.size());
        for(size_t i=0;i<values.size();++i) {
            const U x=U(values[i]);
            wanted[i]=op==PointwiseOp::AddTensor?wrap(U(x+x)):op==PointwiseOp::SubTensor?T(0):op==PointwiseOp::MulTensor?wrap(U(x*x)):values[i];
        }
        check(wanted);
    }
    std::vector<T> divisors(values.size());
    for(size_t i=0;i<divisors.size();++i) divisors[i]=i%4==0?T(-1):i%4==1?T(0):i%4==2?T(2):T(3);
    auto divisor_host=Tensor::empty_pageable_host({values.size()},dtype);
    std::copy(divisors.begin(),divisors.end(),static_cast<T*>(divisor_host.data_ptr()));
    const auto divisor=divisor_host.to(Device::GPU);
    for(auto op : {PointwiseOp::DivTensor,PointwiseOp::ModTensor,PointwiseOp::PowTensor}) {
        program.op=op;
        program.scalar.kind=ScalarKind::Float;program.scalar.value.float_value=NAN;program.scalar.scalar_on_right=false;
        gpu.binary(program,storage_ref(input),storage_ref(divisor),destination,values.size(),{});
        std::vector<T> wanted(values.size());
        for(size_t i=0;i<values.size();++i) {
            const T a=values[i],b=divisors[i];
            if(op==PointwiseOp::PowTensor) {
                if constexpr(std::is_signed_v<T>) if(b<0) {wanted[i]=a==T(1)?T(1):a==T(-1)?(b%2?T(-1):T(1)):T(0);continue;}
                U result=1,factor=U(a),exponent=U(b);
                while(exponent) {if(exponent&1)result=U(result*factor);factor=U(factor*factor);exponent>>=1;}
                wanted[i]=wrap(result);
            } else if(!b) wanted[i]=0;
            else {
                if constexpr(std::is_signed_v<T>) if(a==std::numeric_limits<T>::min() && b==T(-1)) {wanted[i]=op==PointwiseOp::DivTensor?a:T(0);continue;}
                wanted[i]=op==PointwiseOp::DivTensor?T(a/b):T(a%b);
            }
        }
        check(wanted);
    }
    program.op=PointwiseOp::SubScalar;program.scalar.kind=ScalarKind::Int64;program.scalar.value.int64_value=16777217;
    for(bool right : {true,false}) {
        program.scalar.scalar_on_right=right;
        gpu.unary(program,storage_ref(input),destination,values.size(),{});
        std::vector<T> wanted(values.size());
        for(size_t i=0;i<values.size();++i) wanted[i]=wrap(right?U(U(values[i])-U(T(16777217))):U(U(T(16777217))-U(values[i])));
        check(wanted);
    }
    program.scalar.scalar_on_right=true;
    for(auto op : {PointwiseOp::Abs,PointwiseOp::Neg,PointwiseOp::Square,PointwiseOp::Sign,PointwiseOp::Relu}) {
        program.op=op;
        gpu.unary(program,storage_ref(input),destination,values.size(),{});
        std::vector<T> wanted(values.size());
        for(size_t i=0;i<values.size();++i) {
            const T x=values[i];const U u=U(x);
            wanted[i]=op==PointwiseOp::Neg?wrap(U(U(0)-u)):op==PointwiseOp::Square?wrap(U(u*u)):
                op==PointwiseOp::Abs?(x<T(0)?wrap(U(U(0)-u)):x):op==PointwiseOp::Relu?std::max(T(0),x):T((x>T(0))-(x<T(0)));
        }
        check(wanted);
    }
    program.op=PointwiseOp::AddScalar;program.scalar.kind=ScalarKind::Float;program.scalar.value.float_value=1.75f;
    gpu.unary(program,storage_ref(input),destination,values.size(),{});
    std::vector<T> wanted(values.size());for(size_t i=0;i<values.size();++i) wanted[i]=wrap(U(U(values[i])+U(1)));
    check(wanted);
    program.scalar.value.float_value=NAN;
    bool rejected=false;try{gpu.unary(program,storage_ref(input),destination,values.size(),{});}catch(const std::exception&){rejected=true;}
    require(rejected,"HIP integer arithmetic accepted nonfinite scalar");check(wanted);
}
void integer_arithmetic() {
    integer_arithmetic_type<int32_t>(DataType::Int32,{INT32_MIN,-16777217,-1,0,1,16777217,INT32_MAX});
    integer_arithmetic_type<int64_t>(DataType::Int64,{INT64_MIN,-9007199254740993ll,-1,0,1,9007199254740993ll,INT64_MAX});
    integer_arithmetic_type<uint32_t>(DataType::UInt32,{0,1,16777217u,2147483649u,UINT32_MAX});
    integer_arithmetic_type<uint8_t>(DataType::UInt8,{0,1,17,127,254,255});
    for(auto dtype : {DataType::Float32,DataType::Int32}) {
        auto empty=Tensor::empty({0},Device::GPU,dtype);
        require(empty.cumsum(0).numel()==0,"HIP empty scan failed");
        require((empty-1).numel()==0,"HIP empty pointwise failed");
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
void native_norm() {
    auto& gpu=backend_ops(GpuBackend::HIP);
    for(auto dtype : {DataType::Float32,DataType::Float16})for(size_t cols : {size_t(1),size_t(17),size_t(65),size_t(257),size_t(1025)}) {
        constexpr size_t rows=7;const size_t count=rows*cols;
        std::vector<float> values(count),weights(cols),biases(cols);
        for(size_t i=0;i<count;++i)values[i]=std::sin(float(i)*.37f)*2;
        for(size_t i=0;i<cols;++i){weights[i]=1.f+.2f*std::cos(float(i));biases[i]=.1f*std::sin(float(i));}
        auto input=Tensor::from_vector(values,{rows,cols},Device::GPU).to(dtype);
        auto weight=Tensor::from_vector(weights,{cols},Device::GPU).to(dtype);
        auto bias=Tensor::from_vector(biases,{cols},Device::GPU).to(dtype);
        auto host=input.to(DataType::Float32).cpu(),hw=weight.to(DataType::Float32).cpu(),hb=bias.to(DataType::Float32).cpu();
        for(bool layer : {false,true}) {
            auto output=Tensor::from_vector(std::vector<float>(count+2,-77),{count+2},Device::GPU).to(dtype);
            auto out=storage_ref(output);out.byte_offset=dtype==DataType::Float32?4:2;
            gpu.nn_norm(storage_ref(input),storage_ref(weight),layer?std::optional(storage_ref(bias)):std::nullopt,out,{rows,cols,1e-5f},{});
            auto observed=output.to(DataType::Float32).cpu();
            require(observed.ptr<float>()[0]==-77 && observed.ptr<float>()[count+1]==-77,"HIP norm overwrote output guards");
            for(size_t row=0;row<rows;++row) {
                double mean=0,squares=0;
                if(layer){for(size_t col=0;col<cols;++col)mean+=host.ptr<float>()[row*cols+col];mean/=cols;}
                for(size_t col=0;col<cols;++col){double x=host.ptr<float>()[row*cols+col]-mean;squares+=x*x;}
                const double deviation=std::sqrt(squares/cols+1e-5f);
                for(size_t col=0;col<cols;++col) {
                    const double wanted=(host.ptr<float>()[row*cols+col]-mean)/deviation*hw.ptr<float>()[col]+(layer?hb.ptr<float>()[col]:0);
                    const double tolerance=dtype==DataType::Float16?2e-3:2e-5;
                    require(std::abs(observed.ptr<float>()[1+row*cols+col]-wanted)<tolerance*std::max(1.,std::abs(wanted)),"HIP layer/RMS norm differs from reference");
                }
            }
            auto inplace=input.clone();
            gpu.nn_norm(storage_ref(inplace),storage_ref(weight),layer?std::optional(storage_ref(bias)):std::nullopt,storage_ref(inplace),{rows,cols,1e-5f},{});
            auto self=inplace.to(DataType::Float32).cpu();
            for(size_t i=0;i<count;++i)require(self.ptr<float>()[i]==observed.ptr<float>()[i+1],"HIP norm inplace changed result");
        }
    }
}
float nn_activation(float x,int activation) {
    if(activation==1)return std::fmax(x,0.f);
    if(activation==2)return .5f*x*(1.f+std::tanh(.7978845608028654f*(x+.044715f*x*x*x)));
    if(activation==3){const float a=std::abs(x*.7071067811865475f),t=1.f/(1.f+.3275911f*a);
        float r=1.f-(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-.284496736f)*t+.254829592f)*t*std::exp(-a*a);
        return .5f*x*(1.f+(x<0?-r:r));}
    if(activation==4)return x/(1.f+std::exp(-x));return x;
}
void native_linear() {
    auto& gpu=backend_ops(GpuBackend::HIP);
    for(auto dtype : {DataType::Float32,DataType::Float16})for(size_t m : {size_t(3),size_t(17),size_t(65)})
    for(bool transpose : {false,true})for(bool batched : {false,true}) {
        constexpr size_t batches=2,n=19,k=33;const size_t output_count=batches*m*n;
        std::vector<float> av(batches*m*k),wv((batched?batches:1)*n*k),bv(n),sv(n),rv(output_count);
        for(size_t i=0;i<av.size();++i)av[i]=.3f*std::sin(float(i)*.17f);
        for(size_t i=0;i<wv.size();++i)wv[i]=.2f*std::cos(float(i)*.13f);
        for(size_t i=0;i<n;++i){bv[i]=.02f*float(int(i%7)-3);sv[i]=.7f+float(i%3)*.2f;}
        for(size_t i=0;i<rv.size();++i)rv[i]=.01f*float(int(i%13)-6);
        auto a=Tensor::from_vector(av,{av.size()},Device::GPU).to(dtype),w=Tensor::from_vector(wv,{wv.size()},Device::GPU).to(dtype);
        auto bias=Tensor::from_vector(bv,{n},Device::GPU).to(dtype),scale=Tensor::from_vector(sv,{n},Device::GPU).to(dtype);
        auto residual=Tensor::from_vector(rv,{rv.size()},Device::GPU).to(dtype);
        auto ah=a.to(DataType::Float32).cpu(),wh=w.to(DataType::Float32).cpu(),bh=bias.to(DataType::Float32).cpu(),sh=scale.to(DataType::Float32).cpu(),rh=residual.to(DataType::Float32).cpu();
        for(int activation=0;activation<=4;++activation)for(bool affine : {false,true}) {
            auto output=Tensor::from_vector(std::vector<float>(output_count+2,-77),{output_count+2},Device::GPU).to(dtype);
            auto out=storage_ref(output);out.byte_offset=dtype==DataType::Float32?4:2;
            gpu.nn_linear(storage_ref(a),storage_ref(w),affine?std::optional(storage_ref(bias)):std::nullopt,
                affine?std::optional(storage_ref(scale)):std::nullopt,affine?std::optional(storage_ref(residual)):std::nullopt,out,
                {batches,m,n,k,transpose,batched,activation},{});
            auto observed=output.to(DataType::Float32).cpu();
            require(observed.ptr<float>()[0]==-77 && observed.ptr<float>()[output_count+1]==-77,"HIP linear overwrote output guards");
            for(size_t batch=0;batch<batches;++batch)for(size_t row=0;row<m;++row)for(size_t col=0;col<n;++col) {
                float value=0;const size_t offset=(batch*m+row)*n+col;
                for(size_t t=0;t<k;++t)value=std::fma(ah.ptr<float>()[(batch*m+row)*k+t],wh.ptr<float>()[(batched?batch*n*k:0)+(transpose?col*k+t:t*n+col)],value);
                if(affine)value+=bh.ptr<float>()[col];value=nn_activation(value,activation);
                if(affine){value*=sh.ptr<float>()[col];value+=rh.ptr<float>()[offset];}
                const float tolerance=dtype==DataType::Float16?1e-3f:3e-5f;
                require(std::isfinite(observed.ptr<float>()[offset+1]) && std::abs(observed.ptr<float>()[offset+1]-value)<tolerance*std::max(1.f,std::abs(value)),"HIP fused linear differs from reference");
            }
        }
    }
}
int main() try {
    GpuBackendScope scope(GpuBackend::HIP);
    unary();integer_arithmetic();native_norm();native_linear();precise_bits();gathered_and_packed();fold_and_host();cache_and_threads();
    std::puts("PASS HIP expressions: native fused linear and layer/RMS norms FP16/FP32 with guards and inplace, native math, exact Int32/Int64/UInt32/UInt8 arithmetic and empty operations, IEEE values, gather bounds, Float16/Bool/UInt8 packed tails, host constants, short/runtime folds, prepare, shape cache and concurrent launches");
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"HIP expression failed: %s\n",e.what());return 1;}
