/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/detail/fused_pointwise.hpp>
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
using lfs::core::internal::PointwiseOp;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main() try {
    const auto configured=configured_gpu_backend();
    {
        const GpuBackendScope scope(GpuBackend::HIP);
        auto& backend=internal::backend_ops(GpuBackend::HIP);
        for(auto [segments,width]:std::array<std::pair<size_t,size_t>,10>{{{777,3},{777,4},{65,31},{65,32},{65,33},{3,257},{3,5003},{1,300001},{1,1},{1,32}}}) {
            const size_t count=segments*width;
            std::vector<float> x(count),y(count);
            for(size_t i=0;i<count;++i){x[i]=float(int(i%67)-33)*.03125f;y[i]=float(int(i%19)-9)*.015625f;}
            auto input=Tensor::from_vector(x,{count},Device::GPU),rhs=Tensor::from_vector(y,{count},Device::GPU);
            auto guarded=Tensor::full({segments+8},12345.f,Device::GPU);
            auto output=guarded.slice(0,4,4+segments);
            const auto a=internal::storage_ref(input),b=internal::storage_ref(rhs),out=internal::storage_ref(output);
            tensor_ops::FusedPointwiseOpChain chain{};chain.num_ops=3;
            chain.ops[0].kind=uint8_t(PointwiseOp::Square);
            chain.ops[1].kind=uint8_t(PointwiseOp::MulScalar);chain.ops[1].scalar=-.5f;
            chain.ops[2].kind=uint8_t(PointwiseOp::AddTensor);chain.ops[2].rhs=reinterpret_cast<const float*>(static_cast<const char*>(b.data)+b.byte_offset);
            const std::array leases{b};
            for(auto op:{ReduceOp::Sum,ReduceOp::Mean,ReduceOp::Min,ReduceOp::Max}) {
                internal::ReduceProgram program{};program.op=op;program.result_dtype=DataType::Float32;
                if(segments==1)backend.fused_transform_reduce(a,out,count,chain,program,leases,{});
                else backend.fused_segmented_transform_reduce(a,out,segments,width,chain,program,leases,{});
                auto actual=guarded.cpu().to_vector();
                for(size_t j=0;j<4;++j)require(actual[j]==12345.f && actual[segments+4+j]==12345.f,"HIP fused reduction overwrote output guards");
                for(size_t s=0;s<segments;++s) {
                    double reference=op==ReduceOp::Min?INFINITY:op==ReduceOp::Max?-INFINITY:0;
                    for(size_t j=0;j<width;++j) {
                        const size_t at=s*width+j;const float value=(x[at]*x[at])*(-.5f)+y[at];
                        if(op==ReduceOp::Min)reference=std::min(reference,double(value));else if(op==ReduceOp::Max)reference=std::max(reference,double(value));else reference+=value;
                    }
                    if(op==ReduceOp::Mean)reference/=double(width);
                    require(std::abs(actual[s+4]-reference)<=2e-6*std::max(1.,std::abs(reference)),"HIP fused reduction differs from independent CPU reference");
                }
            }
            internal::ReduceProgram program{};program.op=ReduceOp::Sum;program.result_dtype=DataType::Float32;
            bool rejected=false;try{backend.fused_segmented_transform_reduce(a,out,segments,width,chain,program,{},{});}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"HIP fused reduction accepted missing rhs lease");
            rejected=false;try{backend.fused_transform_reduce(a,a,count,chain,program,leases,{});}catch(const std::exception&){rejected=true;}
            require(rejected,"HIP fused reduction accepted aliased output");
        }
        // The public Tensor expression exercises the facade's segmented fusion.
        auto values=Tensor::from_vector(std::vector<float>{1,2,3,-1,-2,-3},{2,3},Device::GPU);
        auto result=values.square().sum(1).cpu().to_vector();
        require(result==std::vector<float>{14,14},"HIP public Tensor segmented fusion failed");
    }
    require(configured_gpu_backend()==configured && bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP fused reduction changed configuration or leaked native storage");
    std::puts("PASS HIP fused reductions: short/long/split segments, full reductions, four operations, native rhs leases, output guards and public Tensor facade");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP FUSED REDUCTION FAILED: %s\n",e.what());return 1;}
