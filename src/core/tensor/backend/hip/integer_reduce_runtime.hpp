/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "integer_reduce_api.h"
namespace {
template<class In,class Out,unsigned Op> __global__ void typed_axis_reduce(const In* input,Out* output,LfsHipLayout kept,LfsHipLayout reduced) {
    __shared__ uint64_t scratch[256];
    for(size_t row=blockIdx.x;row<kept.elements;row+=gridDim.x) {
        uint64_t value=Op==6?1:0;
        if constexpr(Op==2)value=static_cast<uint64_t>(std::numeric_limits<Out>::lowest());
        if constexpr(Op==3)value=static_cast<uint64_t>(std::numeric_limits<Out>::max());
        const auto offset=layout_index(row,kept);
        for(size_t i=threadIdx.x;i<reduced.elements;i+=blockDim.x) {
            const auto source=input[offset+layout_index(i,reduced)];
            if constexpr(Op==0)value+=static_cast<uint64_t>(source);
            if constexpr(Op==2){const Out a=static_cast<Out>(value),b=static_cast<Out>(source);value=static_cast<uint64_t>(a>b?a:b);}
            if constexpr(Op==3){const Out a=static_cast<Out>(value),b=static_cast<Out>(source);value=static_cast<uint64_t>(a<b?a:b);}
            if constexpr(Op==5)value|=source!=In(0);
            if constexpr(Op==6)value&=source!=In(0);
        }
        scratch[threadIdx.x]=value;__syncthreads();
        for(unsigned step=128;step;step>>=1) {
            if(threadIdx.x<step) {
                const auto other=scratch[threadIdx.x+step];value=scratch[threadIdx.x];
                if constexpr(Op==0)value+=other;
                if constexpr(Op==2)value=static_cast<uint64_t>(static_cast<Out>(value)>static_cast<Out>(other)?static_cast<Out>(value):static_cast<Out>(other));
                if constexpr(Op==3)value=static_cast<uint64_t>(static_cast<Out>(value)<static_cast<Out>(other)?static_cast<Out>(value):static_cast<Out>(other));
                if constexpr(Op==5)value|=other;
                if constexpr(Op==6)value&=other;
                scratch[threadIdx.x]=value;
            }
            __syncthreads();
        }
        if(threadIdx.x==0)output[row]=static_cast<Out>(scratch[0]);
        __syncthreads();
    }
}
}
extern "C" int lfs_hip_reduce_typed_layout(LfsHipContext* context,LfsHipView input,LfsHipView output,const LfsHipLayout* source,uint32_t axes,uint32_t operation,uint32_t input_dtype,uint32_t output_dtype,LfsHipError* error) {
    return boundary(error,[&] {
        require(context&&source&&(operation==0||operation==1||operation==2||operation==3||operation==5||operation==6),"Invalid HIP typed axis reduction");
        const bool logical=operation==5||operation==6;
        const bool half_numeric=!logical && input_dtype==LFS_HIP_F16;
        require(operation!=1 || half_numeric,"HIP integer mean is unsupported");
        require(logical?output_dtype==LFS_HIP_BOOL:half_numeric?(output_dtype==LFS_HIP_F16 || output_dtype==LFS_HIP_F32):
            ((input_dtype==LFS_HIP_I32&&(output_dtype==LFS_HIP_I32||output_dtype==LFS_HIP_I64))||
             (input_dtype==LFS_HIP_I64&&output_dtype==LFS_HIP_I64)||
             (input_dtype==LFS_HIP_BOOL&&((operation==0&&output_dtype==LFS_HIP_I64)||(operation!=0&&output_dtype==LFS_HIP_BOOL)))),"Unsupported HIP typed reduction dtypes");
        const size_t iw=typed_bytes(input_dtype,1),ow=typed_bytes(output_dtype,1);
        const auto input_bytes=layout_span(*source,unsigned(iw),false);
        require((axes>>source->rank)==0,"HIP typed reduction axis exceeds rank");
        if(!axes)axes=(1u<<source->rank)-1;
        LfsHipLayout kept{},reduced{};kept.elements=reduced.elements=1;
        for(unsigned axis=0;axis<source->rank;++axis) {
            auto& layout=(axes&(1u<<axis))?reduced:kept;
            layout.dims[layout.rank]=source->dims[axis];layout.strides[layout.rank++]=source->strides[axis];
        }
        auto product=[](LfsHipLayout& layout) {
            for(unsigned i=0;i<layout.rank;++i)if(!layout.dims[i]){layout.elements=0;return;}
            for(unsigned i=0;i<layout.rank;++i){require(layout.dims[i]<=SIZE_MAX/layout.elements,"HIP typed reduction count overflow");layout.elements*=layout.dims[i];}
        };product(kept);product(reduced);
        require(!kept.elements||reduced.elements||(operation!=2&&operation!=3),"HIP integer extrema require a nonempty axis");
        const auto output_bytes=typed_bytes(output_dtype,kept.elements);
        validate(context,input.buffer,input.offset,input_bytes);validate(context,output.buffer,output.offset,output_bytes);
        require(input.offset%iw==0&&output.offset%ow==0,"Unaligned HIP typed reduction");
        require(input.buffer!=output.buffer||!input_bytes||!output_bytes||input.offset>=output.offset+output_bytes||output.offset>=input.offset+input_bytes,"HIP typed reduction input/output overlap");
        if(!kept.elements)return;
        DeviceScope selected(context->device);
        const void* in=static_cast<char*>(input.buffer->data)+input.offset;void* out=static_cast<char*>(output.buffer->data)+output.offset;
        if(half_numeric) {
            if(!context->reduction_workspace)check(hipMalloc(&context->reduction_workspace,reduction_workspace_floats*sizeof(float)));
            const auto dispatch=[&]<class O>() {
                switch(operation) {
                case 0:launch_axis<LFS_HIP_REDUCE_SUM>(context,static_cast<const __half*>(in),static_cast<O*>(out),kept,reduced,false);break;
                case 1:launch_axis<LFS_HIP_REDUCE_SUM>(context,static_cast<const __half*>(in),static_cast<O*>(out),kept,reduced,true);break;
                case 2:launch_axis<LFS_HIP_REDUCE_MAX>(context,static_cast<const __half*>(in),static_cast<O*>(out),kept,reduced,false);break;
                case 3:launch_axis<LFS_HIP_REDUCE_MIN>(context,static_cast<const __half*>(in),static_cast<O*>(out),kept,reduced,false);break;
                }
            };
            if(output_dtype==LFS_HIP_F16)dispatch.template operator()<__half>();else dispatch.template operator()<float>();
            return;
        }
        auto launch=[&]<class I,class O,unsigned Op>() {
            hipLaunchKernelGGL((typed_axis_reduce<I,O,Op>),dim3(unsigned(std::min<size_t>(kept.elements,65535))),dim3(256),0,context->stream,static_cast<const I*>(in),static_cast<O*>(out),kept,reduced);
        };
        auto numeric=[&]<class I,class O>() {
            switch(operation){case 0:launch.template operator()<I,O,0>();break;case 2:launch.template operator()<I,O,2>();break;case 3:launch.template operator()<I,O,3>();break;}
        };
        if(logical) {
            auto dispatch=[&]<class I>(){if(operation==5)launch.template operator()<I,uint8_t,5>();else launch.template operator()<I,uint8_t,6>();};
            switch(input_dtype){case LFS_HIP_F32:dispatch.template operator()<float>();break;case LFS_HIP_F16:dispatch.template operator()<__half>();break;case LFS_HIP_I32:dispatch.template operator()<int32_t>();break;case LFS_HIP_I64:dispatch.template operator()<int64_t>();break;case LFS_HIP_U32:dispatch.template operator()<uint32_t>();break;case LFS_HIP_U8:case LFS_HIP_BOOL:dispatch.template operator()<uint8_t>();break;default:throw std::invalid_argument("Invalid HIP logical reduction dtype");}
        } else if(input_dtype==LFS_HIP_I32) {
            if(output_dtype==LFS_HIP_I32)numeric.template operator()<int32_t,int32_t>();else numeric.template operator()<int32_t,int64_t>();
        } else if(input_dtype==LFS_HIP_I64)numeric.template operator()<int64_t,int64_t>();
        else if(output_dtype==LFS_HIP_BOOL)numeric.template operator()<uint8_t,uint8_t>();
        else numeric.template operator()<uint8_t,int64_t>();
        check(hipGetLastError());
    });
}
