/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_tree_api.h"
namespace {
template<bool Spacing>
__global__ void point_tree_query_native(const float* points,const float* sorted,const float* boxes,const int32_t* visit,
    const float* radii,const uint8_t* queries,void* output,hs::PointTreeProgram program) {
    for(size_t rank=size_t(blockIdx.x)*blockDim.x+threadIdx.x;rank<program.points;rank+=size_t(gridDim.x)*blockDim.x) {
        const int32_t index=visit[rank];
        if(index<0 || uint32_t(index)>=program.points) continue;
        const int64_t reference_rank=rank<program.references ? int64_t(rank) : -1;
        if constexpr(Spacing)
            static_cast<float*>(output)[index]=hs::pointTreeSpacing(sorted,boxes,program,points+size_t(index)*3,reference_rank,program.radius);
        else
            static_cast<int32_t*>(output)[index]=(!queries || queries[index]) ?
                hs::pointTreeCount(sorted,boxes,program,points+size_t(index)*3,reference_rank,radii[index],program.max_count) : 0;
    }
}
}
int lfs_hip_point_tree_query(LfsHipContext* context,const LfsHipView* inputs,LfsHipView output,
    const LfsHipPointTree* descriptor,uint32_t spacing,LfsHipError* error) {
    return boundary(error,[&] {
        require(context && inputs && descriptor && spacing<=1,"Invalid HIP point-tree descriptor");
        const auto& p=*descriptor;
        require(p.points<=INT32_MAX && p.references<=p.points && p.levels<=8 &&
                (spacing ? std::isnormal(p.radius) && p.radius>0 : p.max_count>0),"Invalid HIP point-tree parameters");
        size_t boxes_count=0;
        uint32_t children=p.references,levels=0;
        while(children) {
            require(levels<p.levels,"Missing HIP point-tree level");
            const uint32_t count=(children-1)/32+1;
            require(p.level_offset[levels]==boxes_count && p.level_count[levels]==count,"Invalid HIP point-tree level layout");
            boxes_count+=count;++levels;
            if(count<=32) break;
            children=count;
        }
        require(levels==p.levels && (!spacing || !p.points || p.references>0),"Invalid HIP point-tree depth");
        if(!p.points) return;
        const auto checked_bytes=[](size_t n,size_t width) {
            require(n<=SIZE_MAX/width,"HIP point-tree byte range overflow");return n*width;
        };
        const size_t sizes[]{checked_bytes(p.points,12),checked_bytes(p.references,12),checked_bytes(boxes_count,24),
                             checked_bytes(p.points,4),spacing?0:checked_bytes(p.points,4),spacing?0:size_t(p.points)};
        const auto bytes=checked_bytes(p.points,4);
        validate(context,output.buffer,output.offset,bytes);
        require(output.offset%4==0,"Misaligned HIP point-tree output");
        for(unsigned i=0;i<6;++i) {
            if(!sizes[i] || (i==5 && !inputs[i].buffer)) continue;
            validate(context,inputs[i].buffer,inputs[i].offset,sizes[i]);
            require(i==5 || inputs[i].offset%4==0,"Misaligned HIP point-tree input");
            require(inputs[i].buffer!=output.buffer || inputs[i].offset>=output.offset+bytes ||
                    output.offset>=inputs[i].offset+sizes[i],"HIP point-tree output aliases input");
        }
        const auto pointer=[](LfsHipView v)->void* {return v.buffer ? static_cast<char*>(v.buffer->data)+v.offset : nullptr;};
        hs::PointTreeProgram program{};
        program.points=p.points;program.references=p.references;program.levels=p.levels;
        std::copy_n(p.level_offset,8,program.level_offset);std::copy_n(p.level_count,8,program.level_count);
        program.radius=p.radius;program.max_count=p.max_count;
        DeviceScope device(context->device);
        const auto launch=[&]<bool Spacing>() {
            hipLaunchKernelGGL((point_tree_query_native<Spacing>),dim3(blocks(p.points)),dim3(256),0,context->stream,
                static_cast<const float*>(pointer(inputs[0])),static_cast<const float*>(pointer(inputs[1])),
                static_cast<const float*>(pointer(inputs[2])),static_cast<const int32_t*>(pointer(inputs[3])),
                static_cast<const float*>(pointer(inputs[4])),static_cast<const uint8_t*>(pointer(inputs[5])),pointer(output),program);
        };
        if(spacing) launch.template operator()<true>();else launch.template operator()<false>();
        check(hipGetLastError());
    });
}
