/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/mrnf_hip.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_mrnf_api.h"
#include <algorithm>
#include <climits>
#include <cmath>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
static void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
static uint32_t count(size_t n){require(n<=size_t(INT_MAX)/3,"HIP MRNF count exceeds native index range");return uint32_t(n);}
static void shape(In t,size_t n){require(t.is_valid()&&t.numel()==n,"HIP MRNF operand shape mismatch");}
static void optional(In t,size_t n){if(t.is_valid())shape(t,n);}
struct Call {
    hip::ApplicationCall owner;LfsHipMrnfCall p{};
    Call(unsigned operation,size_t n){p.struct_bytes=sizeof(p);p.operation=operation;p.count=count(n);p.device=owner.queue.device;p.queue=owner.queue.native_queue;}
    void bind(unsigned i,In t,DataType dtype=DataType::Float32){p.bindings[i]=owner.bind(t,dtype);}
    void submit(){if(!p.count)return;const auto fn=owner.module->symbol<decltype(&lfs_hip_application_mrnf)>("lfs_hip_application_mrnf");LfsHipError error{};if(int code=fn(&p,&error))throw hip::TrainingError(code,error);}
};
void noise(Out means,In opacity,In visibility,In frozen,const MrnfNoiseParams& args) {
    require(means.ndim()==2&&means.shape()[1]==3,"HIP MRNF means must be Nx3");
    const auto n=means.shape()[0];shape(opacity,n);shape(visibility,n);
    require(std::isfinite(args.lr_mean)&&std::isfinite(args.noise_weight)&&std::isfinite(args.median_scale)&&args.median_scale>=0,"Invalid HIP MRNF noise parameters");
    Call c(0,n);c.bind(0,means);c.bind(1,opacity);c.bind(2,visibility);c.bind(3,frozen,DataType::Bool);
    c.p.mask_count=frozen.is_valid()?count(frozen.numel()):0;c.p.seed=args.seed;
    c.p.value[0]=args.lr_mean;c.p.value[1]=args.noise_weight;c.p.value[2]=args.median_scale;c.submit();
}
void decay(Out opacity,Out scales,In frozen,const DecayParams& args) {
    require(scales.ndim()==2&&scales.shape()[1]==3,"HIP MRNF scales must be Nx3");
    const auto n=scales.shape()[0];shape(opacity,n);optional(args.rendered_count,n);
    require(std::isfinite(args.opacity_decay)&&std::isfinite(args.scale_decay)&&std::isfinite(args.train_t),"Invalid HIP MRNF decay parameters");
    Call c(1,n);c.bind(0,opacity);c.bind(1,scales);c.bind(2,frozen,DataType::Bool);c.bind(3,args.rendered_count);
    c.p.mask_count=frozen.is_valid()?count(frozen.numel()):0;
    c.p.value[0]=args.opacity_decay;c.p.value[1]=args.scale_decay;c.p.value[2]=args.train_t;c.submit();
}
Bounds bounds(In means,float percentile) {
    require(means.ndim()==2&&means.shape()[1]==3&&means.shape()[0]>0&&std::isfinite(percentile)&&percentile>=0&&percentile<=1,"Invalid HIP MRNF bounds input");
    const auto n=means.shape()[0];count(n);
    const auto lo=size_t(((1.f-percentile)*.5f)*float(n-1));
    const auto hi=size_t((1.f-(1.f-percentile)*.5f)*float(n-1));
    Bounds result{};std::array<float,3> ext{};
    for(int axis=0;axis<3;++axis) {
        auto sorted=means.slice(1,axis,axis+1).contiguous().reshape(core::TensorShape{n}).sort(0,false).first;
        const float low=sorted.slice(0,lo,lo+1).cpu().item<float>();
        const float high=sorted.slice(0,hi,hi+1).cpu().item<float>();
        result.center[axis]=(low+high)*.5f;result.extent[axis]=(high-low)*.5f;ext[axis]=result.extent[axis];
    }
    std::sort(ext.begin(),ext.end());result.median_size=2*ext[1];result.max_extent=ext[2];return result;
}
void gumbel(GumbelTopKScratch*,In weights,Out indices,const GumbelParams& args) {
    const auto n=weights.numel(),k=indices.numel();count(n);
    require(k<=n&&indices.dtype()==DataType::Int64&&weights.dtype()==DataType::Float32,"Invalid HIP MRNF Gumbel input");
    require(!args.compact_sparse||args.known_nnz<=n,"HIP MRNF Gumbel known_nnz exceeds count");
    if(!k)return;
    if(k==n){Call c(7,n);c.bind(0,indices,DataType::Int64);c.submit();return;}
    Tensor sources,compact_indices;size_t active=n;
    if(args.compact_sparse) {
        sources=(weights>0.f).nonzero().flatten();active=sources.numel();
        require(!args.known_nnz||args.known_nnz==active,"HIP MRNF known_nnz differs from positive-weight count");
    }
    const bool compact=args.compact_sparse&&active>=k&&active<n;
    const auto sort_count=compact?active:n;
    auto keys=Tensor::empty({sort_count},core::Device::GPU);
    Call c(compact?5:4,sort_count);c.bind(0,weights);c.p.seed=args.seed;
    if(compact){compact_indices=sources.to(DataType::Int32);c.bind(1,compact_indices,DataType::Int32);c.bind(2,keys);}
    else c.bind(1,keys);
    c.submit();
    auto selected=keys.sort(0,true).second.slice(0,0,k);
    if(compact)selected=sources.index_select(0,selected);
    indices.copy_from(selected.reshape(indices.shape()));
}
void fold(Out maximum,Out densification) {
    shape(densification,maximum.numel()*2);Call c(2,maximum.numel());c.bind(0,maximum);c.bind(1,densification);c.submit();
}
size_t compact(In mask,Out indices,size_t requested) {
    if(!mask.numel()||!requested)return 0;
    require(indices.dtype()==DataType::Int64,"HIP MRNF compaction needs Int64 indices");
    auto found=mask.nonzero().flatten();const auto total=found.numel(),written=std::min({total,requested,indices.numel()});
    if(written)indices.flatten().slice(0,0,written).copy_from(found.slice(0,0,written));return total;
}
void prune(In means,In scale,Out mask,std::array<float,3> center,float maximum,float logmax) {
    require(means.ndim()==2&&means.shape()[1]==3,"HIP MRNF prune means must be Nx3");
    const auto n=means.shape()[0];shape(scale,n);shape(mask,n);
    Call c(6,n);c.bind(0,means);c.bind(1,scale);c.bind(2,mask,DataType::Bool);
    std::copy(center.begin(),center.end(),c.p.value);c.p.value[3]=maximum;c.p.value[4]=logmax;c.submit();
}
void parent(In opacity,In visibility,In active,In trainable,In edge,Out weights) {
    const auto n=opacity.numel();shape(visibility,n);shape(weights,n);optional(active,n);optional(trainable,n);optional(edge,n);
    Call c(3,n);c.bind(0,opacity);c.bind(1,visibility);c.bind(2,active,DataType::Bool);c.bind(3,trainable,DataType::Bool);c.bind(4,edge);c.bind(5,weights);c.submit();
}
}
const gpu_ops::MrnfOps& hip_mrnf_ops(){static const gpu_ops::MrnfOps ops{noise,decay,bounds,gumbel,fold,compact,prune,parent};return ops;}
}
