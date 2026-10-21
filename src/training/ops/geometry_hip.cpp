/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/geometry_hip.hpp"
#include "training/hip/application_color_tensor.hpp"
#include <algorithm>
#include <cmath>
namespace lfs::training {
namespace {
using namespace gpu_ops;using hip::color_require;using hip::color_shape;using hip::color_count;
size_t image(hip::ColorCall& c,In t) {
    color_require(t.ndim()==2&&t.numel()>0,"HIP geometry requires a nonempty HW image");color_count(t.numel()*3);
    c.p.dimensions[0]=color_count(t.shape()[1]);c.p.dimensions[1]=color_count(t.shape()[0]);return t.numel();
}
void normal_shape(In t,In hw){color_require(t.ndim()==3&&t.shape()[0]==3&&t.shape()[1]==hw.shape()[0]&&t.shape()[2]==hw.shape()[1],"HIP geometry requires CHW normals matching the image");}
void scratch(In loss,In partials,size_t pixels,bool depth) {
    color_require(loss.numel()>=1&&partials.numel()>=(depth?kernels::depth_loss_partial_count(pixels):kernels::normal_loss_partial_count(pixels)),"HIP geometry scratch or loss buffer is too small");
}
void optional_weight(hip::ColorCall& c,unsigned slot,In weight,size_t n){if(weight.is_valid())color_shape(weight,n);c.bind(slot,weight);}
void submit(hip::ColorCall& c){c.submit("lfs_hip_application_geometry");}
void depth(In depth,In alpha,In target,In weight,Out gd,Out ga,Out loss,Out partials,const DepthParams& p) {
    hip::ColorCall c(0);const auto n=image(c,depth);
    for(const auto* t:std::initializer_list<const Tensor*>{&alpha,&target,&gd,&ga})color_shape(*t,n);scratch(loss,partials,n,true);
    c.bind(0,depth);c.bind(1,alpha);c.bind(2,target);c.bind(3,gd);c.bind(4,ga);c.bind(5,loss);c.bind(6,partials);optional_weight(c,7,weight,n);
    c.p.values[0]=p.weight;c.p.values[1]=p.gradient_weight;c.p.values[2]=p.prior_quantization_step;
    if(p.anchor){c.p.flags[0]=p.anchor->valid;c.p.dimensions[2]=p.anchor->model;c.p.values[3]=p.anchor->scale;c.p.values[4]=p.anchor->shift;c.p.values[5]=p.anchor->floor;}submit(c);
}
void normal(In normal,In alpha,In target,In weight,Out gn,Out loss,Out partials,float strength) {
    hip::ColorCall c(1);const auto n=image(c,alpha);normal_shape(normal,alpha);normal_shape(target,alpha);color_shape(gn,n*3);scratch(loss,partials,n,false);
    c.bind(0,normal);c.bind(1,alpha);c.bind(2,target);c.bind(3,gn);c.bind(5,loss);c.bind(6,partials);optional_weight(c,7,weight,n);c.p.values[0]=strength;submit(c);
}
void coupled(unsigned operation,In normal,In depth,In alpha,In weight,Out gn,Out gd,Out ga,Out loss,Out partials,const GeometryCamera& k,float strength) {
    hip::ColorCall c(operation);const auto n=image(c,depth);normal_shape(normal,depth);color_shape(alpha,n);color_shape(gd,n);color_shape(ga,n);scratch(loss,partials,n,false);
    c.bind(0,normal);c.bind(1,depth);c.bind(2,alpha);if(operation==2){color_shape(gn,n*3);c.bind(3,gn);}c.bind(4,gd);c.bind(5,ga);c.bind(6,loss);c.bind(8,partials);optional_weight(c,9,weight,n);
    if(k.rays.is_valid())color_require(k.rays.ndim()==3&&k.rays.shape()[0]==depth.shape()[0]&&k.rays.shape()[1]==depth.shape()[1]&&k.rays.shape()[2]==3,"HIP geometry camera rays must be HWC3");c.bind(10,k.rays);
    c.p.values[0]=k.fx;c.p.values[1]=k.fy;c.p.values[2]=k.cx;c.p.values[3]=k.cy;c.p.values[4]=strength;c.p.flags[0]=k.wrap_horizontal;submit(c);
}
void consistency(In n,In d,In a,In w,Out gn,Out gd,Out ga,Out l,Out p,GeometryCamera k,float s){coupled(2,n,d,a,w,gn,gd,ga,l,p,k,s);}
void prior_depth(In n,In d,In a,In w,Out gd,Out ga,Out l,Out p,GeometryCamera k,float s){Tensor unused;coupled(3,n,d,a,w,unused,gd,ga,l,p,k,s);}
std::vector<AnchorSample> anchor(In points,In view,In prior,const AnchorParams& params) {
    color_require(points.ndim()==2&&points.shape()[1]==3&&view.numel()==16&&prior.ndim()==2&&prior.numel()>0,"Invalid HIP anchor tensor shapes");color_count(points.numel());color_count(prior.numel());
    if(!points.numel())return {};
    hip::ApplicationCall owner;LfsHipAnchorCall p{};p.struct_bytes=sizeof(p);p.capacity=262144;p.device=owner.queue.device;p.queue=owner.queue.native_queue;
    p.points=owner.bind(points,core::DataType::Float32);p.point_count=points.shape()[0];p.view=owner.bind(view,core::DataType::Float32);p.prior=owner.bind(prior,core::DataType::Float32);p.width=color_count(prior.shape()[1]);p.height=color_count(prior.shape()[0]);p.model=params.projection.model;
    p.intrinsics[0]=params.intrinsics.fx;p.intrinsics[1]=params.intrinsics.fy;p.intrinsics[2]=params.intrinsics.cx;p.intrinsics[3]=params.intrinsics.cy;p.near_plane=params.near_plane;
    std::copy_n(params.aabb_lo.data(),3,p.aabb_lo);std::copy_n(params.aabb_hi.data(),3,p.aabb_hi);std::copy_n(params.projection.radial,6,p.radial);std::copy_n(params.projection.tangential,2,p.tangential);std::copy_n(params.projection.thin_prism,4,p.thin_prism);
    for(float x:p.intrinsics)color_require(std::isfinite(x),"Nonfinite HIP anchor intrinsics");
    std::vector<AnchorSample> samples(p.capacity);uint32_t count=0;LfsHipError error{};
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_anchor)>("lfs_hip_application_anchor");
    if(int code=fn(&p,reinterpret_cast<float*>(samples.data()),&count,&error))throw hip::TrainingError(code,error);color_require(count<=samples.size(),"Invalid HIP anchor result count");samples.resize(count);return samples;
}
}
const gpu_ops::GeometryLossOps& hip_geometry_ops(){static const gpu_ops::GeometryLossOps ops{depth,normal,consistency,prior_depth,anchor};return ops;}
}
