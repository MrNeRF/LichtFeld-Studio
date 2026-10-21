/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/bilateral_hip.hpp"
#include "training/hip/application_color_tensor.hpp"
#include <algorithm>
namespace lfs::training {
namespace {
using namespace gpu_ops;
using hip::color_require;using hip::color_shape;using hip::color_count;
static void submit(hip::ColorCall& c){c.submit("lfs_hip_application_bilateral");}
static size_t grid(hip::ColorCall& c,In t) {
    color_require(t.ndim()==5&&t.numel()>0,"HIP bilateral grid must be nonempty [N,C,L,H,W]");
    color_count(t.numel());for(unsigned i=0;i<5;++i)c.p.dimensions[i]=color_count(t.shape()[i]);
    c.bind(0,t);return t.shape()[0]*t.shape()[2]*t.shape()[3]*t.shape()[4];
}
static void image(hip::ColorCall& c,In rgb,In offset,const GridSliceParams& p) {
    color_require(c.p.dimensions[0]==1&&rgb.ndim()==3&&p.layout!=Layout::HW,"HIP bilateral slice needs one grid and a three-channel image");
    const bool chw=p.layout==Layout::CHW,exposure=p.transform==GridTransform::ExposureChroma;
    color_require(rgb.shape()[chw?0:2]==3&&c.p.dimensions[1]>=(exposure?9:12),"HIP bilateral channel mismatch");
    c.p.dimensions[5]=color_count(rgb.shape()[chw?1:0]);c.p.dimensions[6]=color_count(rgb.shape()[chw?2:1]);color_count(rgb.numel());
    color_require(rgb.numel()>0,"HIP bilateral image is empty");
    if(offset.is_valid())color_require(offset.numel()>=size_t(exposure?9:12),"HIP bilateral offset is too small");
    c.p.flags[0]=chw;c.p.flags[1]=exposure;c.p.flags[2]=p.warp_aggregate;
}
void forward(In g,In rgb,In offset,Out output,const GridSliceParams& p) {
    color_shape(output,rgb.numel());hip::ColorCall c(0);grid(c,g);image(c,rgb,offset,p);
    c.bind(1,rgb);c.bind(2,offset);c.bind(3,output);submit(c);
}
void backward(In g,In rgb,In grad,In offset,Out grad_grid,Out grad_rgb,const GridSliceParams& p) {
    color_shape(grad,rgb.numel());color_shape(grad_rgb,rgb.numel());color_shape(grad_grid,g.numel());
    hip::ColorCall c(1);grid(c,g);image(c,rgb,offset,p);
    c.bind(1,rgb);c.bind(2,grad);c.bind(3,offset);c.bind(4,grad_grid);c.bind(5,grad_rgb);submit(c);
}
void tv(In grids,Out loss,Out temporary,int norm_n) {
    hip::ColorCall c(2);const auto cells=grid(c,grids);
    color_require(loss.numel()>=1&&temporary.numel()>=std::min((cells+255)/256,size_t{2048}),"HIP bilateral TV buffers are too small");
    c.p.dimensions[7]=norm_n;c.bind(1,loss);c.bind(2,temporary);submit(c);
}
void tv_backward(In grids,Out gradient,float grad_loss,int norm_n) {
    color_shape(gradient,grids.numel());hip::ColorCall c(3);grid(c,grids);c.p.dimensions[7]=norm_n;c.p.values[0]=grad_loss;c.bind(1,gradient);submit(c);
}
void project(Out grids,In mean,In identity,int per_image) {
    hip::ColorCall c(4);grid(c,grids);const auto channels=grids.shape()[1];
    color_require(mean.numel()>=(per_image?grids.shape()[0]*channels:channels)&&identity.numel()>=channels,"HIP bilateral mean/identity buffers are too small");
    c.p.flags[0]=per_image!=0;c.bind(1,mean);c.bind(2,identity);submit(c);
}
void offset(Out sum,Out shared,In identity,In old_mean,In new_mean,float spatial,float inverse) {
    if(!sum.numel())return;for(const auto* t:std::initializer_list<const Tensor*>{&shared,&identity,&old_mean,&new_mean})color_shape(*t,sum.numel());
    hip::ColorCall c(5);c.p.dimensions[0]=color_count(sum.numel());c.p.values[0]=spatial;c.p.values[1]=inverse;
    c.bind(0,sum);c.bind(1,shared);c.bind(2,identity);c.bind(3,old_mean);c.bind(4,new_mean);submit(c);
}
void adam(Out g,Out m,Out v,In gradient,const AdamUpdateParams& p) {
    if(!g.numel())return;for(const auto* t:std::initializer_list<const Tensor*>{&m,&v,&gradient})color_shape(*t,g.numel());
    hip::ColorCall c(6);c.p.dimensions[0]=color_count(g.numel());
    c.p.values[0]=p.lr;c.p.values[1]=p.beta1;c.p.values[2]=p.beta2;c.p.values[3]=p.bc1_rcp;c.p.values[4]=p.bc2_sqrt_rcp;c.p.values[5]=p.eps;
    c.bind(0,g);c.bind(1,m);c.bind(2,v);c.bind(3,gradient);submit(c);
}
void scale(Out m,Out v,float scale1,float scale2) {
    if(!m.numel())return;color_shape(v,m.numel());hip::ColorCall c(7);c.p.dimensions[0]=color_count(m.numel());c.p.values[0]=scale1;c.p.values[1]=scale2;c.bind(0,m);c.bind(1,v);submit(c);
}
void copy_slice(In source,Out destination,size_t a,size_t b,size_t n) {
    color_require(a<=source.numel()&&n<=source.numel()-a&&b<=destination.numel()&&n<=destination.numel()-b,"HIP bilateral transfer exceeds tensor bounds");
    destination.flatten().slice(0,b,b+n).copy_from(source.flatten().slice(0,a,a+n));
}
void upload(Out host,Out device,size_t a,size_t b,size_t n){copy_slice(host,device,a,b,n);}
void download(Out host,Out device,size_t a,size_t b,size_t n){copy_slice(device,host,b,a,n);}
}
const gpu_ops::BilateralOps& hip_bilateral_ops(){static const gpu_ops::BilateralOps ops{forward,backward,tv,tv_backward,project,offset,adam,scale,upload,download,true};return ops;}
}
