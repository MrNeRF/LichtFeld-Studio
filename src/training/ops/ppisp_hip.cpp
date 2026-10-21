/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/ppisp_hip.hpp"
#include "training/hip/application_color_tensor.hpp"
namespace lfs::training {
namespace {
using namespace gpu_ops;using hip::color_count;using hip::color_shape;using hip::color_require;
static void submit(hip::ColorCall& c){c.submit("lfs_hip_application_ppisp");}
static void parameters(hip::ColorCall& c,const PPISPInputs& p,int cameras,int frames) {
    color_require(cameras>=0&&frames>=0,"HIP ISP camera/frame counts must be nonnegative");
    color_require(p.exposure.numel()>=size_t(frames)&&p.vignetting.numel()>=size_t(cameras)*15&&p.color.numel()>=size_t(frames)*8&&p.crf.numel()>=size_t(cameras)*12,"HIP ISP parameter buffers are too small");
    c.bind(0,p.exposure);c.bind(1,p.vignetting);c.bind(2,p.color);c.bind(3,p.crf);c.p.dimensions[6]=cameras;c.p.dimensions[7]=frames;
}
static void image(hip::ColorCall& c,In rgb) {
    color_require(rgb.ndim()==3&&rgb.shape()[0]==3&&rgb.numel()>0,"HIP ISP requires nonempty CHW RGB");
    color_count(rgb.numel());c.p.dimensions[0]=color_count(rgb.shape()[1]);c.p.dimensions[1]=color_count(rgb.shape()[2]);c.bind(4,rgb);
}
void forward(const PPISPInputs& p,In rgb,Out corrected,const PPISPRegion& r) {
    color_shape(corrected,rgb.numel());hip::ColorCall c(0);parameters(c,p,r.cameras,r.frames);image(c,rgb);c.bind(5,corrected);
    c.p.dimensions[2]=r.x_offset;c.p.dimensions[3]=r.y_offset;c.p.dimensions[4]=r.full_width>0?r.full_width:c.p.dimensions[1];c.p.dimensions[5]=r.full_height;c.p.dimensions[8]=r.camera_index;c.p.dimensions[9]=r.frame_index;submit(c);
}
void backward(const PPISPInputs& p,In rgb,In grad,const PPISPOutputs& g,Out grad_rgb,int cameras,int frames,int camera,int frame) {
    color_shape(grad,rgb.numel());color_shape(grad_rgb,rgb.numel());
    color_shape(g.exposure,p.exposure.numel());color_shape(g.vignetting,p.vignetting.numel());color_shape(g.color,p.color.numel());color_shape(g.crf,p.crf.numel());
    hip::ColorCall c(1);parameters(c,p,cameras,frames);image(c,rgb);c.bind(5,grad);c.bind(6,g.exposure);c.bind(7,g.vignetting);c.bind(8,g.color);c.bind(9,g.crf);c.bind(10,grad_rgb);c.p.dimensions[8]=camera;c.p.dimensions[9]=frame;submit(c);
}
static void adam_values(hip::ColorCall& c,const PPISPAdamUpdateParams& p){c.p.values[0]=p.lr;c.p.values[1]=p.beta1;c.p.values[2]=p.beta2;c.p.values[3]=p.bc1_rcp;c.p.values[4]=p.bc2_sqrt_rcp;c.p.values[5]=p.eps;}
static void adam_group(hip::ColorCall& c,const PPISPAdamGroup& g,unsigned i) {
    if(!g.parameter.is_valid()||!g.parameter.numel())return;
    color_shape(g.moment1,g.parameter.numel());color_shape(g.moment2,g.parameter.numel());color_shape(g.gradient,g.parameter.numel());
    c.p.dimensions[i]=color_count(g.parameter.numel());c.bind(4*i,g.parameter);c.bind(4*i+1,g.moment1);c.bind(4*i+2,g.moment2);c.bind(4*i+3,g.gradient);
}
void adam(const PPISPAdamGroup& g,const PPISPAdamUpdateParams& p) {
    if(!g.parameter.is_valid()||!g.parameter.numel())return;
    hip::ColorCall c(2);adam_group(c,g,0);adam_values(c,p);submit(c);
}
void batch(const std::array<PPISPAdamGroup,4>& groups,const PPISPAdamUpdateParams& p) {
    hip::ColorCall c(3);size_t count=0;for(unsigned i=0;i<4;++i){adam_group(c,groups[i],i);count+=c.p.dimensions[i];}color_count(count);if(!count)return;adam_values(c,p);submit(c);
}
void regularization(In parameters,Out gradient,Out loss,float center,float channel,float nonpositive) {
    if(!parameters.numel()||(!gradient.is_valid()&&!loss.is_valid()))return;
    color_require(parameters.numel()%15==0,"HIP ISP vignetting shape is not a multiple of 15");if(gradient.is_valid())color_shape(gradient,parameters.numel());if(loss.is_valid())color_require(loss.numel()>=1,"HIP ISP regularization loss is empty");
    hip::ColorCall c(4);c.p.dimensions[0]=color_count(parameters.numel()/15);c.p.values[0]=center;c.p.values[1]=channel;c.p.values[2]=nonpositive;c.bind(0,parameters);c.bind(1,gradient);c.bind(2,loss);submit(c);
}
void project(Out exposure,Out color) {
    if(!exposure.numel())return;color_shape(color,exposure.numel()*8);
    hip::ColorCall c(5);c.p.dimensions[0]=color_count(exposure.numel());c.bind(0,exposure);c.bind(1,color);submit(c);
}
void initialize(const PPISPOutputs& p) {
    color_require(p.vignetting.numel()>0&&p.vignetting.numel()%15==0&&p.exposure.numel()>0,"HIP ISP initialization needs camera/frame groups");
    const auto cameras=p.vignetting.numel()/15,frames=p.exposure.numel();color_shape(p.color,frames*8);color_shape(p.crf,cameras*12);
    hip::ColorCall c(6);c.p.dimensions[0]=color_count(cameras);c.p.dimensions[1]=color_count(frames);c.bind(0,p.exposure);c.bind(1,p.vignetting);c.bind(2,p.color);c.bind(3,p.crf);submit(c);
}
}
const gpu_ops::PPISPOps& hip_ppisp_ops(){static const gpu_ops::PPISPOps ops{forward,backward,adam,batch,regularization,project,initialize,true};return ops;}
}
