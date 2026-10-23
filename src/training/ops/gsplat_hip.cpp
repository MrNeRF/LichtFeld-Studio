/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/gsplat_hip.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/sh.hpp"
#include "training/hip/application_tensor.hpp"
#include "training/hip/application_gsplat_api.h"
#include "core/sh_layout.hpp"
#include "core/tensor_execution.hpp"
#include "diagnostics/vram_profiler.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <climits>
namespace lfs::training {
namespace {
using namespace gpu_ops;using core::DataType;using core::Device;
void require(bool condition,const char* message){if(!condition)throw std::invalid_argument(message);}
void shape(In t,size_t n){require(t.is_valid()&&t.numel()==n,"HIP Gsplat tensor extent mismatch");}
int narrow(size_t n){require(n<=INT_MAX,"HIP Gsplat dimension exceeds int32");return int(n);}
struct Implementation {
    hip::ApplicationCall owner;LfsHipApplicationGsplat* handle=nullptr;Tensor packed,alpha,rays;GsplatParams params{};
    int count=0,width=0,height=0,channels=0;bool live=false;
#define ENTRY(name) decltype(&::name) name=nullptr
    ENTRY(lfs_hip_application_gsplat_create);ENTRY(lfs_hip_application_gsplat_forward);ENTRY(lfs_hip_application_gsplat_backward);ENTRY(lfs_hip_application_gsplat_release);ENTRY(lfs_hip_application_gsplat_info);ENTRY(lfs_hip_application_gsplat_destroy);
    ENTRY(lfs_hip_application_gsplat_pop_scores);
#undef ENTRY
    Implementation() {
#define LOAD(name) name=owner.module->symbol<decltype(name)>(#name)
        LOAD(lfs_hip_application_gsplat_create);LOAD(lfs_hip_application_gsplat_forward);LOAD(lfs_hip_application_gsplat_backward);LOAD(lfs_hip_application_gsplat_release);LOAD(lfs_hip_application_gsplat_info);LOAD(lfs_hip_application_gsplat_destroy);
        LOAD(lfs_hip_application_gsplat_pop_scores);
#undef LOAD
        LfsHipError e{};if(int c=lfs_hip_application_gsplat_create(owner.queue.device,owner.queue.native_queue,2ull*1024*1024*1024,&handle,&e))throw hip::TrainingError(c,e);
    }
    void release(){if(!handle)return;LfsHipError e{};if(int c=lfs_hip_application_gsplat_release(handle,&e))throw hip::TrainingError(c,e);live=false;owner.retained.clear();}
    void close(){if(!handle)return;LfsHipError e{};if(int c=lfs_hip_application_gsplat_destroy(handle,&e))throw hip::TrainingError(c,e);handle=nullptr;live=false;owner.retained.clear();}
    uint64_t bind(In t){return owner.bind(t,DataType::Float32);}
};
struct Saved final:BackendState {
    std::unique_ptr<Implementation> impl;std::string message;
    ~Saved()override{if(impl)try{impl->close();}catch(const std::exception& e){ /* LFS-CENSUS-OK(empty-catch): Return an explicit raster failure or report failed cleanup while retaining owners. */ std::fprintf(stderr,"[error] HIP Gsplat retained after failed completion: %s\n",e.what());(void)impl.release();}}
};
Saved& state(GsplatSaved& s){if(!s.backend)s.backend=std::make_unique<Saved>();auto* p=dynamic_cast<Saved*>(s.backend.get());require(p,"HIP Gsplat received a foreign saved state");return *p;}
State create(){return std::make_unique<Saved>();}
Tensor distortion(In input,size_t count){if(!input.is_valid()||!input.numel())return {};require(input.dtype()==DataType::Float32&&input.ndim()==1&&input.numel()<=count,"Invalid HIP Gsplat distortion coefficients");auto output=Tensor::zeros({count},Device::GPU);output.slice(0,0,input.numel()).copy_from(input);return output;}
RasterResult forward(GsplatSaved& saved,const SplatInputs& splats,In view,In radial,In tangential,In bg,In bg_image,const GsplatParams& p,const RenderOutputs& outputs) {
    auto& box=state(saved);
    const core::TensorExecutionTarget::Scope execution(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    try {
        require(splats.means.ndim()==2&&splats.means.shape()[1]==3&&splats.means.shape()[0]>0,"HIP Gsplat requires a nonempty model");const auto n=splats.means.shape()[0];
        shape(splats.raw_scales,n*3);shape(splats.raw_rotations,n*4);shape(splats.raw_opacities,n);shape(splats.sh0,n*3);shape(view,16);
        require(p.sh.active_bases==1||p.sh.active_bases==4||p.sh.active_bases==9||p.sh.active_bases==16,"Invalid HIP Gsplat active SH bases");require(p.sh.layout_bases==1||p.sh.layout_bases==4||p.sh.layout_bases==9||p.sh.layout_bases==16,"Invalid HIP Gsplat SH layout");require(p.sh.active_bases<=p.sh.layout_bases,"HIP Gsplat active SH exceeds storage layout");
        const int mode=int(p.render_mode);require(mode>=0&&mode<=5&&std::isfinite(p.scaling_modifier)&&p.scaling_modifier>0,"Invalid HIP Gsplat render policy");
        const int w=p.tile_w>0?p.tile_w:p.full_image.w,h=p.tile_h>0?p.tile_h:p.full_image.h,channels=(mode==1||mode==2)?1:(mode==3||mode==4)?4:mode==5?8:3;
        require(w>0&&h>0&&uint64_t(w)*h<=INT_MAX/8&&std::isfinite(p.intrinsics.fx)&&p.intrinsics.fx>0&&std::isfinite(p.intrinsics.fy)&&p.intrinsics.fy>0,"Invalid HIP Gsplat viewport/intrinsics");
        if(!box.impl)box.impl=std::make_unique<Implementation>();auto& s=*box.impl;s.release();s.owner.retained.clear();
        if(s.width!=w||s.height!=h||s.channels!=channels||!s.packed.is_valid()){s.packed=Tensor::empty_exact({size_t(channels),size_t(h),size_t(w)});s.alpha=Tensor::empty_exact({1,size_t(h),size_t(w)});s.rays={};}
        s.count=narrow(n);s.width=w;s.height=h;s.channels=channels;s.params=p;
        Tensor rest=splats.shN;const auto rest_count=core::sh_swizzled_float_count(n,p.sh.layout_bases-1);
        if(p.sh.active_bases>1) {
            require(rest.is_valid(),"HIP Gsplat needs active SH-rest storage");
            if(p.sh.storage==ShStorage::Q16){auto decoded=Tensor::empty_exact({rest_count});training_ops(core::GpuBackend::HIP).sh->decode_q16(rest,splats.sh_value_bounds,decoded,n,p.sh.layout_bases-1);rest=std::move(decoded);}
            else if(p.sh.storage==ShStorage::IeeeFloat16){require(rest.dtype()==DataType::Float16,"HIP Gsplat half SH requires Float16 storage");rest=rest.to(DataType::Float32);}
            require(rest.dtype()==DataType::Float32&&rest.numel()>=rest_count,"HIP Gsplat SH-rest storage is too small");
        } else rest={};
        auto scales=p.scaling_modifier==1?splats.raw_scales:Tensor(splats.raw_scales+std::log(p.scaling_modifier));
        const bool equirect=p.camera_model==core::CameraModelType::EQUIRECTANGULAR;
        require(p.full_image.w>0&&p.full_image.h>0,"HIP Gsplat requires positive full-image dimensions");
        std::vector<float> k{equirect?float(p.full_image.w):p.intrinsics.fx,0,equirect?float(p.tile_x):p.intrinsics.cx-p.tile_x,0,equirect?float(p.full_image.h):p.intrinsics.fy,equirect?float(p.tile_y):p.intrinsics.cy-p.tile_y,0,0,1};auto K=Tensor::from_vector(k,{3,3},Device::GPU);
        Tensor r,t,prism;const int camera=int(p.camera_model);
        if(camera==0){require(!radial.is_valid()||radial.numel()<=3||radial.numel()==6,"Pinhole radial coefficients must have at most three or exactly six entries");if(tangential.is_valid()&&tangential.numel())shape(tangential,2);r=distortion(radial,6);t=distortion(tangential,2);}
        else if(camera==2){r=distortion(radial,4);}
        else if(camera==4){if(radial.is_valid()&&radial.numel())shape(radial,4);if(tangential.is_valid()&&tangential.numel())shape(tangential,4);r=distortion(radial,4);prism=distortion(tangential,4);}
        Tensor background,background_image;
        if(channels!=1&&bg.is_valid()&&bg.numel()){shape(bg,3);background=Tensor::zeros({size_t(channels)},Device::GPU);background.slice(0,0,3).copy_from(bg.flatten());}
        if(channels!=1&&bg_image.is_valid()&&bg_image.numel()){shape(bg_image,size_t(w)*h*3);background_image=Tensor::zeros({size_t(channels),size_t(h),size_t(w)},Device::GPU);background_image.slice(0,0,3).copy_from(bg_image.reshape(core::TensorShape{3,size_t(h),size_t(w)}));}
        const bool rays_out=channels==8||p.camera_rays;
        if(rays_out&&!s.rays.is_valid())s.rays=Tensor::empty_exact({size_t(h),size_t(w),3});
        LfsHipGsplatForward f{};f.struct_bytes=sizeof(f);f.count=s.count;f.width=w;f.height=h;f.channels=channels;f.camera_model=camera;f.active_degree=int(std::sqrt(p.sh.active_bases))-1;f.layout_degree=int(std::sqrt(p.sh.layout_bases))-1;
        f.means=s.bind(splats.means);f.quats=s.bind(splats.raw_rotations);f.scales=s.bind(scales);f.opacities=s.bind(splats.raw_opacities);f.sh0=s.bind(splats.sh0);f.shN=s.bind(rest);f.view=s.bind(view);f.K=s.bind(K);f.radial=s.bind(r);f.tangential=s.bind(t);f.thin_prism=s.bind(prism);f.background=s.bind(background);f.image_background=s.bind(background_image);f.image=s.bind(s.packed);f.alpha=s.bind(s.alpha);f.rays=rays_out?s.bind(s.rays):0;
        LfsHipError error{};const int code=s.lfs_hip_application_gsplat_forward(s.handle,&f,&error);
        if(code){box.message=error.message;return {code==LFS_HIP_TRAIN_RESOURCE_EXHAUSTED?RasterResult::Code::ResourceExhausted:RasterResult::Code::Failed,false,box.message};}s.live=true;
        outputs.image={};outputs.depth={};outputs.normal={};outputs.alpha=s.alpha;
        if(channels!=1)outputs.image=s.packed.slice(0,0,3);
        if(channels!=3){auto d=s.packed.slice(0,channels==1?0:3,channels==1?1:4);outputs.depth=(mode==2||mode==4)?Tensor(d.div(s.alpha.clamp_min(1e-10f))):d;}
        if(channels==8)outputs.normal=s.packed.slice(0,4,7);saved.camera_rays=rays_out?s.rays:Tensor{};saved.equirectangular=camera==3;
        uint64_t bytes=0,intersections=0;if(int c=s.lfs_hip_application_gsplat_info(s.handle,&bytes,&intersections,&error))throw hip::TrainingError(c,error);return {RasterResult::Code::Success,intersections>0,{}};
    }catch(const hip::TrainingError& e){box.message=e.what();return {e.status==LFS_HIP_TRAIN_RESOURCE_EXHAUSTED?RasterResult::Code::ResourceExhausted:RasterResult::Code::Failed,false,box.message};}
    catch(const std::exception& e){ /* LFS-CENSUS-OK(empty-catch): Return an explicit raster failure or report failed cleanup while retaining owners. */ box.message=e.what();return {RasterResult::Code::Failed,false,box.message};}
}
void backward(GsplatSaved& saved,In image,In alpha,const GsplatGradients& g,Out density,In error_map,In edge_map,Out edge_scores,Out shares) {
    const core::TensorExecutionTarget::Scope execution(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    auto& box=state(saved);require(box.impl&&box.impl->live&&g.get,"HIP Gsplat backward has no live frame or gradient resolver");auto& s=*box.impl;const size_t pixels=size_t(s.width)*s.height;
    auto packed=Tensor::zeros(s.packed.shape(),Device::GPU),ga=Tensor::zeros(s.alpha.shape(),Device::GPU);
    if(image.is_valid()&&image.numel()){shape(image,pixels*3);require(s.channels!=1,"Depth-only Gsplat cannot receive RGB gradients");packed.slice(0,0,3).copy_from(image.reshape(core::TensorShape{3,size_t(s.height),size_t(s.width)}));}
    if(alpha.is_valid()&&alpha.numel()){shape(alpha,pixels);ga.copy_from(alpha.reshape(ga.shape()));}
    if(g.depth.is_valid()&&g.depth.numel()){shape(g.depth,pixels);require(s.channels!=3,"RGB Gsplat cannot receive depth gradients");const int offset=s.channels==1?0:3;packed.slice(0,offset,offset+1).copy_from(g.depth.reshape(core::TensorShape{1,size_t(s.height),size_t(s.width)}));}
    if(g.normal.is_valid()&&g.normal.numel()){shape(g.normal,pixels*3);require(s.channels==8,"Gspat normals require RGB_D_N");packed.slice(0,4,7).copy_from(g.normal.reshape(core::TensorShape{3,size_t(s.height),size_t(s.width)}));}
    auto bind_optional=[&](In t,size_t count){if(t.is_valid()&&t.numel()){require(t.numel()>=count,"HIP Gsplat optional buffer too small");return s.bind(t);}return uint64_t{0};};
    LfsHipGsplatBackward b{};b.struct_bytes=sizeof(b);b.flatten_weight=g.flatten_weight;b.expected_depth=s.params.render_mode==GsplatRenderMode::ED||s.params.render_mode==GsplatRenderMode::RGB_ED;b.image=s.bind(packed);b.alpha=s.bind(ga);
    b.densification=bind_optional(density,size_t(s.count)*2);b.error_map=bind_optional(error_map,pixels);b.edge_map=bind_optional(edge_map,pixels);b.edge_scores=bind_optional(edge_scores,s.count);b.screen_share=bind_optional(shares,s.count);
    auto gradient=[&](AdamSlot slot,size_t count){auto& t=g.get(g.owner,slot);require(t.numel()>=count,"HIP Gsplat gradient buffer too small");return s.bind(t);};
    b.grad_means=gradient(AdamSlot::Means,size_t(s.count)*3);b.grad_quats=gradient(AdamSlot::Rotation,size_t(s.count)*4);b.grad_scales=gradient(AdamSlot::Scaling,size_t(s.count)*3);b.grad_opacity=gradient(AdamSlot::Opacity,s.count);
    if(s.channels!=1){b.grad_sh0=gradient(AdamSlot::Sh0,size_t(s.count)*3);if(s.params.sh.active_bases>1)b.grad_shN=gradient(AdamSlot::ShN,core::sh_swizzled_float_count(s.count,s.params.sh.layout_bases-1));}
    LfsHipError e{};if(int c=s.lfs_hip_application_gsplat_backward(s.handle,&b,&e))throw hip::TrainingError(c,e);s.live=false;
}
void release(GsplatSaved& saved)noexcept{auto* p=dynamic_cast<Saved*>(saved.backend.get());if(!p||!p->impl)return;try{p->impl->release();}catch(const std::exception& e){ /* LFS-CENSUS-OK(empty-catch): Return an explicit raster failure or report failed cleanup while retaining owners. */ std::fprintf(stderr,"[error] HIP Gsplat frame retained: %s\n",e.what());}}
bool release_caches(GsplatSaved& saved)noexcept{auto* p=dynamic_cast<Saved*>(saved.backend.get());if(!p||!p->impl)return true;if(p->impl->live)return false;try{p->impl->close();p->impl.reset();saved.camera_rays={};return true;}catch(const std::exception& e){ /* LFS-CENSUS-OK(empty-catch): Return an explicit raster failure or report failed cleanup while retaining owners. */ std::fprintf(stderr,"[error] HIP Gsplat caches retained: %s\n",e.what());return false;}}
void record(const GsplatSaved& saved,In,In,In,In,In){auto* p=dynamic_cast<const Saved*>(saved.backend.get());if(!p||!p->impl)return;uint64_t bytes=0,intersections=0;LfsHipError e{};if(int c=p->impl->lfs_hip_application_gsplat_info(p->impl->handle,&bytes,&intersections,&e))throw hip::TrainingError(c,e);diagnostics::VramProfiler::instance().recordCurrentBytes("rasterizer.gsplat","hip.workspace",bytes,diagnostics::VramAllocationMethod::Arena);}
}
const gpu_ops::GsplatRasterOps& hip_gsplat_ops(){static const gpu_ops::GsplatRasterOps ops{create,forward,backward,release,record,release_caches,
    [](const GsplatSaved& saved,In background,In background_image,Out scores) {
        const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
        const auto* box=dynamic_cast<const Saved*>(saved.backend.get());
        require(box && box->impl && box->impl->live && box->impl->channels>=3,"HIP Gsplat POPSpa needs a live RGB forward");
        auto& s=*box->impl;
        require(scores.is_valid() && scores.dtype()==DataType::UInt8 && scores.ndim()==2 &&
                scores.shape()[0]==size_t(s.count) && scores.shape()[1]==8,"HIP Gsplat POPSpa scores require UInt8 [N,8]");
        require(!background.is_valid() || !background.numel() || background.numel()==3,"HIP Gsplat POPSpa background must have three channels");
        require(!background_image.is_valid() || !background_image.numel() || background_image.numel()==size_t(s.width)*s.height*3,
                "HIP Gsplat POPSpa background image extent mismatch");
        const auto bg=s.bind(background),image=s.bind(background_image),output=s.owner.bind(scores,DataType::UInt8);
        LfsHipError error{};
        if(int code=s.lfs_hip_application_gsplat_pop_scores(s.handle,bg,image,output,&error))throw hip::TrainingError(code,error);
    }};return ops;}
}
