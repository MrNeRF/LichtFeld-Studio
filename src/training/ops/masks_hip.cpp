/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/masks_hip.hpp"
#include "training/hip/application_color_tensor.hpp"
#include "training/hip/application_masks_api.h"
#include "mesh_mask_camera.hpp"
#include "core/tensor_execution.hpp"
#include <cstring>
namespace lfs::training {
namespace {
using namespace gpu_ops;using hip::color_require;using hip::color_shape;using hip::color_count;
void mask(hip::ColorCall& c,In m,In roi,Out output,unsigned mode) {
    color_require(m.ndim()==2&&m.numel()>0&&m.numel()<=INT_MAX-255,"HIP masks require a nonempty HW image");
    color_require(m.dtype()==core::DataType::Float32||m.dtype()==core::DataType::UInt8||m.dtype()==core::DataType::Bool,"HIP mask dtype must be Float32, UInt8 or Bool");color_shape(output,m.numel());if(roi.is_valid())color_shape(roi,m.numel());
    c.p.dimensions[0]=color_count(m.shape()[0]);c.p.dimensions[1]=color_count(m.shape()[1]);c.p.flags[0]=m.dtype()!=core::DataType::Float32;c.p.flags[1]=mode;
    c.p.bindings[1]=c.owner.bind(m,m.dtype());c.bind(2,roi);c.bind(3,output);
}
void photo(In m,In roi,Out output,MaskPhotoMode mode){hip::ColorCall c(0);mask(c,m,roi,output,unsigned(mode));c.submit("lfs_hip_application_masks");}
void penalty(unsigned operation,In alpha,In m,In roi,Out grad,Out temp,Out loss,unsigned mode,float power,float scale) {
    hip::ColorCall c(operation);mask(c,m,roi,grad,mode);color_shape(alpha,m.numel());color_require(temp.numel()>=std::min((m.numel()+255)/256,size_t{1024})&&loss.numel()>=1,"HIP mask reduction buffers are too small");
    c.bind(0,alpha);c.bind(4,temp);c.bind(5,loss);c.p.values[0]=power;c.p.values[1]=scale;c.submit("lfs_hip_application_masks");
}
void opacity(In a,In m,In roi,Out grad,Out temp,Out loss,MaskOpacityMode mode,float power,float scale){penalty(1,a,m,roi,grad,temp,loss,unsigned(mode),power,scale);}
void alpha(In a,In m,In roi,Out grad,Out temp,Out loss,float weight){penalty(2,a,m,roi,grad,temp,loss,0,0,weight);}
void mesh_submit(hip::ApplicationCall& owner,LfsHipMeshCall& p) {
    p.struct_bytes=sizeof(p);p.device=owner.queue.device;p.queue=owner.queue.native_queue;LfsHipError error{};
    const auto fn=owner.module->symbol<decltype(&lfs_hip_application_mesh)>("lfs_hip_application_mesh");if(int code=fn(&p,&error))throw hip::TrainingError(code,error);
}
void camera(LfsHipMeshCall& p,const MeshMaskCamera& camera,In samples,const core::UndistortParams* distortion) {
    const auto packed=pack_mesh_mask_camera(camera,distortion?&samples:nullptr,distortion);
    static_assert(sizeof(p.camera)==sizeof(packed));static_assert(offsetof(LfsHipMeshCamera,distortion)==offsetof(MeshMaskCameraBlock,distortion));std::memcpy(&p.camera,&packed,sizeof(packed));
}
Tensor mesh(In vertices,In indices,const MeshMaskCamera& k,In samples,const core::UndistortParams* distortion,float z_near) {
    const core::TensorExecutionTarget::Scope execution(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    validate_mesh_coverage(vertices,indices,k,distortion?&samples:nullptr,distortion,z_near);
    color_require(vertices.shape()[0]<=INT_MAX/3&&indices.shape()[0]<=INT_MAX/3,"HIP mesh coverage exceeds indexing range");
    auto output=Tensor::zeros({size_t(k.height),size_t(k.width)},core::Device::GPU,core::DataType::UInt8);if(!indices.numel()||!vertices.numel())return output;
    auto v=vertices.contiguous(),i=indices.contiguous(),s=distortion?samples.contiguous():Tensor{};
    auto faces=Tensor::empty({indices.shape()[0]},core::Device::GPU,core::DataType::Int32),counts=Tensor::zeros({2},core::Device::GPU,core::DataType::Int32);
    hip::ApplicationCall owner;LfsHipMeshCall p{};p.vertex_count=color_count(vertices.shape()[0]);p.face_count=color_count(indices.shape()[0]);p.z_near=z_near;camera(p,k,s,distortion);
    p.vertices=owner.bind(v,core::DataType::Float32);p.indices=owner.bind(i,core::DataType::Int32);p.samples=owner.bind(s,core::DataType::Float32);p.large_faces=owner.bind(faces,core::DataType::Int32);p.counters=owner.bind(counts,core::DataType::Int32);p.mask=owner.bind(output,core::DataType::UInt8);mesh_submit(owner,p);p.operation=1;mesh_submit(owner,p);return output;
}
Tensor points(In means,const MeshMaskCamera& k,int radius,const core::UndistortParams* distortion) {
    const core::TensorExecutionTarget::Scope execution(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    color_require(means.ndim()==2&&means.shape()[1]==3&&means.shape()[0]<=INT_MAX/3&&k.width>0&&k.height>0&&uint64_t(k.width)*k.height<=INT_MAX&&radius>=0&&radius<=32767,"Invalid HIP point coverage geometry");
    auto output=Tensor::zeros({size_t(k.height),size_t(k.width)},core::Device::GPU,core::DataType::UInt8);if(!means.numel())return output;
    const auto packed=pack_mesh_mask_camera(k,nullptr,distortion);auto m=means.contiguous();hip::ApplicationCall owner;LfsHipMeshCall p{};p.operation=2;p.vertex_count=color_count(means.shape()[0]);p.radius=radius;
    std::memcpy(&p.camera,&packed,sizeof(packed));p.vertices=owner.bind(m,core::DataType::Float32);p.mask=owner.bind(output,core::DataType::UInt8);mesh_submit(owner,p);return output;
}
}
const gpu_ops::MaskOps& hip_masks_ops(){static const gpu_ops::MaskOps ops{points,photo,opacity,alpha,mesh};return ops;}
}
