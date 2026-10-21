/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lfs/training/ops/fast_hip.hpp"
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "training/hip/application_fast_api.h"
#include "training/hip/application_module.hpp"
#include "training/hip/session.hpp"
#include <core/tensor_execution.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#ifndef LFS_HIP_TRAINING_DEFAULT_MODULE
#define LFS_HIP_TRAINING_DEFAULT_MODULE ""
#endif
namespace lfs::training {
namespace {
using namespace gpu_ops;
using core::DataType;
using core::Device;
void require(bool condition, const char* message) { if (!condition) throw std::invalid_argument(message); }
int narrow(size_t count) {
    require(count <= size_t(std::numeric_limits<int>::max()), "HIP FastGS count exceeds int32");
    return static_cast<int>(count);
}
core::internal::HipBackendOps& native_ops() {
    return static_cast<core::internal::HipBackendOps&>(core::internal::backend_ops(core::GpuBackend::HIP));
}
struct Implementation {
    std::shared_ptr<hip::ApplicationModule> module;
    core::internal::HipQueueLease queue;
    LfsHipApplicationFast* handle = nullptr;
#define ENTRY(name) decltype(&::name) name = nullptr
    ENTRY(lfs_hip_application_fast_create);
    ENTRY(lfs_hip_application_fast_forward);
    ENTRY(lfs_hip_application_fast_backward);
    ENTRY(lfs_hip_application_fast_release);
    ENTRY(lfs_hip_application_fast_info);
    ENTRY(lfs_hip_application_fast_destroy);
#undef ENTRY
    std::vector<Tensor> retained, backward_inputs;
    Tensor image, alpha, depth, normal;
    int width = 0, height = 0, count = 0, sh_gradient_attributes = 0;
    bool live = false;
    Implementation() {
        const char* path = std::getenv("LFS_HIP_TRAINING_MODULE");
        module = hip::cached_application_module(path ? path : LFS_HIP_TRAINING_DEFAULT_MODULE);
        queue = native_ops().execution_lease({});
        native_ops().retain_application_module(module);
#define LOAD(name) name = module->symbol<decltype(name)>(#name)
        LOAD(lfs_hip_application_fast_create);
        LOAD(lfs_hip_application_fast_forward);
        LOAD(lfs_hip_application_fast_backward);
        LOAD(lfs_hip_application_fast_release);
        LOAD(lfs_hip_application_fast_info);
        LOAD(lfs_hip_application_fast_destroy);
#undef LOAD
        LfsHipError error{};
        const int code = lfs_hip_application_fast_create(queue.device, queue.native_queue, 2ull*1024*1024*1024, &handle, &error);
        if (code) throw hip::TrainingError(code, error);
    }
    void close() {
        if (!handle) return;
        LfsHipError error{};
        const int code = lfs_hip_application_fast_destroy(handle, &error);
        if (code) throw hip::TrainingError(code, error);
        handle = nullptr;
        live = false;
    }
    uint64_t bind(In tensor, std::vector<Tensor>& owners, DataType dtype = DataType::Float32, bool raw = false) {
        if (!tensor.is_valid() || !tensor.numel()) return 0;
        require(core::gpu_backend_of(tensor) == core::GpuBackend::HIP && tensor.is_contiguous(), "HIP FastGS bindings require contiguous native HIP tensors");
        require(raw || tensor.dtype() == dtype, "HIP FastGS binding dtype mismatch");
        // Match the native backends' mutable pointer boundary: settle pending
        // expressions and preserve lazy readers before a pointer escapes.
        (void)const_cast<Tensor&>(tensor).data_ptr();
        owners.push_back(tensor);
        core::pin_operands({&owners.back()});
        return native_ops().storage_address(core::internal::storage_ref(owners.back()), owners.back().bytes());
    }
};
struct FastState final : BackendState {
    std::unique_ptr<Implementation> implementation;
    std::string message;
    ~FastState() override {
        if (!implementation) return;
        try { implementation->close(); }
        catch (const std::exception& error) {
            std::fprintf(stderr, "[error] HIP application raster retained after incomplete teardown: %s\n", error.what());
            (void)implementation.release(); // retain module, queue, model and outputs
        }
    }
};
FastState& state(FastSaved& saved) {
    if (!saved.backend) saved.backend = std::make_unique<FastState>();
    auto* result = dynamic_cast<FastState*>(saved.backend.get());
    require(result != nullptr, "HIP FastGS received another backend's saved frame");
    return *result;
}
State create() { return std::make_unique<FastState>(); }
RasterResult failure(FastState& state, RasterResult::Code code, const char* message) {
    state.message = message;
    return {code, false, state.message};
}
RasterResult forward(FastSaved& saved, const SplatInputs& splats, In view, In camera, In bg_color, In bg_image,
                     const FastParams& params, const RenderOutputs& output, Out screen_share) {
    auto& owner = state(saved);
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    try {
        const auto count = splats.means.is_valid() && splats.means.ndim() == 2 ? splats.means.shape()[0] : 0;
        if (!count) return failure(owner, RasterResult::Code::Failed, "n_primitives is 0 - model has no gaussians");
        require(splats.means.numel()==count*3 && splats.raw_scales.numel()==count*3 && splats.raw_rotations.numel()==count*4 &&
                splats.raw_opacities.numel()==count && splats.sh0.numel()==count*3 && view.numel()==16 && camera.numel()==3,
                "HIP FastGS model/camera dimensions mismatch");
        const int width = params.tile_w > 0 ? params.tile_w : params.full_image.w;
        const int height = params.tile_h > 0 ? params.tile_h : params.full_image.h;
        require(width>0 && height>0 && size_t(width)*height <= size_t(std::numeric_limits<int>::max())/3, "Invalid HIP FastGS viewport");
        if (!owner.implementation) owner.implementation = std::make_unique<Implementation>();
        auto& s = *owner.implementation;
        s.live = false;
        s.retained.clear(); s.backward_inputs.clear();
        if (!s.image.is_valid() || width!=s.width || height!=s.height) {
            s.image = Tensor::empty_exact({3,size_t(height),size_t(width)});
            s.alpha = Tensor::empty_exact({1,size_t(height),size_t(width)});
            s.depth = {}; s.normal = {};
        }
        if (params.render_depth && !s.depth.is_valid()) s.depth = Tensor::empty_exact({1,size_t(height),size_t(width)});
        if (params.render_normal && !s.normal.is_valid()) s.normal = Tensor::empty_exact({3,size_t(height),size_t(width)});
        s.width=width; s.height=height; s.count=narrow(count);
        s.sh_gradient_attributes=((params.sh.layout_bases-1)*3+3)/4*4;
        auto bind = [&](In t) { return s.bind(t,s.retained); };
        LfsHipFastForward f{};
        f.struct_bytes=sizeof(f); f.count=s.count; f.width=width; f.height=height;
        f.means=bind(splats.means); f.scales=bind(splats.raw_scales); f.rotations=bind(splats.raw_rotations); f.opacities=bind(splats.raw_opacities); f.sh0=bind(splats.sh0);
        f.shN=s.bind(splats.shN,s.retained,DataType::Float32,true); f.sh_bounds=bind(splats.sh_value_bounds);
        f.sh_bits=params.sh.storage==ShStorage::Float32 ? 0 : 16;
        f.sh_cells=params.sh.storage==ShStorage::Q16 ? (params.sh.layout_bases-1)*3 : 0;
        f.active_bases=params.sh.active_bases; f.layout_bases=params.sh.layout_bases;
        f.view=bind(view); f.camera=bind(camera);
        require(!bg_color.is_valid() || !bg_color.numel() || bg_color.numel()==3, "HIP FastGS background color needs three values");
        require(!bg_image.is_valid() || !bg_image.numel() || bg_image.numel()==size_t(width)*height*3, "HIP FastGS background image shape mismatch");
        f.background_color=bind(bg_color); f.background_image=bind(bg_image);
        f.image=bind(s.image); f.alpha=bind(s.alpha);
        if (params.render_depth) f.depth=bind(s.depth);
        if (params.render_normal) f.normal=bind(s.normal);
        require(!screen_share.is_valid() || !screen_share.numel() || screen_share.numel()>=count, "HIP FastGS screen-share storage too small");
        f.screen_share=bind(screen_share);
        f.fx=params.intrinsics.fx; f.fy=params.intrinsics.fy;
        f.cx=params.intrinsics.cx-params.tile_x; f.cy=params.intrinsics.cy-params.tile_y;
        f.dilation=params.dilation_scale; f.mip_filter=params.mip_filter;
        LfsHipError error{};
        const int code=s.lfs_hip_application_fast_forward(s.handle,&f,&error);
        if (code) return failure(owner,code==LFS_HIP_TRAIN_RESOURCE_EXHAUSTED ? RasterResult::Code::ResourceExhausted : code==LFS_HIP_FAST_INSTANCE_OVERFLOW ? RasterResult::Code::InstanceOverflow : RasterResult::Code::Failed,error.message);
        LfsHipFastInfo info{}; info.struct_bytes=sizeof(info);
        const int info_code=s.lfs_hip_application_fast_info(s.handle,&info,&error);
        if (info_code) throw hip::TrainingError(info_code,error);
        output.image=s.image; output.alpha=s.alpha;
        output.depth=params.render_depth ? s.depth : Tensor{};
        output.normal=params.render_normal ? s.normal : Tensor{};
        s.live=true;
        return {RasterResult::Code::Success,info.instances>0,{}};
    } catch (const hip::TrainingError& error) {
        return failure(owner,error.status==LFS_HIP_TRAIN_RESOURCE_EXHAUSTED ? RasterResult::Code::ResourceExhausted : RasterResult::Code::Failed,error.what());
    } catch (const std::exception& error) { return failure(owner,RasterResult::Code::Failed,error.what()); }
}
void backward(FastSaved& saved, const RenderGradients& gradients, Out densification, In error_map, In edge_map, Out edge_scores,
              const BackwardAdam& adam, DensificationType densification_type) {
    auto& owner=state(saved);
    require(owner.implementation && owner.implementation->live,"HIP FastGS backward needs a live application forward");
    auto& s=*owner.implementation;
    const core::TensorExecutionTarget::Scope scope(core::TensorExecutionTarget::default_queue(core::GpuBackend::HIP));
    s.backward_inputs.clear();
    auto bind=[&](In t) { return s.bind(t,s.backward_inputs); };
    auto mask=[&](In t) { return s.bind(t,s.backward_inputs,DataType::Bool); };
    const size_t area=size_t(s.width)*s.height;
    require(gradients.image.is_valid() && gradients.image.numel()==area*3,"HIP FastGS image gradient shape mismatch");
    for (const auto* t : {&gradients.alpha,&gradients.depth,&error_map,&edge_map})
        require(!t->is_valid() || !t->numel() || t->numel()==area,"HIP FastGS scalar image gradient/map shape mismatch");
    require(!gradients.normal.is_valid() || !gradients.normal.numel() || gradients.normal.numel()==area*3,"HIP FastGS normal gradient shape mismatch");
    require(!densification.is_valid() || !densification.numel() || densification.numel()>=size_t(s.count)*2,"HIP FastGS densification storage too small");
    require(!edge_scores.is_valid() || !edge_scores.numel() || edge_scores.numel()>=size_t(s.count),"HIP FastGS edge-score storage too small");
    const bool has_impact_sum = adam.impact_sum.is_valid() && adam.impact_sum.numel();
    const bool has_impact_peak = adam.impact_peak.is_valid() && adam.impact_peak.numel();
    require(has_impact_sum == has_impact_peak, "HIP FastGS impact sum and peak must be supplied together");
    require(!has_impact_sum || (adam.impact_sum.numel() == size_t(s.count) && adam.impact_peak.numel() == size_t(s.count)),
            "HIP FastGS impact output shape mismatch");
    LfsHipFastBackward b{}; b.struct_bytes=sizeof(b);
    b.grad_image=bind(gradients.image); b.grad_alpha=bind(gradients.alpha); b.grad_depth=bind(gradients.depth); b.grad_normal=bind(gradients.normal);
    b.densification=bind(densification); b.error_map=bind(error_map); b.edge_map=bind(edge_map); b.edge_scores=bind(edge_scores);
    b.impact_sum=bind(adam.impact_sum); b.impact_peak=bind(adam.impact_peak);
    b.densification_type=static_cast<int>(densification_type);
    b.beta1=adam.beta1; b.beta2=adam.beta2; b.eps=adam.eps;
    b.scale_reg_weight=adam.scale_reg_weight; b.scale_reg_log=adam.scale_reg_log; b.scale_reg_normalizer=adam.scale_reg_normalizer;
    b.opacity_reg_weight=adam.opacity_reg_weight; b.flatten_reg_weight=adam.flatten_reg_weight;
    b.erank_reg_weight=adam.erank_reg_weight; b.dc_reg_weight=adam.dc_reg_weight; b.sh_reg_weight=adam.sh_rest_reg_weight;
    b.scale_loss=bind(adam.scale_reg_loss); b.opacity_loss=bind(adam.opacity_reg_loss); b.erank_loss=bind(adam.erank_reg_loss);
    b.dc_loss=bind(adam.dc_reg_loss); b.sh_loss=bind(adam.sh_rest_reg_loss); b.rendered_count=bind(adam.rendered_count);
    b.per_splat_mean_step=adam.per_splat_mean_step; b.median_extent=adam.mean_step_median_extent;
    b.mean_far=mask(adam.mean_step_far_mask); b.mean_far_count=adam.mean_step_far_mask.is_valid() ? narrow(std::min(adam.mean_step_far_mask.numel(),size_t(s.count))) : 0;
    for (size_t i=0;i<6;++i) {
        const auto& source=adam.groups[i]; auto& g=b.groups[i];
        const bool capture = source.gradient.is_valid() && source.gradient.numel();
        require(!capture || (source.primitives == s.count && source.elements > 0 &&
                            source.gradient.numel() == size_t(source.elements)), "HIP FastGS raw gradient capture shape mismatch");
        if(capture && i==5)
            require(source.attributes==s.sh_gradient_attributes && size_t(source.elements)==size_t(s.count)*size_t(s.sh_gradient_attributes),
                    "HIP FastGS SH raw gradient layout mismatch");
        g.gradient_out=bind(source.gradient);
        if ((source.enabled || capture) && i<5) {
            constexpr size_t attributes[]{3,3,4,1,3};
            require(source.primitives==s.count && source.attributes==int(attributes[i]) && source.elements>=0 && size_t(source.elements)==size_t(s.count)*attributes[i] &&
                    source.parameter.is_valid() && source.parameter.dtype()==DataType::Float32 && source.parameter.numel()>=size_t(source.elements),
                    "HIP FastGS Adam parameter dimensions mismatch");
            require(!source.enabled || (source.packed_moments.is_valid() && source.packed_moments.bytes()>=size_t(source.elements)*4 &&
                    source.joint_bounds.is_valid() && source.joint_bounds.numel()>=size_t((s.count+255)/256)*4),
                    "HIP FastGS Adam moment/bound storage too small");
        }

        g.parameter=s.bind(source.parameter,s.backward_inputs,DataType::Float32,true);
        g.packed=s.bind(source.packed_moments,s.backward_inputs,DataType::UInt8);
        g.bounds=bind(source.joint_bounds); g.value_bounds=bind(source.sh_value_bounds);
        g.frozen=mask(source.frozen_mask); g.crop=mask(source.crop_damping_mask);
        if (source.screen_share_limit>0.f && source.screen_share_limit<1.f) {
            g.screen_share=bind(source.screen_share);
            g.screen_count=source.screen_share.is_valid() ? narrow(source.screen_share.numel()) : 0;
            g.screen_limit=source.screen_share_limit; g.screen_penalty=source.screen_share_penalty;
        }
        g.joint_bits=source.joint_bits; g.value_bits=source.value_bits; g.value_cells=source.value_cells;
        g.primitives=source.primitives; g.elements=source.elements; g.attributes=source.attributes; g.enabled=source.enabled;
        g.frozen_count=source.frozen_mask.is_valid() ? narrow(source.frozen_mask.numel()) : 0;
        g.crop_count=source.crop_damping_mask.is_valid() ? narrow(source.crop_damping_mask.numel()) : 0;
        g.step_size=source.step_size; g.bc2_sqrt_rcp=source.bc2_sqrt_rcp;
        g.frozen_lr=source.frozen_lr_scale; g.crop_lr=source.cropbox_lr_scale;
    }
    LfsHipError error{};
    const int code=s.lfs_hip_application_fast_backward(s.handle,&b,&error);
    s.live=false;
    if (code) throw hip::TrainingError(code,error);
}
void release(FastSaved& saved) noexcept {
    if (!saved.backend) return;
    auto* owner=dynamic_cast<FastState*>(saved.backend.get());
    if (!owner || !owner->implementation) return;
    auto& s=*owner->implementation;
    LfsHipError error{};
    if (s.lfs_hip_application_fast_release(s.handle,&error))
        std::fprintf(stderr,"[error] HIP application FastGS frame release failed: %s\n",error.message);
    s.live=false;
    s.retained.clear(); s.backward_inputs.clear();
}
void warmup() {}
void record_vram(const FastSaved& saved, In, In, bool, size_t) {
    if (!saved.backend) return;
    const auto* owner=dynamic_cast<const FastState*>(saved.backend.get());
    if (!owner || !owner->implementation) return;
    auto& s=*owner->implementation;
    LfsHipFastInfo info{}; info.struct_bytes=sizeof(info); LfsHipError error{};
    const int code=s.lfs_hip_application_fast_info(s.handle,&info,&error);
    if (code) throw hip::TrainingError(code,error);
    diagnostics::VramProfiler::instance().recordCurrentBytes("rasterizer.fastgs","hip.workspace",info.live_bytes,diagnostics::VramAllocationMethod::Arena);
}
void release_caches(FastSaved& saved) noexcept {
    if (!saved.backend) return;
    auto* owner=dynamic_cast<FastState*>(saved.backend.get());
    if (!owner || !owner->implementation || owner->implementation->live) return;
    try { owner->implementation->close(); owner->implementation.reset(); }
    catch (const std::exception& error) { std::fprintf(stderr,"[error] HIP application FastGS caches retained: %s\n",error.what()); }
}
}
const gpu_ops::FastRasterOps& hip_fast_ops() {
    static const gpu_ops::FastRasterOps table{create,forward,backward,release,warmup,record_vram,release_caches};
    return table;
}
}
