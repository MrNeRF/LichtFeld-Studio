/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
// Included only by the native HIP training translation unit, after its shared
// checked allocator and generated FastGS entry points have been defined.
#include "application_fast_api.h"
#include <exception>
#ifdef LFS_HIP_FAST_RUNTIME
struct LfsHipApplicationFast {
    int device;
    hipStream_t queue;
    Budget budget;
    FastFrame workspace;
    LfsHipFastForward input{};
    bool live = false, poisoned = false, optimizer_submitted = false;
    LfsHipApplicationFast(int device, hipStream_t queue, uint64_t bytes)
        : device(device), queue(queue), budget{bytes}, workspace(budget, queue) {}
};
namespace {
    template<class T = float> T* app_pointer(uint64_t value) {
        return reinterpret_cast<T*>(static_cast<uintptr_t>(value));
    }
    template<class F> int app_fast_boundary(LfsHipApplicationFast* state, LfsHipError* error, F&& operation) {
        return boundary(nullptr, error, [&] {
            require(state, "Missing application HIP FastGS state");
            if (state->poisoned)
                throw Failure(LFS_HIP_TRAIN_POISONED, "Application HIP FastGS state is poisoned");
            DeviceScope selected(state->device);
            try {
                operation();
            } catch (...) {
                const auto failure = std::current_exception();
                const auto completion = hipStreamSynchronize(state->queue);
                if (completion != hipSuccess || state->optimizer_submitted)
                    state->poisoned = true;
                state->live = false;
                if (completion != hipSuccess)
                    hip_check(completion, "failed application HIP FastGS completion");
                if (state->optimizer_submitted)
                    throw Failure(LFS_HIP_TRAIN_POISONED, "HIP application failure after Adam submission; do not retry this model step");
                try { std::rethrow_exception(failure); }
                catch (const std::overflow_error& overflow) {
                    throw Failure(LFS_HIP_FAST_INSTANCE_OVERFLOW, overflow.what());
                }

            }
        });
    }
    fast_lfs::rasterization::FusedAdamParam app_adam_group(const LfsHipFastAdamGroup& source) {
        fast_lfs::rasterization::FusedAdamParam result{};
        result.gradient_out = app_pointer(source.gradient_out);
        result.param = app_pointer(source.parameter);
        result.joint_packed = app_pointer<uint8_t>(source.packed);
        result.joint_bounds = app_pointer(source.bounds);
        result.sh_value_bounds = app_pointer(source.value_bounds);
        result.joint_bits = source.joint_bits;
        result.sh_value_bits = source.value_bits;
        result.sh_value_n_cells = source.value_cells;
        result.n_primitives = source.primitives;
        result.n_elements = source.elements;
        result.n_attributes = source.attributes;
        result.frozen_mask = app_pointer<bool>(source.frozen);
        result.frozen_mask_size = source.frozen_count;
        result.crop_damping_mask = app_pointer<bool>(source.crop);
        result.crop_damping_mask_size = source.crop_count;
        result.screen_share_max = app_pointer(source.screen_share);
        result.screen_share_n = source.screen_count;
        result.frozen_lr_scale = source.frozen_lr;
        result.cropbox_lr_scale = source.crop_lr;
        result.screen_share_limit = source.screen_limit;
        result.screen_share_penalty = source.screen_penalty;
        result.step_size = source.step_size;
        result.bias_correction2_sqrt_rcp = source.bc2_sqrt_rcp;
        result.enabled = source.enabled != 0;
        return result;
    }
}
extern "C" {
int lfs_hip_application_fast_create(int32_t ordinal, uint64_t queue, uint64_t bytes, LfsHipApplicationFast** output, LfsHipError* error) {
    return boundary(nullptr, error, [&] {
        require(output, "Missing application HIP FastGS output");
        *output = nullptr;
        require(queue && queue <= UINTPTR_MAX && bytes, "Invalid application HIP queue or workspace budget");
        int count = 0;
        hip_check(hipGetDeviceCount(&count), "HIP device count");
        require(ordinal >= 0 && ordinal < count, "Invalid application HIP device");
        DeviceScope selected(ordinal);
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, ordinal), "HIP application device properties");
        const std::string_view arch(properties.gcnArchName);
        require(arch.substr(0, arch.find(':')) == LFS_HIP_COMPILED_ARCH, "Application HIP FastGS architecture mismatch");
        auto native_queue = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(queue));
        unsigned flags = 0;
        hip_check(hipStreamGetFlags(native_queue, &flags), "validate borrowed HIP queue");
        *output = new LfsHipApplicationFast(ordinal, native_queue, bytes);
    });
}
int lfs_hip_application_fast_forward(LfsHipApplicationFast* state, const LfsHipFastForward* input, LfsHipError* error) {
    return app_fast_boundary(state, error, [&] {
        require(input && input->struct_bytes == sizeof(*input) && input->count > 0 && input->width > 0 && input->height > 0,
                "Invalid application HIP FastGS forward dimensions");
        require(input->means && input->scales && input->rotations && input->opacities && input->sh0 && input->view && input->camera && input->image && input->alpha,
                "Missing application HIP FastGS forward binding");
        require((input->active_bases == 1 || input->active_bases == 4 || input->active_bases == 9 || input->active_bases == 16) &&
                (input->layout_bases == 1 || input->layout_bases == 4 || input->layout_bases == 9 || input->layout_bases == 16) &&
                input->active_bases <= input->layout_bases && (input->layout_bases == 1 || input->shN), "Invalid application HIP spherical harmonics layout");
        require(input->sh_bits == 0 || input->sh_bits == 16, "Invalid application HIP SH codec");
        require(std::isfinite(input->fx) && input->fx > 0 && std::isfinite(input->fy) && input->fy > 0 &&
                std::isfinite(input->cx) && std::isfinite(input->cy) && std::isfinite(input->dilation) && input->dilation > 0,
                "Invalid application HIP camera intrinsics");
        state->live = false;
        state->input = *input;
        auto& f = state->workspace;
        f.cursor = 0;
        f.phase_candidate = nullptr;
        f.phase_candidate_bytes = 0;
        auto allocate = [&](size_t bytes) { return f.allocate(bytes); };
        f.result = fast_lfs::rasterization::forward(allocate, [](size_t) {}, [&](size_t bytes) { return f.allocate_phase(bytes); },
            [&](const void* source, size_t bytes) {
                auto* output = f.allocate(bytes);
                hip_check(hipMemcpyAsync(output, source, bytes, hipMemcpyDeviceToDevice, state->queue), "retain HIP FastGS phase prefix");
                return output;
            }, [&](size_t bytes) { return f.tiles = f.allocate(bytes); },
            app_pointer<float3>(input->means), app_pointer<float3>(input->scales), app_pointer<float4>(input->rotations),
            app_pointer(input->opacities), app_pointer<float3>(input->sh0), app_pointer<float4>(input->shN),
            app_pointer<float2>(input->sh_bounds), input->sh_cells, input->sh_bits,
            app_pointer<float4>(input->view), app_pointer<float3>(input->camera),
            app_pointer(input->image), app_pointer(input->alpha), app_pointer(input->depth), app_pointer(input->normal),
            app_pointer(input->background_color), app_pointer(input->background_image),
            input->count, input->active_bases, input->layout_bases, input->width, input->height,
            input->fx, input->fy, input->cx, input->cy, 0.01f, 1e10f, input->mip_filter != 0,
            state->queue, app_pointer(input->screen_share), input->dilation);
        hip_check(hipGetLastError(), "application HIP FastGS forward submission");
        state->live = true;
    });
}
int lfs_hip_application_fast_backward(LfsHipApplicationFast* state, const LfsHipFastBackward* input, LfsHipError* error) {
    return app_fast_boundary(state, error, [&] {
        require(input && input->struct_bytes == sizeof(*input) && state->live && input->grad_image,
                "Application HIP backward needs a live forward and image gradient");
        require(input->densification_type >= 0 && input->densification_type <= 2, "Invalid HIP densification mode");
        const auto& forward = state->input;
        require(!input->grad_normal || forward.normal, "HIP normal backward needs normal forward output");
        fast_lfs::rasterization::FusedAdamSettings settings{};
        std::array<fast_lfs::rasterization::FusedAdamParam*, 6> groups{&settings.means,&settings.scaling,&settings.rotation,&settings.opacity,&settings.sh0,&settings.shN};
        for (size_t i=0; i<groups.size(); ++i) {
            const auto& source = input->groups[i];
            require(!source.enabled || (source.parameter && source.packed && source.bounds &&
                    source.primitives == forward.count && source.elements >= 0 && source.attributes >= 0 &&
                    ((i == 5 && source.joint_bits == 8) || (i != 5 && source.joint_bits == 16))),
                    "Invalid application HIP fused Adam group");
            *groups[i] = app_adam_group(source);
            require(!source.gradient_out || (source.parameter && source.primitives == forward.count && source.elements > 0),
                    "Invalid HIP raw gradient capture binding");
            settings.enabled |= source.enabled != 0 || source.gradient_out != 0;
        }
        require(settings.enabled, "Application HIP FastGS requires fused Adam state");
        settings.beta1 = input->beta1; settings.beta2 = input->beta2; settings.eps = input->eps;
        settings.scale_reg_weight = input->scale_reg_weight; settings.scale_reg_log = input->scale_reg_log != 0;
        settings.scale_reg_normalizer = input->scale_reg_normalizer; settings.opacity_reg_weight = input->opacity_reg_weight;
        settings.flatten_reg_weight = input->flatten_reg_weight; settings.erank_reg_weight = input->erank_reg_weight;
        settings.dc_reg_weight = input->dc_reg_weight; settings.sh_rest_reg_weight = input->sh_reg_weight;
        settings.scale_reg_loss_out = app_pointer(input->scale_loss); settings.opacity_reg_loss_out = app_pointer(input->opacity_loss);
        settings.erank_reg_loss_out = app_pointer(input->erank_loss); settings.dc_reg_loss_out = app_pointer(input->dc_loss);
        settings.sh_rest_reg_loss_out = app_pointer(input->sh_loss); settings.rendered_count = app_pointer(input->rendered_count);
        settings.per_splat_mean_step = input->per_splat_mean_step != 0; settings.mean_step_median_extent = input->median_extent;
        settings.mean_step_far_mask = app_pointer<bool>(input->mean_far); settings.mean_step_far_mask_n = input->mean_far_count;
        auto& f = state->workspace;
        const size_t helper_count = f.reuse_phase_helpers ? std::max(f.result.n_visible, 1) : forward.count;
        auto* helper = f.backward_helpers(multiply(helper_count, 16 * sizeof(float)));
        float* opacity = helper;
        auto* color = reinterpret_cast<float3*>(helper+helper_count);
        auto* mean = reinterpret_cast<float2*>(helper+helper_count*4);
        auto* conic = reinterpret_cast<float3*>(helper+helper_count*6);
        float* depth = helper+helper_count*9;
        auto* normal = reinterpret_cast<float3*>(helper+helper_count*10);
#ifdef LFS_HIP_FAST_PACKED_GRADIENTS
        opacity = helper+5; color = reinterpret_cast<float3*>(helper+6);
        mean = reinterpret_cast<float2*>(helper); conic = reinterpret_cast<float3*>(helper+2);
        depth = helper+9; normal = reinterpret_cast<float3*>(helper+10);
#endif
        fast_lfs::rasterization::BackgroundAlphaGradient background{};
        background.bg_color = app_pointer(forward.background_color);
        background.bg_image = app_pointer(forward.background_image);
        background.grad_alpha_extra = app_pointer(input->grad_alpha);
        state->optimizer_submitted = true;
        fast_lfs::rasterization::backward(app_pointer(input->error_map), app_pointer(input->grad_image), background,
            app_pointer(input->grad_depth), app_pointer(input->grad_normal), app_pointer(forward.image), app_pointer(forward.alpha),
            app_pointer<float3>(forward.means), app_pointer<float3>(forward.scales), app_pointer<float4>(forward.rotations),
            app_pointer(forward.opacities), app_pointer<float4>(forward.shN), app_pointer<float4>(forward.view), app_pointer<float3>(forward.camera),
            f.result.primitive_normals, f.result.per_primitive_buffers, f.tiles, f.result.sorted_primitive_indices, f.result.primitive_work_indices,
            opacity, color, mean, conic, depth, input->grad_normal ? normal : nullptr, nullptr, app_pointer(input->densification),
            forward.count, f.result.n_instances, f.result.n_visible, forward.active_bases, forward.layout_bases, forward.width, forward.height,
            forward.fx, forward.fy, forward.cx, forward.cy, forward.mip_filter != 0,
            static_cast<DensificationType>(input->densification_type), settings, app_pointer<float2>(forward.sh_bounds), forward.sh_cells, forward.sh_bits,
            app_pointer(input->edge_map), app_pointer(input->edge_scores), app_pointer(input->impact_sum), app_pointer(input->impact_peak), state->queue);
        hip_check(hipGetLastError(), "application HIP FastGS backward submission");
        state->optimizer_submitted = false;
        state->live = false;
    });
}
int lfs_hip_application_fast_release(LfsHipApplicationFast* state, LfsHipError* error) {
    return app_fast_boundary(state, error, [&] { state->live = false; });
}
int lfs_hip_application_fast_info(LfsHipApplicationFast* state, LfsHipFastInfo* output, LfsHipError* error) {
    return app_fast_boundary(state, error, [&] {
        require(output && output->struct_bytes == sizeof(*output), "Invalid HIP FastGS statistics descriptor");
        *output = {sizeof(*output), state->budget.live, state->budget.peak, state->budget.limit, state->workspace.result.n_instances, state->workspace.result.n_visible};
    });
}
int lfs_hip_application_fast_destroy(LfsHipApplicationFast* state, LfsHipError* error) {
    return boundary(nullptr, error, [&] {
        require(state, "Missing application HIP FastGS state");
        DeviceScope selected(state->device);
        hip_check(hipStreamSynchronize(state->queue), "application HIP FastGS teardown");
        delete state;
    });
}
}
#endif
