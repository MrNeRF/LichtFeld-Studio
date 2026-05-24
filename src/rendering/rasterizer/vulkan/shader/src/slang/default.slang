// Default strategy for Gaussian densification
// Based on https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/default.py

#define GROUP_SIZE 256

#define DEFAULT_PHASE_UpdateState 0
#define DEFAULT_PHASE_ComputeGrowMask 1
#define DEFAULT_PHASE_Duplicate 2
#define DEFAULT_PHASE_Split 3
#define DEFAULT_PHASE_ComputePruneMask 4
#define DEFAULT_PHASE_Prune 5
#define DEFAULT_PHASE_PruneMean 51
#define DEFAULT_PHASE_PruneSH 52
#define DEFAULT_PHASE_ResetOpacity 6

#ifndef DEFAULT_PHASE
#define DEFAULT_PHASE -1
#endif

#include "strategy_utils.slang"


struct Uniforms {
    uint32_t num_splats;
    uint32_t old_num_splats;
    uint32_t seed;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t step;
    float prune_opa;
    float grow_grad2d;
    float grow_scale3d;
    float grow_scale2d;
    float prune_scale3d;
    float prune_scale2d;
    int32_t refine_scale2d_stop_iter;
    int32_t refine_start_iter;
    int32_t refine_stop_iter;
    int32_t reset_every;
    int32_t refine_every;
    int32_t pause_refine_after_reset;
};




#if DEFAULT_PHASE == DEFAULT_PHASE_UpdateState

layout(binding=0) StructuredBuffer<float2> v_xy_vs;  // (N, 2)
layout(binding=1) StructuredBuffer<int32_t> radii;  // (N,)

layout(binding=2) RWStructuredBuffer<float2> running_grad;  // (N, 2)
layout(binding=3) RWStructuredBuffer<float> running_radii;  // (N,)

#define updateState main

[numthreads(GROUP_SIZE, 1, 1)]
void updateState(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.num_splats;
    const uint32_t gid = thread_id.x;
    if (gid >= num_splats)
        return;

    const float width = float(uniforms.image_width);
    const float height = float(uniforms.image_height);

    float grad = 0.5 * length(v_xy_vs[gid] * float2(width, height));
    float radius = float(radii[gid]) / max(width, height);

    if (radius > 0.0) {
        running_grad[gid] = running_grad[gid] + float2(grad, 1.0);
        if (uniforms.refine_scale2d_stop_iter > 0)
            running_radii[gid] = max(running_radii[gid], radius);
    }
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_UpdateState




#if DEFAULT_PHASE == DEFAULT_PHASE_ComputeGrowMask

layout(binding=0) StructuredBuffer<float2> running_grad;  // (N, 2)
layout(binding=1) StructuredBuffer<float> running_radii;  // (N,)
layout(binding=2) StructuredBuffer<float4> scales_opacs;  // (N, 4)

layout(binding=3) RWStructuredBuffer<int32_t> dupli_mask;  // (N,)
layout(binding=4) RWStructuredBuffer<int32_t> split_mask;  // (N,)

#define computeGrowMask main

[numthreads(GROUP_SIZE, 1, 1)]
void computeGrowMask(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.num_splats;
    const uint32_t gid = thread_id.x;
    if (gid >= num_splats)
        return;

    float2 grads = running_grad[gid];
    float grad = grads.x / max(grads.y, 1.0);
    float3 scale = scales_opacs[gid].xyz;

    bool is_grad_high = grad > uniforms.grow_grad2d;
    bool is_small = max(scale.x, max(scale.y, scale.z)) <= uniforms.grow_scale3d;
    bool is_dupli = is_grad_high && is_small;

    bool is_large = !is_small;
    bool is_split = is_grad_high && is_large;
    if (uniforms.step < uniforms.refine_scale2d_stop_iter)
        is_split |= (running_radii[gid] > uniforms.grow_scale2d);

    dupli_mask[gid] = int32_t(is_dupli);
    split_mask[gid] = int32_t(is_split);
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_ComputeGrowMask




#if DEFAULT_PHASE == DEFAULT_PHASE_Duplicate

layout(binding=0) StructuredBuffer<uint32_t> index_map;  // (dN,)

layout(binding=1) RWStructuredBuffer<float> means;  // (N, 3)
layout(binding=2) RWStructuredBuffer<float4> sh_coeffs;  // (N, 16, 3) = (N, 12, 4)
layout(binding=3) RWStructuredBuffer<float4> rotations;  // (N, 4)
layout(binding=4) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

layout(binding=5) RWStructuredBuffer<float> g_means;  // (N, 2, 3)
layout(binding=6) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N, 16, 3) = (N, 12, 4)
layout(binding=7) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N, 16, 3) = (N, 12, 4)
layout(binding=8) RWStructuredBuffer<float4> g_rotations;  // (N, 2, 4)
layout(binding=9) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

layout(binding=10) RWStructuredBuffer<float2> running_grad;  // (N, 2)
layout(binding=11) RWStructuredBuffer<float> running_radii;  // (N,)

#define duplicate main

[numthreads(GROUP_SIZE, 1, 1)]
void duplicate(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.num_splats;
    const uint32_t id_dst = thread_id.x + uniforms.old_num_splats;
    if (id_dst >= num_splats)
        return;

    uint32_t id_src = index_map[thread_id.x];

    // means
    write_mean(read_mean(id_src, means), id_dst, means);
    _FOR_LOOP(i, 2) write_mean(float3(0.0), id_dst, g_means, 2, i);

    // SH coeffs
    _FOR_LOOP(i, 12) sh_coeffs[shAt(id_dst,i)] = sh_coeffs[shAt(id_src,i)];
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(id_dst,i)] = float4(0.0);

    // rotations
    rotations[id_dst] = rotations[id_src];
    _FOR_LOOP(i, 2) g_rotations[id_dst*2+i] = float4(0.0);

    // scales and opacities
    scales_opacs[id_dst] = scales_opacs[id_src];
    _FOR_LOOP(i, 2) g_scales_opacs[id_dst*2+i] = float4(0.0);

    // running states
    running_grad[id_dst] = running_grad[id_src];
    running_radii[id_dst] = running_radii[id_src];
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_Duplicate




#if DEFAULT_PHASE == DEFAULT_PHASE_Split

layout(binding=0) StructuredBuffer<uint32_t> index_map;  // (dN,)

layout(binding=1) RWStructuredBuffer<float> means;  // (N, 3)
layout(binding=2) RWStructuredBuffer<float4> sh_coeffs;  // (N, 16, 3) = (N, 12, 4)
layout(binding=3) RWStructuredBuffer<float4> rotations;  // (N, 4)
layout(binding=4) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)

layout(binding=5) RWStructuredBuffer<float> g_means;  // (N, 2, 3)
layout(binding=6) RWStructuredBuffer<float4> g_sh_coeffs_1;  // (N, 16, 3) = (N, 12, 4)
layout(binding=7) RWStructuredBuffer<float4> g_sh_coeffs_2;  // (N, 16, 3) = (N, 12, 4)
layout(binding=8) RWStructuredBuffer<float4> g_rotations;  // (N, 2, 4)
layout(binding=9) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

layout(binding=10) RWStructuredBuffer<float2> running_grad;  // (N, 2)
layout(binding=11) RWStructuredBuffer<float> running_radii;  // (N,)

#define split main

[numthreads(GROUP_SIZE, 1, 1)]
void split(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.num_splats;
    const uint32_t id_dst = thread_id.x + uniforms.old_num_splats;
    if (id_dst >= num_splats)
        return;

    uint32_t id_src = index_map[thread_id.x];

    float4 scale_opac = scales_opacs[id_src];
    float3 scale = scale_opac.xyz;
    float opac = scale_opac.w;

    // compute displacement
    float4 quat = normalize(rotations[id_src]);
    float3x3 L = sqrt_cov_from_quat_scales(quat, scale);
    float3 n1 = mul(L, randn3(uniforms.seed, id_src));
    float3 n2 = mul(L, randn3(uniforms.seed, id_dst));

    // means (displace)
    float3 mean = read_mean(id_src, means);
    write_mean(mean + n1, id_src, means);
    write_mean(mean + n2, id_dst, means);

    // SH coeffs (copy)
    _FOR_LOOP(i, 12) sh_coeffs[shAt(id_dst,i)] = sh_coeffs[shAt(id_src,i)];

    // rotations (copy, but normalized)
    rotations[id_src] = quat;
    rotations[id_dst] = quat;

    // scales (divide by 1.6 per Inria/GSplat)
    // opacities (copy, optionally use revised scale)
    scale /= 1.6;
    // opac = 1.0 - sqrt(1.0 - opac);
    scales_opacs[id_src] = float4(scale, opac);
    scales_opacs[id_dst] = float4(scale, opac);

    // gradients (both zero)
    _FOR_LOOP(i, 2) write_mean(float3(0.0), id_src, g_means, 2, i);
    _FOR_LOOP(i, 2) write_mean(float3(0.0), id_dst, g_means, 2, i);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(id_src,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(id_src,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_1[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 12) g_sh_coeffs_2[shAt(id_dst,i)] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[id_src*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_rotations[id_dst*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[id_src*2+i] = float4(0.0);
    _FOR_LOOP(i, 2) g_scales_opacs[id_dst*2+i] = float4(0.0);

    // running states, copy
    running_grad[id_dst] = running_grad[id_src];
    running_radii[id_dst] = running_radii[id_src];
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_Split




#if DEFAULT_PHASE == DEFAULT_PHASE_ComputePruneMask

layout(binding=0) StructuredBuffer<float4> scales_opacs;  // (N, 4)
layout(binding=1) StructuredBuffer<float> running_radii;  // (N,)

// ~(prune mask)
layout(binding=2) RWStructuredBuffer<int32_t> keep_mask;  // (N,)

#define computePruneMask main

[numthreads(GROUP_SIZE, 1, 1)]
void computePruneMask(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.num_splats;
    const uint32_t gid = thread_id.x;
    if (gid >= num_splats)
        return;

    float4 scale = scales_opacs[gid];
    bool is_prune = scale.w < uniforms.prune_opa;
    is_prune |= (!isfinite(length(scale)));

    if (uniforms.step > uniforms.reset_every) {
        bool is_too_big = max(scale.x, max(scale.y, scale.z)) > uniforms.prune_scale3d;
        if (uniforms.step < uniforms.refine_scale2d_stop_iter)
            is_too_big |= (running_radii[gid] > uniforms.prune_scale2d);
        is_prune = is_prune || is_too_big;
    }

    keep_mask[gid] = int32_t(!is_prune);
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_ComputePruneMask




#if DEFAULT_PHASE == DEFAULT_PHASE_Prune

layout(binding=0) StructuredBuffer<int32_t> keep_mask;  // (N,)
layout(binding=1) StructuredBuffer<int32_t> keep_mask_psa;  // (N,)

layout(binding=2) StructuredBuffer<float> buffer_in;  // (N, stride)
layout(binding=3) RWStructuredBuffer<float> buffer_out;  // (N, stride)

#define prune main

[numthreads(GROUP_SIZE, 1, 1)]
void prune(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.old_num_splats;
    const uint32_t stride = uniforms.num_splats;
    const uint32_t id_src = thread_id.x / stride;
    if (id_src >= num_splats)
        return;

    if (keep_mask[id_src] == 0)
        return;

    uint32_t id_dst = id_src == 0 ? 0 : keep_mask_psa[id_src-1];

    buffer_out[id_dst*stride+thread_id.x%stride] = buffer_in[thread_id.x];
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_Prune

#if DEFAULT_PHASE == DEFAULT_PHASE_PruneMean

layout(binding=0) StructuredBuffer<int32_t> keep_mask;  // (N,)
layout(binding=1) StructuredBuffer<int32_t> keep_mask_psa;  // (N,)

layout(binding=2) StructuredBuffer<float> buffer_in;  // (N, stride)
layout(binding=3) RWStructuredBuffer<float> buffer_out;  // (N, stride)

#define prune main

[numthreads(GROUP_SIZE, 1, 1)]
void prune(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.old_num_splats;
    const uint stride = uniforms.num_splats;
    const uint32_t id_src = thread_id.x;
    if (id_src >= num_splats)
        return;

    if (keep_mask[id_src] == 0)
        return;

    uint32_t id_dst = id_src == 0 ? 0 : keep_mask_psa[id_src-1];

    for(uint i = 0; i < stride; i++)
        write_mean(read_mean(id_src, buffer_in, stride, i), id_dst, buffer_out, stride, i);
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_PruneMean

#if DEFAULT_PHASE == DEFAULT_PHASE_PruneSH

layout(binding=0) StructuredBuffer<int32_t> keep_mask;  // (N,)
layout(binding=1) StructuredBuffer<int32_t> keep_mask_psa;  // (N,)

layout(binding=2) StructuredBuffer<float4> buffer_in;  // (N, stride)
layout(binding=3) RWStructuredBuffer<float4> buffer_out;  // (N, stride)

#define prune main

[numthreads(GROUP_SIZE, 1, 1)]
void prune(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.old_num_splats;
    const uint stride = uniforms.num_splats;
    const uint32_t id_src = thread_id.x;
    if (id_src >= num_splats)
        return;

    if (keep_mask[id_src] == 0)
        return;

    uint32_t id_dst = id_src == 0 ? 0 : keep_mask_psa[id_src-1];

    _FOR_LOOP(j, 12) for(uint i = 0; i < stride; i++)
        buffer_out[shAt(id_dst,j)*stride+i] = buffer_in[shAt(id_src,j)*stride+i];
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_PruneSH




#if DEFAULT_PHASE == DEFAULT_PHASE_ResetOpacity

layout(binding=0) RWStructuredBuffer<float4> scales_opacs;  // (N, 4)
layout(binding=1) RWStructuredBuffer<float4> g_scales_opacs;  // (N, 2, 4)

#define resetOpacity main

[numthreads(GROUP_SIZE, 1, 1)]
void resetOpacity(
    int32_t3 thread_id : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t num_splats = uniforms.num_splats;
    const uint32_t gid = thread_id.x;
    if (gid >= num_splats)
        return;

    scales_opacs[gid].a = min(scales_opacs[gid].w, 2.0*uniforms.prune_opa);
    g_scales_opacs[2*gid+0].a = 0.0;
    g_scales_opacs[2*gid+1].a = 0.0;
}

#endif  // DEFAULT_PHASE == DEFAULT_PHASE_ResetOpacity

