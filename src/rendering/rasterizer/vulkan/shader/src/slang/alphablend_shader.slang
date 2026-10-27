// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "config.slang"

#ifndef LFS_REVEAL
#define LFS_REVEAL 0
#endif

import utils;

#ifndef LFS_STOCHASTIC_RASTER
#define LFS_STOCHASTIC_RASTER 0
#endif
#if LFS_STOCHASTIC_RASTER
#include "stochastic_sampling.slang"
#endif

#ifndef LFS_3DGUT_RASTERIZE
#define LFS_3DGUT_RASTERIZE 0
#endif

#ifndef LFS_RASTER_OVERLAYS
#define LFS_RASTER_OVERLAYS 1
#endif

#ifndef LFS_SPARK_RAD_RASTER
#define LFS_SPARK_RAD_RASTER 0
#endif

#ifndef LFS_SKIP_DENSE_TILES
#define LFS_SKIP_DENSE_TILES 0
#endif

#ifndef LFS_SELECTION_OUTLINE
#define LFS_SELECTION_OUTLINE 0
#endif

// Outline variants are a separate, selected-tile-only mask dispatch.
#define LFS_OUTLINE_ONLY LFS_SELECTION_OUTLINE

#define SHARED_STRUCT 1

#if SUBTILE_SIZE > 0
#define SUBTILE_PER_SUBGROUP max(SUBGROUP_SIZE / SUBTILE_SIZE, 1)
#define SUBGRID_HEIGHT (TILE_HEIGHT/SUBTILE_HEIGHT)
#define SUBGRID_WIDTH (TILE_WIDTH/SUBTILE_WIDTH)
#define SUBGRID_SIZE (SUBGRID_HEIGHT*SUBGRID_WIDTH)
#endif


[ForceInline]
[Differentiable]
float4 update_pixel_state(float4 pixel_state_t_nm1, float4 gauss_rgba_t_n)
{
    float3 color_t_n = pixel_state_t_nm1.rgb + gauss_rgba_t_n.rgb * pixel_state_t_nm1.a;
    float transmittance_t_n = pixel_state_t_nm1.a * (float(1) - gauss_rgba_t_n.a);
    return float4(color_t_n, transmittance_t_n);
}

[ForceInline]
float4 undo_pixel_state(float4 pixel_state_t_n, float4 gauss_rgba_t_n)
{
    float transmittance_t_nm1 = pixel_state_t_n.a / (float(1) - gauss_rgba_t_n.a);
    float3 color_t_nm1 = pixel_state_t_n.rgb - gauss_rgba_t_n.rgb * transmittance_t_nm1;
    return float4(color_t_nm1, transmittance_t_nm1);
}

static const float FAR_DEPTH = 1.0e10f;

#if SUBTILE_SIZE > 0

[ForceInline]
void decodeThreadToSubtiles(uint thread_rank, inout uint2 pix_coord, out uint out_subgrid_idx, out uint out_pixel_rank) {
  #if 1  // align subtiles with subgroups
    uint subgrid_idx = thread_rank / SUBTILE_SIZE;
    uint subtile_offset = thread_rank % SUBTILE_SIZE;
    uint2 subgrid_idx2 = uint2(subgrid_idx % SUBGRID_WIDTH, subgrid_idx / SUBGRID_WIDTH);
    uint2 subtile_offset2 = uint2(subtile_offset % SUBTILE_WIDTH, subtile_offset / SUBTILE_WIDTH);
    uint2 tile_idx2 = pix_coord / uint2(TILE_WIDTH, TILE_HEIGHT);  // groupID.xy
    uint2 tile_offset2 = subgrid_idx2 * uint2(SUBTILE_WIDTH, SUBTILE_HEIGHT) + subtile_offset2;
    pix_coord = uint2(tile_idx2) * uint2(TILE_WIDTH, TILE_HEIGHT) + tile_offset2;
  #else  // reference (slower)
    uint2 tile_idx2 = pix_coord / uint2(TILE_WIDTH, TILE_HEIGHT);  // groupID.xy
    uint2 tile_offset2 = pix_coord % uint2(TILE_WIDTH, TILE_HEIGHT);  // groupThreadID.xy
    uint2 subgrid_idx2 = tile_offset2 / uint2(SUBTILE_WIDTH, SUBTILE_HEIGHT);
    uint2 subtile_offset2 = tile_offset2 % uint2(SUBTILE_WIDTH, SUBTILE_HEIGHT);
    uint subgrid_idx = subgrid_idx2.y * SUBGRID_WIDTH + subgrid_idx2.x;
    uint subtile_offset = subtile_offset2.y * SUBTILE_WIDTH + subtile_offset2.x;
  #endif
    out_subgrid_idx = subgrid_idx;
    out_pixel_rank = tile_offset2.y * TILE_WIDTH + tile_offset2.x;
}

[ForceInline]
uint32_t computeTileOverlapBitmask(bool exact, float2 g_xy, float4 g_inv_cov_opac) {
    uint32_t bitmask = 0;
    float extend = extend_from_opacity(g_inv_cov_opac.w);
    float3 g_inv_cov = g_inv_cov_opac.xyz;
    g_inv_cov /= (extend*extend);
    for (int yi = 0; yi < SUBGRID_HEIGHT; yi++) {
        float y0 = float(yi * SUBTILE_HEIGHT) - g_xy.y;
        float y1 = y0 + SUBTILE_HEIGHT;
        for (int xi = 0; xi < SUBGRID_WIDTH; xi++) {
            float x0 = float(xi * SUBTILE_WIDTH) - g_xy.x;
            float x1 = x0 + SUBTILE_WIDTH;
            bool overlap = exact ?
                ellipse_box_overlap_test(g_inv_cov, x0, x1, y0, y1) :
                ellipse_box_overlap_test_fast(g_inv_cov, x0, x1, y0, y1);
            bitmask |= (uint32_t(overlap) << (yi * SUBGRID_WIDTH + xi));
        }
    }
    return bitmask;
}

#endif


// forward
#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

#if SHARED_STRUCT
#if LFS_3DGUT_RASTERIZE
groupshared Splat_3D_GUT_AlphaBlend collected_gut_splats[TILE_SIZE];
#else
groupshared Splat_2D_AlphaBlend collected_splats[TILE_SIZE];
#endif
#else
groupshared float2 collected_xy_vs[TILE_SIZE];
groupshared float4 collected_inv_cov_vs_opacity[TILE_SIZE];
#endif

groupshared uint32_t active_subgroups[TILE_SIZE / SUBGROUP_SIZE];
groupshared uint32_t collected_gauss_idx[TILE_SIZE];
#if LFS_STOCHASTIC_RASTER
groupshared float collected_depth[TILE_SIZE];
groupshared float collected_depth_metric[TILE_SIZE];
groupshared uint2 collected_noise[TILE_SIZE];
#endif
#if LFS_RASTER_OVERLAYS
// Packed flags | selection_byte | preview_byte — one global load set per splat
// per workgroup instead of per pixel.
groupshared uint32_t collected_overlay[TILE_SIZE];
#endif

#if SUBTILE_SIZE > 0
groupshared uint32_t subtile_overlap_bitmasks[TILE_SIZE];
#endif


layout(binding=0) StructuredBuffer<int32_t> sorted_gauss_idx;
layout(binding=1) StructuredBuffer<int32_t> tile_ranges;
layout(binding=2) StructuredBuffer<float2> xy_vs;
layout(binding=3) StructuredBuffer<float4> inv_cov_vs_opacity;
layout(binding=4) StructuredBuffer<float> rgb;
layout(binding=5) StructuredBuffer<float> depths;
#if LFS_STOCHASTIC_RASTER
layout(binding=15) StructuredBuffer<uint> stochastic_depth_keys;
#endif

#if LFS_3DGUT_RASTERIZE
layout(binding=6) StructuredBuffer<float> xyz_ws;
layout(binding=7) ByteAddressBuffer rotations;
layout(binding=8) ByteAddressBuffer scaling_raw;
layout(binding=9) ByteAddressBuffer opacity_raw;

layout(binding=10) RWStructuredBuffer<float4> final_pixel_state;
layout(binding=11) RWStructuredBuffer<float> final_pixel_depth;
layout(binding=12) RWStructuredBuffer<int32_t> n_contributors;
layout(binding=13) RWStructuredBuffer<float> pixel_depth_weight;
#else
layout(binding=6) RWStructuredBuffer<float4> final_pixel_state;
layout(binding=7) RWStructuredBuffer<float> final_pixel_depth;
layout(binding=8) RWStructuredBuffer<int32_t> n_contributors;
layout(binding=9) RWStructuredBuffer<float> pixel_depth_weight;
#endif
#if LFS_3DGUT_RASTERIZE
layout(binding=14) ByteAddressBuffer selection_mask;
layout(binding=15) ByteAddressBuffer preview_mask;
layout(binding=16) StructuredBuffer<float4> selection_colors;
layout(binding=17) ByteAddressBuffer gaussian_overlay_flags;
layout(binding=18) StructuredBuffer<float4> overlay_params;
layout(binding=19) ByteAddressBuffer transform_indices;
layout(binding=20) StructuredBuffer<float4> model_transforms;
layout(binding=21) StructuredBuffer<uint> lod_indices;
layout(binding=22) StructuredBuffer<uint> lod_logical_indices;
layout(binding=23) StructuredBuffer<float> lod_weights;
#else
layout(binding=10) ByteAddressBuffer selection_mask;
layout(binding=11) ByteAddressBuffer preview_mask;
layout(binding=12) StructuredBuffer<float4> selection_colors;
layout(binding=13) ByteAddressBuffer gaussian_overlay_flags;
layout(binding=14) StructuredBuffer<float4> overlay_params;
#endif

#if LFS_SELECTION_OUTLINE
#if LFS_3DGUT_RASTERIZE
layout(binding=24) RWStructuredBuffer<uint> outline_ids;
#define outline_lod_indices lod_indices
#define outline_logical_indices lod_logical_indices
layout(binding=25) RWStructuredBuffer<float4> outline_scene_depths;
layout(binding=26) StructuredBuffer<uint> outline_tiles;
#else
layout(binding=15) RWStructuredBuffer<uint> outline_ids;
layout(binding=16) ByteAddressBuffer transform_indices;
layout(binding=17) StructuredBuffer<uint> outline_lod_indices;
layout(binding=18) StructuredBuffer<uint> outline_logical_indices;
layout(binding=19) RWStructuredBuffer<float4> outline_scene_depths;
layout(binding=20) StructuredBuffer<uint> outline_tiles;
#endif
#endif

#if LFS_3DGUT_RASTERIZE
[ForceInline]
bool inverse_3x3(float3x3 m, out float3x3 inv_m)
{
    float3 r0 = float3(m[0][0], m[0][1], m[0][2]);
    float3 r1 = float3(m[1][0], m[1][1], m[1][2]);
    float3 r2 = float3(m[2][0], m[2][1], m[2][2]);
    float3 c0 = cross(r1, r2);
    float3 c1 = cross(r2, r0);
    float3 c2 = cross(r0, r1);
    float det = dot(r0, c0);
    if (abs(det) <= 1.0e-8f) {
        inv_m = float3x3(1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 1.0f);
        return false;
    }

    float inv_det = 1.0f / det;
    inv_m = float3x3(c0.x * inv_det, c1.x * inv_det, c2.x * inv_det,
                     c0.y * inv_det, c1.y * inv_det, c2.y * inv_det,
                     c0.z * inv_det, c1.z * inv_det, c2.z * inv_det);
    return true;
}

[ForceInline]
Splat_3D_GUT_AlphaBlend load_transformed_splat_gut_alphablend(
    int32_t g_idx,
    Uniforms uniforms)
{
    // Sorted ids address projected slots, not the physical RAD page pool.
    const uint slot = uint(g_idx);
    const bool lod_active = (uniforms.lod_enabled & 1u) != 0u;
    const uint physical_idx = lod_active ? lod_indices[slot] : slot;
    const uint logical_idx = lod_active && (uniforms.lod_enabled & 8u) != 0u
        ? lod_logical_indices[slot] : physical_idx;
    Splat_3D_GUT_AlphaBlend g =
        load_splat_gut_alphablend(physical_idx, slot, xyz_ws, rotations, scaling_raw,
                                  opacity_raw, rgb, uniforms.lod_enabled, uniforms.lod_page_splats > 0u,
                                  (uniforms.lod_enabled & 4u) == 0u ? uniforms.splat_render_profile : 0u);
#if LFS_REVEAL
    float3 source_scales = 1.0f / max(float3(length(g.iscl_rot[0]), length(g.iscl_rot[1]), length(g.iscl_rot[2])), 1e-8f);

    float3x3 inv_linear;
    bool has_inverse = false;
    if (uniforms.step != 0u) {
        float4 emphasis_flags = overlay_params[OVERLAY_PARAM_EMPHASIS_FLAGS];
        bool transform_indices_enabled = overlay_enabled(emphasis_flags.y);
        int transform_idx = transform_indices_enabled ? read_int32(transform_indices, logical_idx) : 0;
        transform_idx = clamp(transform_idx, 0, int(uniforms.step) - 1);
        uint transform_base = uint(transform_idx) * 4u;
        g.xyz_ws = transform_point_rows(model_transforms, transform_base, g.xyz_ws);
        float3x3 linear = linear_transform_rows(model_transforms, transform_base);
        has_inverse = inverse_3x3(linear, inv_linear);
    }
    float4 reveal = overlay_params[OVERLAY_PARAM_REVEAL];
    if (overlay_enabled(reveal.x)) {
        float4 bounds = overlay_params[OVERLAY_PARAM_REVEAL_EFFECT_BOUNDS];
        LfsRevealEffect effect = lfs_reveal_effect(g.xyz_ws, source_scales, reveal, bounds);
        g.xyz_ws = effect.position;
        float3 ratio = source_scales / max(source_scales * effect.scale_mix + effect.seed_scale, 1e-8f);
        g.iscl_rot[0] *= ratio.x;
        g.iscl_rot[1] *= ratio.y;
        g.iscl_rot[2] *= ratio.z;
        float c = cos(effect.spin), s = sin(effect.spin);
        g.iscl_rot = mul(g.iscl_rot, float3x3(c,0,-s, 0,1,0, s,0,c));
        g.opacity = lerp(0.3f, g.opacity, effect.color_mix) * effect.opacity_multiplier + effect.color_add;
    }
    if (has_inverse)
        g.iscl_rot = mul(g.iscl_rot, inv_linear);
    if (lod_active && (uniforms.lod_enabled & 16u) != 0u)
        g.opacity *= clamp(lod_weights[slot], 0.0f, 1.0f);
#else
    if (lod_active && (uniforms.lod_enabled & 16u) != 0u)
        g.opacity *= clamp(lod_weights[slot], 0.0f, 1.0f);

    if (uniforms.step == 0u)
        return g;

    float4 emphasis_flags = overlay_params[OVERLAY_PARAM_EMPHASIS_FLAGS];
    bool transform_indices_enabled = overlay_enabled(emphasis_flags.y);
    int transform_idx = transform_indices_enabled ? read_int32(transform_indices, logical_idx) : 0;
    transform_idx = clamp(transform_idx, 0, int(uniforms.step) - 1);
    uint transform_base = uint(transform_idx) * 4u;

    g.xyz_ws = transform_point_rows(model_transforms, transform_base, g.xyz_ws);
    float3x3 linear = linear_transform_rows(model_transforms, transform_base);
    float3x3 inv_linear;
    if (inverse_3x3(linear, inv_linear)) {
        g.iscl_rot = mul(g.iscl_rot, inv_linear);
    }
#endif
    return g;
}

[ForceInline]
float gut_splat_view_depth(
    Splat_3D_GUT_AlphaBlend g,
    float3 ray_origin,
    float3 ray_direction,
    Camera cam)
{
    float3 local_origin = mul(g.iscl_rot, ray_origin - g.xyz_ws);
    float3 local_direction = mul(g.iscl_rot, ray_direction);
    float denom = dot(local_direction, local_direction);
    if (denom <= 1.0e-12f) {
        return FAR_DEPTH;
    }

    float t = -dot(local_direction, local_origin) / denom;
    if (t <= 0.0f || !isfinite(t)) {
        return FAR_DEPTH;
    }

    float3 closest_ws = ray_origin + t * ray_direction;
    float depth = geom_transform_points(closest_ws, cam.world_view_transform).z;
    return depth > NEAR_CLIP && isfinite(depth) ? depth : FAR_DEPTH;
}
#endif

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint32_t H = uniforms.image_height;
    const uint32_t W = uniforms.image_width;
    const uint grid_width = uniforms.grid_width;
    const uint32_t camera_H = uniforms.camera_height == 0 ? H : uniforms.camera_height;
    const uint32_t camera_W = uniforms.camera_width == 0 ? W : uniforms.camera_width;

    uint2 pix_coord = dispatchThreadID.xy;

    uint tile_idx = groupID.y * grid_width + groupID.x;
#if LFS_OUTLINE_ONLY
    if (outline_tiles[tile_idx] == 0u) return;
#endif
    uint32_t tile_idx_start = tile_ranges[tile_idx];
    uint32_t tile_idx_end = tile_ranges[tile_idx+1];
    #if LFS_SKIP_DENSE_TILES
    if (tile_idx_end > tile_idx_start &&
        tile_idx_end - tile_idx_start > RASTER_DENSE_TILE_THRESHOLD) {
        return;
    }
    #endif

    uint thread_rank = groupThreadID.y * TILE_WIDTH + groupThreadID.x;

    #if SUBTILE_SIZE > 0
        uint2 pix_coord_base = groupID.xy * uint2(TILE_WIDTH, TILE_HEIGHT);
        uint subgrid_idx, pixel_rank;
        decodeThreadToSubtiles(thread_rank, pix_coord, subgrid_idx, pixel_rank);
    #endif
    bool is_inside = (pix_coord.x < W && pix_coord.y < H);
    const uint pix_id = pix_coord.y * W + pix_coord.x;
    float4 curr_pixel_state = float4(0, 0, 0, 1);
    float median_depth = FAR_DEPTH;
    float weighted_depth = 0.0f;
    float weighted_alpha = 0.0f;
    int32_t local_n_contrib = 0;
#if LFS_STOCHASTIC_RASTER
    uint selected_id = 0xffffffffu;
    float nearest_metric = asfloat(0x7f800000u);
#endif
    if (uniforms.depth_wave > 0u && is_inside) {
        curr_pixel_state = final_pixel_state[pix_id];
        local_n_contrib = n_contributors[pix_id];
#if LFS_STOCHASTIC_RASTER
        selected_id = asuint(local_n_contrib);
        nearest_metric = pixel_depth_weight[pix_id];
#endif
        if (uniforms.expected_far > 0.0f) {
            weighted_depth = final_pixel_depth[pix_id];
            weighted_alpha = pixel_depth_weight[pix_id];
        } else {
            median_depth = final_pixel_depth[pix_id];
        }
    }
#if LFS_SELECTION_OUTLINE
    uint outline_id = uniforms.depth_wave > 0u && is_inside ? outline_ids[pix_id] & 0x3fffffffu : 0u;
    const bool outline_has_indices = overlay_enabled(overlay_params[OVERLAY_PARAM_EMPHASIS_FLAGS].y);
    // All lanes in this workgroup belong to the same render tile. The
    // prepass stores its final selected rank in the canonical radial sort.
    uint outline_tile_bound = outline_tiles[groupID.y * uniforms.grid_width + groupID.x];
    uint outline_rank_offset = uniforms.grid_width * uniforms.grid_height;
    bool outline_active = is_inside && outline_tile_bound != 0u;
    float4 previous_outline_depths = uniforms.depth_wave > 0u && is_inside
                                        ? outline_scene_depths[pix_id] : float4(FAR_DEPTH, FAR_DEPTH, 0.0f, 0.0f);
    float outline_scene_depth = previous_outline_depths.x;
    float outline_selected_depth = previous_outline_depths.y;
    float outline_scene_coverage = previous_outline_depths.z;
    float outline_selected_coverage = previous_outline_depths.w;
    outline_active = outline_active && outline_selected_coverage < overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w;
#endif
    bool thread_active = LFS_OUTLINE_ONLY == 0 && is_inside && curr_pixel_state.a >= 0.0001f;
#if LFS_STOCHASTIC_RASTER
    // An opaque sample does not terminate the search: a nearer accepted splat
    // can occur later in this tile or in a subsequent unsorted wave.
    thread_active = is_inside;
#endif

    const uint2 projection_origin = full_frame_binning(uniforms)
        ? uint2(uniforms.render_origin_x, uniforms.render_origin_y) : uint2(0u);
    float2 center_pix_coord = pix_coord + projection_origin;
#if LFS_STOCHASTIC_RASTER
    // The reference sampler retains exactly the same bits, with pixel/frame
    // hashes hoisted out of the fragment loop and primitive hashes shared.
    const uint2 noise_pixel = pix_coord + projection_origin;
    const uint noise_pixel_seed = stochastic_hash(noise_pixel.x ^ stochastic_hash(noise_pixel.y));
    const uint noise_frame_seed = stochastic_hash(uniforms.stochastic_frame + 0x85ebca6bu);
#endif
  #if SUBTILE_SIZE > 0
    pix_coord_base += projection_origin;
  #endif
#if LFS_3DGUT_RASTERIZE
    // The host picks the *_rad variant for Spark RAD scenes; a compile-time flag keeps the
    // LOD-parent kernel out of the ordinary 3DGUT loop.
    const bool spark_rad_opacity_path = LFS_SPARK_RAD_RASTER != 0;
#else
    const bool spark_rad_opacity_path = (uniforms.lod_enabled & 4u) != 0u;
#endif
#if LFS_3DGUT_RASTERIZE
    float3 ray_origin = float3(0.0);
    float3 ray_direction = float3(0.0, 0.0, 1.0);
    Camera cam = load_camera(uniforms.world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
        base_camera_model(uniforms.camera_model), uniforms.dist_coeffs,
        camera_H, camera_W);
    const float2 camera_pix_coord =
        float2(pix_coord) + float2(uniforms.render_origin_x, uniforms.render_origin_y);
    WorldRay ray = pixel_to_world_ray(camera_pix_coord + 0.5f, cam);
    ray_origin = ray.origin;
    ray_direction = ray.direction;
    thread_active = thread_active && ray.valid;
  #if LFS_SELECTION_OUTLINE
    outline_active = outline_active && ray.valid;
  #endif
#endif

    const int shared_memory_rounds = ((tile_idx_end - tile_idx_start + TILE_SIZE - 1) / TILE_SIZE);

  #if LFS_RASTER_OVERLAYS
    // Hoist uniform overlay scalars once per thread; hot loop never reloads
    // overlay_params.
    const OverlayUniformState overlay_state = load_overlay_uniform_state(overlay_params);
  #endif

    int32_t num_cull = 0;  // for visualization
    int splats_left_to_process = tile_idx_end - tile_idx_start;
    for (int i = 0; i < shared_memory_rounds; i++)
    {
      #if LFS_OUTLINE_ONLY
        // Uniform exit before loading another batch. Once its first rank
        // exceeds the final selected rank, selected coverage cannot increase.
        // This remains safe for hidden selections and spans all depth waves.
        uint first_slot = uint(sorted_gauss_idx[tile_idx_start + i * TILE_SIZE]);
        if (outline_tiles[outline_rank_offset + first_slot] > outline_tile_bound)
            break;
      #endif
        // break if thread_active is false for all, doesn't seem to speed up
        // equivalent to CUDA `if (__syncthreads_count(thread_active) == 0) break;`
        const uint subgroup_idx = thread_rank / SUBGROUP_SIZE;
        const uint32_t subgroup_has_active_thread =
            uint32_t(WaveActiveAnyTrue(thread_active
              #if LFS_SELECTION_OUTLINE
                || outline_active
              #endif
            ));
        if (WaveIsFirstLane())
            active_subgroups[subgroup_idx] = subgroup_has_active_thread;
        GroupMemoryBarrierWithGroupSync();
        if (thread_rank == 0u) {
            uint32_t any_active = 0u;
            for (uint s = 0u; s < TILE_SIZE / SUBGROUP_SIZE; ++s)
                any_active |= active_subgroups[s];
            active_subgroups[0] = any_active;
        }
        GroupMemoryBarrierWithGroupSync();
        if (active_subgroups[0] == 0u)
            break;

        // Collectively fetch per-Gaussian data from global to shared
        int splat_pointer_offset = i * TILE_SIZE + thread_rank;
        if (tile_idx_start + splat_pointer_offset < tile_idx_end) {
            uint32_t coll_id = uint32_t(sorted_gauss_idx[tile_idx_start + splat_pointer_offset]);
#if LFS_STOCHASTIC_RASTER
            collected_depth[thread_rank] = read_t1_float(coll_id, depths);
            collected_depth_metric[thread_rank] = asfloat(stochastic_depth_keys[coll_id]);
            collected_noise[thread_rank] = stochastic_primitive_noise(coll_id);
#endif
            #if SHARED_STRUCT
              #if LFS_3DGUT_RASTERIZE
                collected_gut_splats[thread_rank] =
                    load_transformed_splat_gut_alphablend(int32_t(coll_id), uniforms);
              #else
                collected_splats[thread_rank] = load_splat_alphablend(coll_id, xy_vs, inv_cov_vs_opacity, rgb);
              #endif
              collected_gauss_idx[thread_rank] = coll_id;
            #else
            collected_xy_vs[thread_rank] = read_t2_float2(coll_id, xy_vs);
            collected_inv_cov_vs_opacity[thread_rank] = read_t4_float4(coll_id, inv_cov_vs_opacity);
            collected_gauss_idx[thread_rank] = coll_id;
            #endif
            #if LFS_RASTER_OVERLAYS
            collected_overlay[thread_rank] = load_overlay_per_splat_pack(
                coll_id, gaussian_overlay_flags, selection_mask, preview_mask, overlay_state);
            #endif
        }
        GroupMemoryBarrierWithGroupSync();

        // check if each splat overlaps with each subtile
        #if SUBTILE_SIZE > 0
        if (thread_rank < min(TILE_SIZE, splats_left_to_process)) {
            #if LFS_3DGUT_RASTERIZE
                subtile_overlap_bitmasks[thread_rank] = 0xffffffffu;
            #else
              #if SHARED_STRUCT
                Splat_2D_AlphaBlend g = collected_splats[thread_rank];
                float4 g_inv_cov_opac = g.inv_cov_vs_opacity;
                float2 g_xy = g.xy_vs;
              #else
                float4 g_inv_cov_opac = collected_inv_cov_vs_opacity[thread_rank];
                float2 g_xy = collected_xy_vs[thread_rank];
              #endif
                subtile_overlap_bitmasks[thread_rank] =
                    computeTileOverlapBitmask(false, g_xy - pix_coord_base, g_inv_cov_opac);
            #endif
        }
        GroupMemoryBarrierWithGroupSync();
        #endif

        if (thread_active
          #if LFS_SELECTION_OUTLINE
            || outline_active
          #endif
        ) {
            for (int j = 0; j < min(TILE_SIZE, splats_left_to_process); j++) {
                if (thread_active)
                    local_n_contrib++;

                // subtile culling
            #if SUBTILE_SIZE > 0
                int overlap = (subtile_overlap_bitmasks[j] >> subgrid_idx) & 1;
                if (overlap == 0) {
                    num_cull++;
                    continue;
                }
            #endif

            #if SHARED_STRUCT
                uint32_t splat_id = collected_gauss_idx[j];
              #if LFS_RASTER_OVERLAYS
                const uint overlay_pack = collected_overlay[j];
                const uint flags = unpack_overlay_flags(overlay_pack);
              #endif
#if LFS_STOCHASTIC_RASTER
                float splat_depth = collected_depth[j];
                float depth_metric = collected_depth_metric[j];
                // Unsorted traversal still permits an early depth test. Reject
                // farther fragments before evaluating their Gaussian kernel.
                if (depth_metric > nearest_metric || (depth_metric == nearest_metric && splat_id >= selected_id))
                    continue;
#else
                float splat_depth = read_t1_float(splat_id, depths);
#endif
                float4 gauss_rgba;
              #if LFS_RASTER_OVERLAYS
                Splat_2D_AlphaBlend overlay_splat;
              #endif
              #if LFS_3DGUT_RASTERIZE
                    Splat_3D_GUT_AlphaBlend g = collected_gut_splats[j];
                    if (uniforms.splat_render_profile == 1u && !spark_rad_opacity_path && !portal_gut_quad_contains(
                            center_pix_coord, xy_vs[splat_id], inv_cov_vs_opacity[splat_id].xyz,
                            g.opacity, cam))
                        continue;
                    gauss_rgba = evaluate_gut_splat_lod(g, ray_origin, ray_direction, spark_rad_opacity_path,
                                                        uniforms.splat_render_profile);
                  #if LFS_RASTER_OVERLAYS
                    overlay_splat = load_splat_alphablend(splat_id, xy_vs, inv_cov_vs_opacity, rgb);
                  #endif
              #else
                    Splat_2D_AlphaBlend g = collected_splats[j];
                    gauss_rgba = spark_rad_opacity_path
                                      ? evaluate_splat_spark_rad(g, center_pix_coord)
                                      : evaluate_splat_for_profile(g, center_pix_coord, uniforms.splat_render_profile);
                  #if LFS_RASTER_OVERLAYS
                    overlay_splat = g;
                  #endif
              #endif
                if (gauss_rgba.a < ALPHA_THRESHOLD)
                    continue;
#if LFS_SELECTION_OUTLINE
                // Blender's outline prepass contains selected objects only. An
                // unselected foreground footprint dims the selected silhouette;
                // it must neither clip it nor terminate the outline search.
                if (outline_active) {
                    // Match alpha compositing, including individually faint splats.
                    // Persist both accumulators across depth waves.
                    if (outline_scene_coverage < overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w) {
                        outline_scene_coverage += (1.0f - outline_scene_coverage) * gauss_rgba.a;
                        if (outline_scene_coverage >= overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w)
                            outline_scene_depth = splat_depth;
                    }
                    // Outline-only variants avoid the unrelated point-selection and ring work.
                    #if LFS_RASTER_OVERLAYS
                    uint outline_flags = flags;
                    #else
                    uint outline_flags = read_byte(gaussian_overlay_flags, splat_id);
                    #endif
                    if ((outline_flags & OVERLAY_FLAG_OUTLINE_SELECTED) != 0u && outline_selected_coverage < overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w) {
                        outline_selected_coverage += (1.0f - outline_selected_coverage) * gauss_rgba.a;
                        if (outline_selected_coverage >= overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w) {
                            uint logical_id = splat_id;
                            if ((uniforms.lod_enabled & 1u) != 0u)
                                logical_id = (uniforms.lod_enabled & 8u) != 0u
                                    ? outline_logical_indices[splat_id] : outline_lod_indices[splat_id];
                            uint node = outline_has_indices ? uint(max(read_int32(transform_indices, logical_id), 0)) : 0u;
                            outline_id = node + 2u;
                            outline_selected_depth = splat_depth;
                        }
                    }
                    outline_active = outline_selected_coverage < overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w || outline_scene_coverage < overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w;
                }
                if (!thread_active) {
                    if (!outline_active)
                        break;
                    continue;
                }
#endif

              #if LFS_3DGUT_RASTERIZE
                splat_depth = gut_splat_view_depth(g, ray_origin, ray_direction, cam);
                if (splat_depth < FAR_DEPTH)
                    splat_depth -= uniforms.depth_offset;
              #endif

            #else

                float2 g_xy = collected_xy_vs[j];
                float4 g_inv_cov_opacity = collected_inv_cov_vs_opacity[j];
                float3 g_inv_cov = g_inv_cov_opacity.xyz;
                float g_opacity = g_inv_cov_opacity.w;

                float alpha = spark_rad_opacity_path
                                  ? evaluate_splat_alpha_spark_rad(g_xy, g_opacity, g_inv_cov, center_pix_coord)
                                  : evaluate_splat_alpha_for_profile(g_xy, g_opacity, g_inv_cov, center_pix_coord, uniforms.splat_render_profile);
                if (alpha < ALPHA_THRESHOLD)
                    continue;

                uint32_t splat_id = collected_gauss_idx[j];
              #if LFS_RASTER_OVERLAYS
                const uint overlay_pack = collected_overlay[j];
                const uint flags = unpack_overlay_flags(overlay_pack);
              #endif
                float splat_depth = read_t1_float(splat_id, depths);
                float3 splat_rgb = read_t3_float3(splat_id, rgb);
                Splat_2D_AlphaBlend g = { g_inv_cov_opacity, g_xy, splat_rgb };
              #if LFS_RASTER_OVERLAYS
                Splat_2D_AlphaBlend overlay_splat = g;
              #endif
                float4 gauss_rgba = float4(splat_rgb*alpha, alpha);

            #endif

            #if LFS_RASTER_OVERLAYS
                const uint sel_status = selection_status(
                    overlay_splat.xy_vs,
                    splat_id,
                    flags,
                    unpack_overlay_selection_byte(overlay_pack),
                    unpack_overlay_preview_byte(overlay_pack),
                    overlay_state);
                bool selectable = (flags & OVERLAY_FLAG_NONSELECTABLE) == 0u;
                gauss_rgba = apply_projection_overlay_color(gauss_rgba,
                                                            overlay_splat.xy_vs,
                                                            flags,
                                                            overlay_state);
                gauss_rgba = apply_ring_overlay(gauss_rgba,
                                                overlay_splat,
                                                center_pix_coord,
                                                sel_status,
                                                flags,
                                                overlay_state.show_rings,
                                                overlay_state.ring_width,
                                                selection_colors);

                if (overlay_state.show_center_markers && selectable) {
                    const float INNER_RADIUS_SQ = 2.25f;
                    const float OUTER_RADIUS_SQ = 6.25f;
                    const float OUTLINE_DARKEN = 0.4f;
                    float2 marker_delta = overlay_splat.xy_vs - center_pix_coord;
                    float dist_sq = dot(marker_delta, marker_delta);
                    if (dist_sq > OUTER_RADIUS_SQ)
                        continue;

                    float3 marker_color = selection_overlay_target_color(sel_status, selection_colors);
                    if (dist_sq > INNER_RADIUS_SQ)
                        marker_color *= OUTLINE_DARKEN;

                    if (curr_pixel_state.a > 0.5f) {
                        median_depth = splat_depth;
                    }
                    curr_pixel_state = float4(marker_color, 0.0f);
                    thread_active = false;
                  #if LFS_SELECTION_OUTLINE
                    if (outline_active)
                        continue;
                  #endif
                    break;
                }

                gauss_rgba = apply_selection_overlay(gauss_rgba, sel_status, selection_colors);
            #endif

#if LFS_STOCHASTIC_RASTER
                const float threshold = uniforms.stochastic_mode == 2u
                    ? stochastic_blue_threshold(pix_coord + projection_origin, collected_noise[j], uniforms.stochastic_frame)
                    : stochastic_threshold(noise_pixel_seed, collected_noise[j].y, noise_frame_seed);
                if ((depth_metric < nearest_metric || (depth_metric == nearest_metric && splat_id < selected_id)) &&
                    threshold < gauss_rgba.a) {
                    curr_pixel_state = float4(gauss_rgba.rgb / gauss_rgba.a, 0.0f);
                    median_depth = splat_depth;
                    nearest_metric = depth_metric;
                    selected_id = splat_id;
                }
#else
                float4 new_pixel_state = update_pixel_state(curr_pixel_state, gauss_rgba);
                if (uniforms.expected_far > 0.0f && splat_depth <= uniforms.expected_far) {
                    float depth_weight = gauss_rgba.a * curr_pixel_state.a;
                    weighted_depth += depth_weight * splat_depth;
                    weighted_alpha += depth_weight;
                }
                if (median_depth >= FAR_DEPTH && curr_pixel_state.a > 0.5f && new_pixel_state.a <= 0.5f) {
                    median_depth = splat_depth;
                }
                if (new_pixel_state.a < 0.0001f) {
                    // This Splat never registred so we subtract it before we break.
                    local_n_contrib--;
                    thread_active = false;
                  #if LFS_SELECTION_OUTLINE
                    if (outline_active)
                        continue;
                  #endif
                    break;
                }
                curr_pixel_state = new_pixel_state;
#endif
            }
        }
        splats_left_to_process -= TILE_SIZE;
        GroupMemoryBarrierWithGroupSync();
    }

    if (is_inside) {
#if LFS_SELECTION_OUTLINE
        float epsilon = 1.0e-4f * max(1.0f, abs(outline_scene_depth));
        if (outline_id > 1u && outline_selected_depth > outline_scene_depth + epsilon)
            outline_id |= 0x80000000u;
        // Coverage is retained across waves; only the crossing splat supplies an ID.
        if (outline_selected_coverage < overlay_params[OVERLAY_PARAM_MARKER_FLAGS].w)
            outline_id |= 0x40000000u;
        outline_ids[pix_id] = outline_id;
        outline_scene_depths[pix_id] = float4(outline_scene_depth, outline_selected_depth, outline_scene_coverage, outline_selected_coverage);
#endif
#if !LFS_OUTLINE_ONLY
#if LFS_STOCHASTIC_RASTER
        n_contributors[pix_id] = asint(selected_id);
        // Preserve native alpha blending's radial ordering while retaining
        // view-space Z separately for reprojection, picking, and mesh depth.
        pixel_depth_weight[pix_id] = nearest_metric;
#else
        n_contributors[pix_id] = local_n_contrib;
#endif
        final_pixel_state[pix_id] = curr_pixel_state;
        if (uniforms.expected_far > 0.0f) {
            // Keep the expected-depth numerator and denominator raw across
            // depth waves. A single post-loop finalize divides them once.
            // The denominator is accumulated alongside the numerator (not
            // derived from final transmittance), including the saturating-splat
            // break path where curr_pixel_state is not updated before exit.
            final_pixel_depth[pix_id] = weighted_depth;
            pixel_depth_weight[pix_id] = weighted_alpha;
        } else {
            final_pixel_depth[pix_id] = median_depth;
        }
#endif
    }
}

#endif // #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD


// backward
#if EXPORT_MODE != EXPORT_MODE_VULKAN_FORWARD

#define BACKWARD_MODE_PER_PIXEL 0
#define BACKWARD_MODE_PER_SPLAT 1
#define BACKWARD_MODE_TENSOR 2

#ifndef BACKWARD_MODE
    // #define BACKWARD_MODE BACKWARD_MODE_TENSOR
    #define BACKWARD_MODE BACKWARD_MODE_PER_SPLAT
    // #define BACKWARD_MODE BACKWARD_MODE_PER_PIXEL
#endif // #ifndef BACKWARD_MODE

#if BACKWARD_MODE == BACKWARD_MODE_PER_PIXEL
    #include "alphablend_shader_bwd_per_pixel.slang"
#elif BACKWARD_MODE == BACKWARD_MODE_PER_SPLAT
    #include "alphablend_shader_bwd_per_splat.slang"
#elif BACKWARD_MODE == BACKWARD_MODE_TENSOR
    #include "alphablend_shader_bwd_tensor.slang"
#endif

#endif
