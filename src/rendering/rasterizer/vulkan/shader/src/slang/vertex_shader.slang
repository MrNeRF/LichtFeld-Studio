#include "config.slang"

import utils;
import spherical_harmonics;

#ifndef LFS_3DGUT_PROJECTION
#define LFS_3DGUT_PROJECTION 0
#endif

// Survivor-list projection (HiGS viewer path): dispatched indirectly over the
// compact survivor list emitted by cull_splats.slang instead of all N splats.
// Outputs are written at wave-appended compact slots, so every per-splat
// output buffer is sized by visible capacity, the depth-sort input is appended
// directly (no visible-flag/prefix/compact passes, no sentinel pre-fill), and
// the sorted ids ARE slots.
#ifndef LFS_PROJECTION_SURVIVORS
#define LFS_PROJECTION_SURVIVORS 0
#endif

// Canonical quantized LOD pool inputs (lod_pool_quant.hpp): xyz stays f32;
// sh0/rotation/log-scale are packed f16, opacity f16 two-per-uint, shN s8
// per swizzled-slot component with per-page band scales from page_frames.
#ifndef LFS_QUANT_POOL
#define LFS_QUANT_POOL 0
#endif


struct Gaussian_3D : IDifferentiable
{
    float3 xyz_ws;
    SpherHarmCoeffs sh_coeffs;
    float4 rotations;
    float3 scales;
    float opacities;
};

Gaussian_3D load_gaussian(
    int32_t g_idx,
    StructuredBuffer<float> xyz_ws,
    StructuredBuffer<float4> sh_coeffs,
    StructuredBuffer<float4> rotations,
    StructuredBuffer<float4> scales_opacs,
    uint active_sh
) {
    float3 g_xyz_ws = read_t3_float3(g_idx, xyz_ws);

    SpherHarmCoeffs g_sh_coeffs = read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh);

    float4 g_rotations = rotations[g_idx];
    float4 g_scales_opacs = scales_opacs[g_idx];
    float3 g_scales = g_scales_opacs.xyz;
    float g_opacities = g_scales_opacs.w;

    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales, g_opacities };
}

Gaussian_3D load_gaussian_split(
    int32_t g_idx,
    StructuredBuffer<float> xyz_ws,
    StructuredBuffer<float> sh0,
    StructuredBuffer<float4> shN,
    StructuredBuffer<float4> rotations,
    StructuredBuffer<float> scaling_raw,
    StructuredBuffer<float> opacity_raw,
    uint lod_flags,
    uint active_sh,
    uint shN_layout_slots
) {
    float3 g_xyz_ws = read_t3_float3(g_idx, xyz_ws);
    SpherHarmCoeffs g_sh_coeffs = read_spherical_harmonics_coeffs_split(
        g_idx, sh0, shN, active_sh, shN_layout_slots);
    float4 g_rotations = rotations[g_idx];
    // Matches CUDA reference: clamp raw log-scale before activation so a
    // single broken splat cannot blow up into a screen-spanning haze floater
    // (exp(50) would otherwise overflow fp32 and the splat would dominate
    // every tile it touches).
    float3 g_raw_scaling = read_t3_float3(g_idx, scaling_raw);
    g_raw_scaling = min(g_raw_scaling, float3(MAX_RAW_SCALE));
    float3 g_scales = exp(g_raw_scaling);
    const bool lod_opacity_encoded = (lod_flags & 4u) != 0u;
    float g_opacities = lod_opacity_encoded
        ? max(opacity_raw[g_idx], 0.0f)
        : sigmoid_activation(opacity_raw[g_idx]);
    if (lod_opacity_encoded && g_opacities > 1.0f) {
        // Spark stores high-opacity LOD nodes in the compact 1..2 range, then
        // maps them back to the 1..5 LOD kernel range in the shader.
        g_opacities = min(g_opacities * 4.0f - 3.0f, 5.0f);
    }

    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales, g_opacities };
}

Gaussian_3D load_gaussian_split_quant(
    int32_t g_idx,
    StructuredBuffer<float> xyz_ws,
    StructuredBuffer<uint2> sh0_q,
    StructuredBuffer<uint> shN_q,
    StructuredBuffer<uint2> rotations_q,
    StructuredBuffer<uint2> scaling_q,
    StructuredBuffer<uint> opacity_q,
    StructuredBuffer<float4> page_frames,
    uint page_splats,
    uint lod_flags,
    uint active_sh,
    uint shN_layout_slots
) {
    float3 g_xyz_ws = read_t3_float3(g_idx, xyz_ws);

    const uint page = page_splats > 0u ? uint(g_idx) / page_splats : 0u;
    // PageFrame float4[0] = per-band SH |max| scales.
    const float4 frame0 = page_frames[page * 4u];
    SpherHarmCoeffs g_sh_coeffs = read_spherical_harmonics_coeffs_split_quant(
        g_idx, sh0_q, shN_q, frame0.xyz, active_sh, shN_layout_slots);

    // Same component order as the fp32 pool: (w, x, y, z) per splat.
    const uint2 rq = rotations_q[g_idx];
    float4 g_rotations = float4(f16tof32(rq.x & 0xFFFFu), f16tof32(rq.x >> 16),
                                f16tof32(rq.y & 0xFFFFu), f16tof32(rq.y >> 16));

    const uint2 sq = scaling_q[g_idx];
    float3 g_raw_scaling = float3(f16tof32(sq.x & 0xFFFFu), f16tof32(sq.x >> 16),
                                  f16tof32(sq.y & 0xFFFFu));
    g_raw_scaling = min(g_raw_scaling, float3(MAX_RAW_SCALE));
    float3 g_scales = exp(g_raw_scaling);

    const uint opacity_word = opacity_q[uint(g_idx) >> 1];
    const float opacity_raw_value =
        f16tof32((uint(g_idx) & 1u) != 0u ? (opacity_word >> 16) : (opacity_word & 0xFFFFu));
    const bool lod_opacity_encoded = (lod_flags & 4u) != 0u;
    float g_opacities = lod_opacity_encoded
        ? max(opacity_raw_value, 0.0f)
        : sigmoid_activation(opacity_raw_value);
    if (lod_opacity_encoded && g_opacities > 1.0f) {
        g_opacities = min(g_opacities * 4.0f - 3.0f, 5.0f);
    }

    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales, g_opacities };
}

[ForceInline]
Gaussian_3D load_zero_gaussian() {
    float3 g_xyz_ws = float3(0);
    float4 g_rotations = float4(1, 0, 0, 0);
    float3 g_scales = float3(1, 1, 1);
    float g_opacities = 0.0;
    float3 z = float3(0);
    SpherHarmCoeffs g_sh_coeffs = { z, z,z,z, z,z,z,z,z, z,z,z,z,z,z,z };
    return { g_xyz_ws, g_sh_coeffs, g_rotations, g_scales, g_opacities };
}

[ForceInline]
float3 lod_debug_color(uint lod_level) {
    const uint palette_index = lod_level % 5u;
    if (palette_index == 0u) return float3(1.0, 0.0, 0.0);
    if (palette_index == 1u) return float3(0.0, 1.0, 0.0);
    if (palette_index == 2u) return float3(0.0, 0.0, 1.0);
    if (palette_index == 3u) return float3(1.0, 1.0, 0.0);
    return float3(1.0, 0.0, 1.0);
}

// Crop/ellipsoid/view-box/node culls and overlay flag classification, shared
// between the position-only cull prepass (cull_splats.slang) and the survivor
// projection. Single source of truth: the cull pass rejects a splat only if
// this exact logic would reject it in projection too.
struct SplatActiveState {
    bool active;
    uint overlay_flags;
    float3 mean_ws;
    uint model_transform_base;
    bool model_transform_enabled;
};
[ForceInline]
void apply_crop_box_filter(
    StructuredBuffer<float4> overlay_params,
    uint flags_idx,
    bool node_data_enabled,
    int node_idx,
    float3 mean_ws,
    inout bool active,
    inout uint overlay_flags)
{
    if (!active) return;
    float4 crop_flags = overlay_params[flags_idx];
    if (!overlay_enabled(crop_flags.x)) return;

    int parent_node_idx = overlay_int(crop_flags.w);
    bool applies = parent_node_idx < 0 || (node_data_enabled && node_idx == parent_node_idx);
    if (!applies) return;

    float3 local = transform_point_rows(overlay_params, flags_idx + 3u, mean_ws);
    float3 bmin = overlay_params[flags_idx + 1u].xyz;
    float3 bmax = overlay_params[flags_idx + 2u].xyz;
    bool inside = local.x >= bmin.x && local.x <= bmax.x &&
                  local.y >= bmin.y && local.y <= bmax.y &&
                  local.z >= bmin.z && local.z <= bmax.z;
    bool outside = inside == overlay_enabled(crop_flags.y);
    if (outside) {
        overlay_flags |= OVERLAY_FLAG_NONSELECTABLE;
        if (overlay_enabled(crop_flags.z)) {
            overlay_flags |= OVERLAY_FLAG_DIM;
        } else {
            active = false;
        }
    }
}

[ForceInline]
void apply_ellipsoid_filter(
    StructuredBuffer<float4> overlay_params,
    uint flags_idx,
    bool node_data_enabled,
    int node_idx,
    float3 mean_ws,
    inout bool active,
    inout uint overlay_flags)
{
    if (!active) return;
    float4 ellipsoid_flags = overlay_params[flags_idx];
    if (!overlay_enabled(ellipsoid_flags.x)) return;

    int parent_node_idx = overlay_int(ellipsoid_flags.w);
    bool applies = parent_node_idx < 0 || (node_data_enabled && node_idx == parent_node_idx);
    if (!applies) return;

    float3 local = transform_point_rows(overlay_params, flags_idx + 2u, mean_ws);
    float3 radii = max(abs(overlay_params[flags_idx + 1u].xyz), float3(1e-8f));
    float norm = (local.x * local.x) / (radii.x * radii.x) +
                 (local.y * local.y) / (radii.y * radii.y) +
                 (local.z * local.z) / (radii.z * radii.z);
    bool inside = norm <= 1.0f;
    bool outside = inside == overlay_enabled(ellipsoid_flags.y);
    if (outside) {
        overlay_flags |= OVERLAY_FLAG_NONSELECTABLE;
        if (overlay_enabled(ellipsoid_flags.z)) {
            overlay_flags |= OVERLAY_FLAG_DIM;
        } else {
            active = false;
        }
    }
}

SplatActiveState compute_splat_active_state(
    uint logical_idx,
    float3 xyz_ws_in,
    uint model_transform_count,
    ByteAddressBuffer transform_indices,
    ByteAddressBuffer node_mask,
    StructuredBuffer<float4> overlay_params,
    StructuredBuffer<float4> model_transforms)
{
    uint overlay_flags = 0;
    bool active = true;
    float4 emphasis_flags = overlay_params[OVERLAY_PARAM_EMPHASIS_FLAGS];
    bool node_data_enabled = overlay_enabled(emphasis_flags.y);
    int node_count = overlay_int(emphasis_flags.z);
    int node_idx = node_data_enabled ? read_int32(transform_indices, logical_idx) : -1;
    int model_transform_idx = node_data_enabled ? node_idx : 0;
    model_transform_idx = clamp(model_transform_idx, 0, int(max(model_transform_count, 1u)) - 1);
    uint model_transform_base = uint(model_transform_idx) * 4u;
    bool model_transform_enabled = model_transform_count > 0u;
    float3 mean_ws = model_transform_enabled
                         ? transform_point_rows(model_transforms, model_transform_base, xyz_ws_in)
                         : xyz_ws_in;

    apply_crop_box_filter(overlay_params, OVERLAY_PARAM_CROP_FLAGS,
                          node_data_enabled, node_idx, mean_ws,
                          active, overlay_flags);
    for (uint crop_i = 0u; crop_i < OVERLAY_PARAM_CROP_EXTRA_COUNT; ++crop_i) {
        apply_crop_box_filter(overlay_params,
                              OVERLAY_PARAM_CROP_EXTRA_BASE + crop_i * OVERLAY_PARAM_CROP_STRIDE,
                              node_data_enabled, node_idx, mean_ws,
                              active, overlay_flags);
    }

    apply_ellipsoid_filter(overlay_params, OVERLAY_PARAM_ELLIPSOID_FLAGS,
                           node_data_enabled, node_idx, mean_ws,
                           active, overlay_flags);
    for (uint ellipsoid_i = 0u; ellipsoid_i < OVERLAY_PARAM_ELLIPSOID_EXTRA_COUNT; ++ellipsoid_i) {
        apply_ellipsoid_filter(overlay_params,
                               OVERLAY_PARAM_ELLIPSOID_EXTRA_BASE + ellipsoid_i * OVERLAY_PARAM_ELLIPSOID_STRIDE,
                               node_data_enabled, node_idx, mean_ws,
                               active, overlay_flags);
    }

    float4 view_flags = overlay_params[OVERLAY_PARAM_VIEW_FLAGS];
    if (active && overlay_enabled(view_flags.x)) {
        float3 local = transform_point_rows(overlay_params, OVERLAY_PARAM_VIEW_TRANSFORM, mean_ws);
        float3 vmin = overlay_params[OVERLAY_PARAM_VIEW_MIN].xyz;
        float3 vmax = overlay_params[OVERLAY_PARAM_VIEW_MAX].xyz;
        bool inside = local.x >= vmin.x && local.x <= vmax.x &&
                      local.y >= vmin.y && local.y <= vmax.y &&
                      local.z >= vmin.z && local.z <= vmax.z;
        if (!inside) {
            overlay_flags |= OVERLAY_FLAG_NONSELECTABLE;
            if (overlay_enabled(view_flags.y)) {
                active = false;
            }
        }
    }

    if (active && overlay_enabled(emphasis_flags.x) && node_data_enabled && node_idx >= 0 && node_idx < node_count) {
        if (read_byte(node_mask, uint(node_idx)) == 0) {
            overlay_flags |= (OVERLAY_FLAG_DIM | OVERLAY_FLAG_NONSELECTABLE);
        }
    }

    if (active && emphasis_flags.w > 0.0f && node_data_enabled && node_idx >= 0 && node_idx < node_count) {
        if (read_byte(node_mask, uint(node_idx)) != 0) {
            overlay_flags |= OVERLAY_FLAG_FLASH;
        }
    }

    float4 visibility_flags = overlay_params[OVERLAY_PARAM_VISIBILITY_FLAGS];
    if (active && overlay_enabled(visibility_flags.x) && node_data_enabled && node_idx >= 0 && node_idx < node_count) {
        if (read_byte(node_mask, uint(node_idx)) == 0) {
            active = false;
        }
    }

    SplatActiveState state;
    state.active = active;
    state.overlay_flags = overlay_flags;
    state.mean_ws = mean_ws;
    state.model_transform_base = model_transform_base;
    state.model_transform_enabled = model_transform_enabled;
    return state;
}

// Position-level visibility tests, exact mirrors of the center rejects inside
// project_gaussian_to_camera_model / _gut. The cull prepass must never reject
// a splat the projection would keep — these are the same expressions, so the
// survivor set is a strict superset of the finally-visible set.
[ForceInline]
bool splat_center_potentially_visible(float3 mean_ws, Camera cam) {
    float3 xyz_vs = project_point(mean_ws, cam);
    return xyz_vs.z > NEAR_CLIP &&
           isfinite(dot(xyz_vs, xyz_vs)) &&
           image_point_in_image_bounds_margin(xyz_vs.xy, cam, 0.2f);
}

[ForceInline]
bool splat_center_potentially_visible_gut(float3 mean_ws, Camera cam) {
    float3 mean_vs = project_point(mean_ws, cam);
    return mean_vs.z > NEAR_CLIP && isfinite(dot(mean_vs, mean_vs));
}

[Differentiable]
Splat_2D_Vertex project_gaussian_to_camera(Gaussian_3D g, Camera cam, uint active_sh) {
    float3 xyz_vs = project_point(g.xyz_ws, cam);
    if (xyz_vs.z <= NEAR_CLIP) {
        return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
    }
    xyz_vs.xy -= 0.5;  // pixel offset

    float4x3 mat = transpose(cam.world_view_transform);
    float3 cam_pos = -mul(float3x3(mat), mat[3]);
    float3 dir = float3(normalize(g.xyz_ws - cam_pos));
    float3 rgb = compute_color_from_sh_coeffs(g.sh_coeffs, dir, active_sh);

    float3x3 cov_ws = get_covariance_from_quat_scales(
        safe_quat_normalize(g.rotations), g.scales
    );
    float4 cov_vs = covariance_3d_to_2d(cam, g.xyz_ws, cov_ws, false, false);

    float det = compute_det(cov_vs.xyz);
    float3 inv_cov_vs = compute_inverse(cov_vs.xyz);
    float opac = g.opacities * cov_vs.w;

    return { xyz_vs, rgb, cov_vs.xyz, float4(inv_cov_vs, opac), det };
}

Splat_2D_Vertex project_gaussian_to_camera_model(
    Gaussian_3D g,
    Camera cam,
    uint active_sh,
    float3 mean_ws,
    StructuredBuffer<float4> model_transform_rows,
    uint model_transform_base,
    bool model_transform_enabled,
    bool mip_filter,
    bool spark_rad_opacity_path) {
    float3 xyz_vs = project_point(mean_ws, cam);
    // Reject behind-camera, non-finite, and far off-screen means. The EWA
    // projection Jacobian degenerates as the mean moves outside the frustum
    // and produces needle-like splats that streak across the visible tiles.
    // CUDA reference uses clip_xy=1.4 (20% margin).
    if (xyz_vs.z <= NEAR_CLIP ||
        !isfinite(dot(xyz_vs, xyz_vs)) ||
        !image_point_in_image_bounds_margin(xyz_vs.xy, cam, 0.2f)) {
        return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
    }
    xyz_vs.xy -= 0.5;  // pixel offset

    float3 dir = view_direction_to_point(mean_ws, cam);
    if (model_transform_enabled) {
        float3 r0, r1, r2;
        if (extract_rotation_rows(model_transform_rows, model_transform_base, r0, r1, r2)) {
            dir = transpose_mul_rows(r0, r1, r2, dir);
        }
    }
    float3 rgb = compute_color_from_sh_coeffs(g.sh_coeffs, dir, active_sh);

    float3x3 cov_ws = get_covariance_from_quat_scales(
        safe_quat_normalize(g.rotations), g.scales
    );
    if (model_transform_enabled) {
        cov_ws = transform_covariance_rows(model_transform_rows, model_transform_base, cov_ws);
    }
    float4 cov_vs = covariance_3d_to_2d(cam, mean_ws, cov_ws, mip_filter, spark_rad_opacity_path);
    float opac = g.opacities * cov_vs.w;
    float3 cov_projected = clamp_covariance_2d_extent(cov_vs.xyz, opac);
    float det = compute_det(cov_projected);
    float3 inv_cov_vs = compute_inverse(cov_projected);

    return { xyz_vs, rgb, cov_projected, float4(inv_cov_vs, opac), det };
}

[ForceInline]
float3 rotation_column(float3x3 rotation, uint column) {
    if (column == 0u)
        return float3(rotation[0][0], rotation[1][0], rotation[2][0]);
    if (column == 1u)
        return float3(rotation[0][1], rotation[1][1], rotation[2][1]);
    return float3(rotation[0][2], rotation[1][2], rotation[2][2]);
}

[ForceInline]
bool valid_gut_image_point(float3 projected, Camera cam) {
    return projected.z > 0.0f &&
           isfinite(dot(projected, projected)) &&
           image_point_in_image_bounds_margin(projected.xy, cam, 0.1f);
}

Splat_2D_Vertex project_gaussian_to_camera_gut(
    Gaussian_3D g,
    Camera cam,
    uint active_sh,
    float3 mean_ws,
    StructuredBuffer<float4> model_transform_rows,
    uint model_transform_base,
    bool model_transform_enabled,
    bool mip_filter) {
    float3 mean_vs = project_point(mean_ws, cam);
    if (mean_vs.z <= NEAR_CLIP || !isfinite(dot(mean_vs, mean_vs))) {
        return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
    }

    static const uint D = 3u;
    static const uint NUM_SIGMA_POINTS = 7u;
    static const float UT_ALPHA = 0.1f;
    static const float UT_BETA = 2.0f;
    static const float UT_KAPPA = 0.0f;
    float lambda = UT_ALPHA * UT_ALPHA * (float(D) + UT_KAPPA) - float(D);
    float sigma_scale = sqrt(float(D) + lambda);
    float weight_mean_0 = lambda / (float(D) + lambda);
    float weight_cov_0 = weight_mean_0 + (1.0f - UT_ALPHA * UT_ALPHA + UT_BETA);
    float weight_sigma = 1.0f / (2.0f * (float(D) + lambda));

    // 3DGUT uses the unscented transform for screen-space binning; final alpha
    // is evaluated against the pixel ray in the raster shader.
    float3x3 rotation = quat_to_rotmat(safe_quat_normalize(g.rotations));
    float3 sigma_points[NUM_SIGMA_POINTS];
    sigma_points[0] = mean_ws;
    for (uint i = 0u; i < D; ++i) {
        float scale_i = i == 0u ? g.scales.x : (i == 1u ? g.scales.y : g.scales.z);
        float3 delta = sigma_scale * scale_i * rotation_column(rotation, i);
        float3 source_plus = g.xyz_ws + delta;
        float3 source_minus = g.xyz_ws - delta;
        sigma_points[1u + i] = model_transform_enabled
                                    ? transform_point_rows(model_transform_rows, model_transform_base, source_plus)
                                    : source_plus;
        sigma_points[1u + D + i] = model_transform_enabled
                                        ? transform_point_rows(model_transform_rows, model_transform_base, source_minus)
                                        : source_minus;
    }

    float2 image_points[NUM_SIGMA_POINTS];
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float3 projected = project_point(sigma_points[i], cam);
        if (!valid_gut_image_point(projected, cam)) {
            return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
        }
        image_points[i] = projected.xy;
    }

    unwrap_image_points_x(image_points, cam);

    float2 mean2d = float2(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float weight = i == 0u ? weight_mean_0 : weight_sigma;
        mean2d += weight * image_points[i];
    }
    float2 covariance_mean2d = mean2d;
    if (is_equirectangular_camera(cam)) {
        mean2d.x = wrap_image_x(mean2d.x, cam);
    }

    float3 cov2d = float3(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float2 d = image_points[i] - covariance_mean2d;
        float weight = i == 0u ? weight_cov_0 : weight_sigma;
        cov2d += weight * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    }

    float4 cov_vs = add_blur_to_covariance2d(cov2d, mip_filter, false);
    float det = compute_det(cov_vs.xyz);
    if (!isfinite(det) || det <= 0.0f) {
        return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
    }

    float3 dir = view_direction_to_point(mean_ws, cam);
    if (model_transform_enabled) {
        float3 r0, r1, r2;
        if (extract_rotation_rows(model_transform_rows, model_transform_base, r0, r1, r2)) {
            dir = transpose_mul_rows(r0, r1, r2, dir);
        }
    }
    float3 rgb = compute_color_from_sh_coeffs(g.sh_coeffs, dir, active_sh);
    float3 inv_cov_vs = compute_inverse(cov_vs.xyz);
    float opac = g.opacities * cov_vs.w;
    float3 xyz_vs = float3(mean2d - 0.5f, mean_vs.z);

    return { xyz_vs, rgb, cov_vs.xyz, float4(inv_cov_vs, opac), det };
}


rectangle get_rectangle_tile_space(
    float2 ndc_xy,
    float2 radii,
    uint grid_height,
    uint grid_width) {

    rectangle rect_tile_space;

    rect_tile_space.min_x = uint16_t(floor(clamp((ndc_xy.x - radii.x) / TILE_WIDTH, 0, grid_width)));
    rect_tile_space.min_y = uint16_t(floor(clamp((ndc_xy.y - radii.y) / TILE_HEIGHT, 0, grid_height)));
    rect_tile_space.max_x = uint16_t(ceil(clamp((ndc_xy.x + radii.x) / TILE_WIDTH, 0, grid_width)));
    rect_tile_space.max_y = uint16_t(ceil(clamp((ndc_xy.y + radii.y) / TILE_HEIGHT, 0, grid_height)));

    return rect_tile_space;
}

rectangle get_wrapped_rectangle_tile_space(
    float2 ndc_xy,
    float2 radii,
    uint grid_height,
    uint grid_width) {

    rectangle rect_tile_space;

    rect_tile_space.min_x = 0;
    rect_tile_space.min_y = uint16_t(floor(clamp((ndc_xy.y - radii.y) / TILE_HEIGHT, 0, grid_height)));
    rect_tile_space.max_x = 0;
    rect_tile_space.max_y = uint16_t(ceil(clamp((ndc_xy.y + radii.y) / TILE_HEIGHT, 0, grid_height)));

    if (grid_width == 0u || rect_tile_space.min_y >= rect_tile_space.max_y)
        return rect_tile_space;

    int tile_width_i = int(grid_width);
    int x0 = int(floor((ndc_xy.x - radii.x) / TILE_WIDTH));
    int x1 = int(ceil((ndc_xy.x + radii.x) / TILE_WIDTH));
    int span_x = x1 - x0;
    if (span_x <= 0)
        return rect_tile_space;

    if (span_x >= tile_width_i) {
        rect_tile_space.min_x = 0;
        rect_tile_space.max_x = uint16_t(grid_width);
        return rect_tile_space;
    }

    int start = x0 % tile_width_i;
    if (start < 0)
        start += tile_width_i;

    rect_tile_space.min_x = uint16_t(start);
    rect_tile_space.max_x = uint16_t(start + span_x);

    return rect_tile_space;
}


// #define EARLY_RETURN_ON_ZERO_GRAD 0
#define EARLY_RETURN_ON_ZERO_GRAD 0


#ifndef NO_MAIN

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

// Input buffers
layout(binding=0) StructuredBuffer<float> xyz_ws;  // 3
#if LFS_QUANT_POOL
layout(binding=1) StructuredBuffer<uint2> sh0;  // f16x3 + pad
layout(binding=2) StructuredBuffer<uint> shN;  // s8x4 per swizzled slot
layout(binding=3) StructuredBuffer<uint2> rotations;  // f16x4 (w,x,y,z)
layout(binding=4) StructuredBuffer<uint2> scaling_raw;  // log f16x3 + pad
layout(binding=5) StructuredBuffer<uint> opacity_raw;  // f16, 2 splats per uint
#else
layout(binding=1) StructuredBuffer<float> sh0;  // 1*3
layout(binding=2) StructuredBuffer<float4> shN;  // rest-only 15*3
layout(binding=3) StructuredBuffer<float4> rotations;  // 4
layout(binding=4) StructuredBuffer<float> scaling_raw;  // 3
layout(binding=5) StructuredBuffer<float> opacity_raw;  // 1
#endif

// Output buffers
#if !LFS_PROJECTION_SURVIVORS
// The macro chain derives coverage from the rect + ellipse; tiles_touched and
// radii are legacy-chain outputs only and their bindings are absent in the
// survivor variant.
layout(binding=6) RWStructuredBuffer<int> out_tiles_touched;
#endif
#if USE_EMULATED_INT64
layout(binding=7) RWStructuredBuffer<int32_t> out_rect_tile_space;
#else
layout(binding=7) RWStructuredBuffer<int64_t> out_rect_tile_space;
#endif
#if !LFS_PROJECTION_SURVIVORS
layout(binding=8) RWStructuredBuffer<int> out_radii;
#endif
layout(binding=9) RWStructuredBuffer<float2> out_xy_vs;
layout(binding=10) RWStructuredBuffer<float> out_depths;
layout(binding=11) RWStructuredBuffer<float4> out_inv_cov_vs;
layout(binding=12) RWStructuredBuffer<float> out_rgb;
layout(binding=13) RWStructuredBuffer<uint> out_overlay_flags;
layout(binding=14) ByteAddressBuffer transform_indices;
layout(binding=15) ByteAddressBuffer node_mask;
layout(binding=16) StructuredBuffer<float4> overlay_params;
layout(binding=17) StructuredBuffer<float4> model_transforms;
#if LFS_PROJECTION_SURVIVORS
// Depth-sort input, appended directly at the compact slot. No sentinel
// pre-fill: only [0, emit_count) is ever sorted.
layout(binding=18) RWStructuredBuffer<uint32_t> out_sort_keys;
#else
// Two-stage sort: per-primitive depth key written only at the success path.
// CPU pre-fills with 0xFFFFFFFFu so invisible primitives sort to the tail.
layout(binding=18) RWStructuredBuffer<uint32_t> out_primitive_depth_keys;
#endif
// LOD index indirection buffer
layout(binding=19) StructuredBuffer<uint32_t> lod_indices;
layout(binding=20) StructuredBuffer<uint32_t> lod_logical_indices;
layout(binding=21) StructuredBuffer<uint32_t> lod_levels;
layout(binding=22) StructuredBuffer<float> lod_weights;
// GPU LoD selector append count; [0] = appended total (may exceed capacity).
layout(binding=23) StructuredBuffer<uint32_t> lod_counts;
#if LFS_PROJECTION_SURVIVORS
// Compact survivor list from the cull prepass; entries are render indices.
layout(binding=24) StructuredBuffer<int32_t> survivors;
// [0] = survivor count (written by the cull pass).
layout(binding=25) StructuredBuffer<uint32_t> survivor_count;
// Depth-sort ids; out_sort_ids[slot] = slot, so the sorted ids are slots.
layout(binding=26) RWStructuredBuffer<int32_t> out_sort_ids;
// [0] = total emitted (may exceed capacity; clamp detection reads this).
layout(binding=27) RWStructuredBuffer<uint32_t> out_emit_count;
// Model-splat index per slot, for bit-per-model-splat selection masks.
layout(binding=28) RWStructuredBuffer<int32_t> out_orig_ids;
#endif
#if LFS_QUANT_POOL
// Per-page dequant frames (lod_pool_quant.hpp PageFrame), bound after every
// variant-specific binding so the C++ buffer list just appends it last.
#if LFS_PROJECTION_SURVIVORS
layout(binding=29) StructuredBuffer<float4> page_frames;
#else
layout(binding=24) StructuredBuffer<float4> page_frames;
#endif
#endif

#if !LFS_PROJECTION_SURVIVORS
// Legacy N-wide variant only: the survivor-list variant never clears, because
// rejected splats simply do not append a compact slot.
[ForceInline]
void clear_forward_projection_output(uint out_idx, uint overlay_flags) {
    out_overlay_flags[out_idx] = overlay_flags;
    out_radii[out_idx] = 0;
    out_tiles_touched[out_idx] = 0;
  #if USE_EMULATED_INT64
    out_rect_tile_space[2*out_idx+0] = 0;
    out_rect_tile_space[2*out_idx+1] = 0;
  #else
    out_rect_tile_space[out_idx] = int64_t(0);
  #endif
    out_xy_vs[out_idx] = float2(0.0);
    out_depths[out_idx] = 0.0;
    out_inv_cov_vs[out_idx] = float4(0.0);
    write_t3_float3(float3(0.0), out_idx, out_rgb);
}
#endif


#elif EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD

// Input buffers
layout(binding=0) StructuredBuffer<float> xyz_ws;  // 3
layout(binding=1) StructuredBuffer<float4> sh_coeffs;  // 16*3
layout(binding=2) StructuredBuffer<float4> rotations;  // 4
layout(binding=3) StructuredBuffer<float4> scales_opacs;  // 4

layout(binding=4) StructuredBuffer<int> out_tiles_touched;

layout(binding=5) StructuredBuffer<float2> v_output_xy_vs;
// layout(binding=6) StructuredBuffer<float> v_output_depths;
layout(binding=7) StructuredBuffer<float4> v_output_inv_cov_vs;
layout(binding=8) StructuredBuffer<float> v_output_rgb;

// Output buffers
layout(binding=9) RWStructuredBuffer<float> v_xyz_ws;  // 3
layout(binding=10) RWStructuredBuffer<float4> v_sh_coeffs;  // 16*3
layout(binding=11) RWStructuredBuffer<float4> v_rotations;  // 4
layout(binding=12) RWStructuredBuffer<float4> v_scales_opacs;  // 4

[ForceInline]
void write_v_xyz_ws(uint32_t g_idx, float3 value) {
    write_t3_float3(value, g_idx, v_xyz_ws);
}

#endif  // EXPORT_MODE == EXPORT_MODE_VULKAN_XWARD

#define no_diff

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint active_sh = uniforms.active_sh;
    const uint model_transform_count = uniforms.step;
    const uint image_height = uniforms.image_height;
    const uint image_width = uniforms.image_width;
    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;
    const uint camera_height = uniforms.camera_height == 0 ? image_height : uniforms.camera_height;
    const uint camera_width = uniforms.camera_width == 0 ? image_width : uniforms.camera_width;
    const uint camera_model = base_camera_model(uniforms.camera_model);
    const float4x4 world_view_transform = (uniforms.world_view_transform);

#if LFS_PROJECTION_SURVIVORS
    // Indirect dispatch over the survivor list. The cull pass already applied
    // the LoD exact-count cut and rejected invalid LoD entries; both checks
    // below are kept as cheap defensive guards only.
    const uint survivor_idx = dispatchThreadID.x;
    if (survivor_idx >= survivor_count[0])
        return;
    uint g_idx = uint(survivors[survivor_idx]);
    uint render_idx = g_idx;
#else
    uint g_idx = dispatchThreadID.x;
    uint render_idx = g_idx;
    const uint out_idx = render_idx;

    if (g_idx >= uniforms.num_splats)
        return;
#endif

    // LOD indirection
    const bool lod_active = (uniforms.lod_enabled & 1u) != 0u;
    uint logical_idx = g_idx;
    if (lod_active) {
      #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD && !LFS_PROJECTION_SURVIVORS
        if ((uniforms.lod_enabled & 32u) != 0u) {
            const uint valid = min(lod_counts[0], uniforms.lod_count);
            if (render_idx >= valid) {
                clear_forward_projection_output(out_idx, 0u);
                return;
            }
        }
      #endif
        g_idx = lod_indices[render_idx];
        if (g_idx == 0xffffffffu || g_idx >= uniforms.model_num_splats) {
          #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD && !LFS_PROJECTION_SURVIVORS
            clear_forward_projection_output(out_idx, 0u);
          #endif
            return;
        }
        logical_idx = (uniforms.lod_enabled & 8u) != 0u
            ? lod_logical_indices[render_idx]
            : g_idx;
    }

  #if EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD
    // no overlap
    bool is_zero_grad = (out_tiles_touched[g_idx] == 0);
  #if EARLY_RETURN_ON_ZERO_GRAD
    // set gradient to zero and early return
    if (is_zero_grad) {
        write_v_xyz_ws(g_idx, float3(0.0));
        #if SH_REORDER_SIZE <= 1
          for (int i = 0; i < 12; i++)
              v_sh_coeffs[12*g_idx+i] = float4(0.0);
        #else
          uint32_t base = (g_idx/SH_REORDER_SIZE) * (12*SH_REORDER_SIZE);
          uint32_t offset = g_idx%SH_REORDER_SIZE;
          for (int i = 0; i < 12; i++)
              v_sh_coeffs[base+SH_REORDER_SIZE*i+offset] = float4(0.0);
        #endif  // SH_REORDER_SIZE
        v_rotations[g_idx] = float4(0.0);
        v_scales_opacs[g_idx] = float4(0.0);
        return;
    }
  #endif  // EARLY_RETURN_ON_ZERO_GRAD
  #endif  // EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD

    // load stuff
    Camera cam = no_diff load_camera(world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy,
        camera_model, uniforms.dist_coeffs,
        camera_height, camera_width);
  #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD
    #if LFS_QUANT_POOL
    Gaussian_3D gauss = load_gaussian_split_quant(
        g_idx, xyz_ws, sh0, shN, rotations,
        scaling_raw, opacity_raw,
        page_frames,
        uniforms.lod_page_splats,
        uniforms.lod_enabled,
        active_sh,
        uniforms.shN_layout_slots
      );
    #else
    Gaussian_3D gauss = load_gaussian_split(
        g_idx, xyz_ws, sh0, shN, rotations,
        scaling_raw, opacity_raw,
        uniforms.lod_enabled,
        active_sh,
        uniforms.shN_layout_slots
      );
    #endif
    if (lod_active && (uniforms.lod_enabled & 16u) != 0u) {
        gauss.opacities *= clamp(lod_weights[render_idx], 0.0f, 1.0f);
    }
  #else
    #if EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD && !EARLY_RETURN_ON_ZERO_GRAD
    Gaussian_3D gauss = load_zero_gaussian();
    if (!is_zero_grad) gauss = load_gaussian(
    #else
    Gaussian_3D gauss = load_gaussian(
    #endif
        g_idx, xyz_ws, sh_coeffs, rotations,
        scales_opacs,
        active_sh
      );
  #endif

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD
    const SplatActiveState active_state = compute_splat_active_state(
        logical_idx, gauss.xyz_ws, model_transform_count,
        transform_indices, node_mask, overlay_params, model_transforms);
    uint overlay_flags = active_state.overlay_flags;
    uint model_transform_base = active_state.model_transform_base;
    bool model_transform_enabled = active_state.model_transform_enabled;
    float3 mean_ws = active_state.mean_ws;

    if (!active_state.active) {
      #if LFS_PROJECTION_SURVIVORS
        return;
      #else
        clear_forward_projection_output(out_idx, overlay_flags);
        return;
      #endif
    }
#endif

    const bool spark_rad_opacity_path = (uniforms.lod_enabled & 4u) != 0u;

    // projection (differentiable)
  #if EXPORT_MODE != EXPORT_MODE_VULKAN_BACKWARD
    // forward
  #if LFS_3DGUT_PROJECTION
    Splat_2D_Vertex splat = project_gaussian_to_camera_gut(
        gauss,
        cam,
        active_sh,
        mean_ws,
        model_transforms,
        model_transform_base,
        model_transform_enabled,
        (uniforms.mip_filter & 1u) != 0u);
  #else
    Splat_2D_Vertex splat = project_gaussian_to_camera_model(
        gauss,
        cam,
        active_sh,
        mean_ws,
        model_transforms,
        model_transform_base,
        model_transform_enabled,
        (uniforms.mip_filter & 1u) != 0u,
        spark_rad_opacity_path);
  #endif
  #else
    // backward
    float2 v_out_xy_vs = is_zero_grad ? float2(0) : v_output_xy_vs[g_idx];
    float v_out_depths = 0.0;
    float4 v_out_inv_cov_vs_opac = is_zero_grad ? float4(0) :
      v_output_inv_cov_vs[g_idx];
    float3 v_out_rgb = is_zero_grad ? float3(0) :
      float3(read_t3_float3(g_idx, v_output_rgb));

    Splat_2D_Vertex v_splat = {
        float3(v_out_xy_vs, v_out_depths),  // xyz_vs
        v_out_rgb,  // rgb
        float3(0.0),  // cov_vs
        v_out_inv_cov_vs_opac,  // inv_cov_vs
        0.0  // det
    };
    DifferentialPair<Gaussian_3D> v_gauss = diffPair(gauss);
    DifferentialPair<Camera> v_cam = diffPair(cam);
    bwd_diff(project_gaussian_to_camera)(v_gauss, v_cam, active_sh, v_splat);
  #endif

    // radii, bounds, number of tiles
#if EXPORT_MODE != EXPORT_MODE_VULKAN_BACKWARD
    splat.xyz_vs.xy -= float2(uniforms.render_origin_x, uniforms.render_origin_y);

    if (splat.xyz_vs.z <= NEAR_CLIP ||
        splat.inv_cov_vs_opac.w <= ALPHA_THRESHOLD ||
        splat.det <= 0.0
    ) {
        #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD && !LFS_PROJECTION_SURVIVORS
            clear_forward_projection_output(out_idx, overlay_flags);
        #endif
        return;
    }

    Splat_2D_Vertex splat_dist = splat;

    float radius = splat_radius(splat_dist.cov_vs.xyz, splat_dist.det);
    bool equirectangular_camera = is_equirectangular_camera(cam);

  #if TILE_CULLING_MODE == 0
    float2 radii = float2(radius);
  #else
    float extend = no_diff extend_from_opacity(gauss.opacities);
    float2 radii = sqrt(max(splat_dist.cov_vs.xz, float2(0.0f))) * extend;
  #endif
    if (equirectangular_camera) {
        radii.x = min(radii.x, float(image_width));
        radii.y = min(radii.y, 0.49f * float(image_height));
    }

    rectangle rect_tile_space = equirectangular_camera
        ? get_wrapped_rectangle_tile_space(splat_dist.xyz_vs.xy, radii, grid_height, grid_width)
        : get_rectangle_tile_space(splat_dist.xyz_vs.xy, radii, grid_height, grid_width);

    int32_t n_tiles = int32_t(rect_tile_space.max_x - rect_tile_space.min_x) * int32_t(rect_tile_space.max_y - rect_tile_space.min_y);
    if (n_tiles == 0) {
        #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD && !LFS_PROJECTION_SURVIVORS
            clear_forward_projection_output(out_idx, overlay_flags);
        #endif
        return;
    }

  #if TILE_CULLING_MODE == 2
    if (!equirectangular_camera) {
        n_tiles = no_diff count_ellipse_grid_overlaps(
            splat_dist.xyz_vs.xy, splat_dist.inv_cov_vs_opac.xyz/(extend*extend),
            rect_tile_space
        );
    }
  #endif

#if LFS_PROJECTION_SURVIVORS
    // TILE_CULLING_MODE == 2 can refine n_tiles to zero after the bbox check.
    if (n_tiles == 0)
        return;

    // Wave-aggregated compact-slot allocation. Lanes that early-returned are
    // inactive, so the ballot counts exactly the emitting lanes. out_emit_count
    // keeps the unclamped total for overflow detection; writes are clamped to
    // capacity (uniforms.sort_capacity carries the visible capacity here).
    const uint emit_in_wave = WaveActiveCountBits(true);
    const uint lane_offset = WavePrefixCountBits(true);
    uint wave_base = 0;
    if (WaveIsFirstLane()) {
        InterlockedAdd(out_emit_count[0], emit_in_wave, wave_base);
    }
    wave_base = WaveReadLaneFirst(wave_base);
    const uint out_idx = wave_base + lane_offset;
    if (out_idx >= uniforms.sort_capacity)
        return;

  #if USE_EMULATED_INT64
    out_rect_tile_space[2*out_idx+0] = (int32_t)rect_tile_space.min_x | ((int32_t)rect_tile_space.min_y << 16);
    out_rect_tile_space[2*out_idx+1] = (int32_t)rect_tile_space.max_x | ((int32_t)rect_tile_space.max_y << 16);
  #else
    out_rect_tile_space[out_idx] = reinterpret<int64_t>(rect_tile_space);
  #endif

    // Same radial depth key as the N-wide path; ids are slots, so after the
    // depth radix sort the sorted ids index the compact outputs directly.
    float3 cam_origin = no_diff camera_origin_ws(cam);
    float3 sort_delta = mean_ws - cam_origin;
    float radial_sort_metric = dot(sort_delta, sort_delta);
    out_sort_keys[out_idx] = reinterpret<uint32_t>(radial_sort_metric);
    out_sort_ids[out_idx] = int32_t(out_idx);
    out_orig_ids[out_idx] = int32_t(logical_idx);
#else
    // out_radii[g_idx] = (uint32_t)radius;  // inria
    out_radii[out_idx] = (uint32_t)ceil(max(radii.x, radii.y));  // gsplat, only used in densification
    out_tiles_touched[out_idx] = n_tiles;
  #if USE_EMULATED_INT64
    out_rect_tile_space[2*out_idx+0] = (int32_t)rect_tile_space.min_x | ((int32_t)rect_tile_space.min_y << 16);
    out_rect_tile_space[2*out_idx+1] = (int32_t)rect_tile_space.max_x | ((int32_t)rect_tile_space.max_y << 16);
  #else
    out_rect_tile_space[out_idx] = reinterpret<int64_t>(rect_tile_space);
  #endif

    if (n_tiles == 0) {
        out_overlay_flags[out_idx] = overlay_flags;
        return;
    }

    // Two-stage sort: radial distance squared as the depth key, matching the
    // CUDA reference (`gsplat_fwd/kernels_forward.cuh: primitive_depth_keys[...]
    // = __float_as_uint(radial_sort_metric)`). Float-as-uint preserves
    // monotonic order for non-negative values, so the radix sort sees a
    // canonical front-to-back ordering. Invisible primitives keep the 0xFFFFFFFFu
    // sentinel pre-filled on the host.
    float3 cam_origin = no_diff camera_origin_ws(cam);
    float3 sort_delta = mean_ws - cam_origin;
    float radial_sort_metric = dot(sort_delta, sort_delta);
    out_primitive_depth_keys[out_idx] = reinterpret<uint32_t>(radial_sort_metric);
#endif  // LFS_PROJECTION_SURVIVORS

#endif  // EXPORT_MODE != EXPORT_MODE_VULKAN_BACKWARD

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

    if ((uniforms.lod_enabled & 2u) != 0u) {
        const float3 debug_rgb = lod_debug_color(lod_levels[render_idx]);
        splat.rgb *= debug_rgb;
    }

    out_xy_vs[out_idx] = splat.xyz_vs.xy;
    out_depths[out_idx] = splat.xyz_vs.z;
    out_inv_cov_vs[out_idx] = splat.inv_cov_vs_opac;
    write_t3_float3(splat.rgb, out_idx, out_rgb);
    out_overlay_flags[out_idx] = overlay_flags;

#elif EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD

    write_v_xyz_ws(g_idx, v_gauss.d.xyz_ws);
    write_spherical_harmonics_coeffs_grads(
      v_gauss.d.sh_coeffs, g_idx, v_sh_coeffs, active_sh, step);
    v_rotations[g_idx] = v_gauss.d.rotations;
    v_scales_opacs[g_idx] = float4(v_gauss.d.scales, v_gauss.d.opacities);

#endif
}

#endif  // NO_MAIN
