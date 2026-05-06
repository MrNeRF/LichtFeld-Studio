#include "config.slang"

import utils;
import spherical_harmonics;

#ifndef VKSPLAT_GUT_PROJECTION
#define VKSPLAT_GUT_PROJECTION 0
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
        normalize(g.rotations), g.scales
    );
    float4 cov_vs = covariance_3d_to_2d(cam, g.xyz_ws, cov_ws);

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
    bool model_transform_enabled) {
    float3 xyz_vs = project_point(mean_ws, cam);
    if (xyz_vs.z <= NEAR_CLIP) {
        return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
    }
    xyz_vs.xy -= 0.5;  // pixel offset

    float4x3 mat = transpose(cam.world_view_transform);
    float3 cam_pos = -mul(float3x3(mat), mat[3]);
    float3 dir = float3(normalize(mean_ws - cam_pos));
    if (model_transform_enabled) {
        float3 r0, r1, r2;
        if (extract_rotation_rows(model_transform_rows, model_transform_base, r0, r1, r2)) {
            dir = transpose_mul_rows(r0, r1, r2, dir);
        }
    }
    float3 rgb = compute_color_from_sh_coeffs(g.sh_coeffs, dir, active_sh);

    float3x3 cov_ws = get_covariance_from_quat_scales(
        normalize(g.rotations), g.scales
    );
    if (model_transform_enabled) {
        cov_ws = transform_covariance_rows(model_transform_rows, model_transform_base, cov_ws);
    }
    float4 cov_vs = covariance_3d_to_2d(cam, mean_ws, cov_ws);

    float det = compute_det(cov_vs.xyz);
    float3 inv_cov_vs = compute_inverse(cov_vs.xyz);
    float opac = g.opacities * cov_vs.w;

    return { xyz_vs, rgb, cov_vs.xyz, float4(inv_cov_vs, opac), det };
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
    bool model_transform_enabled) {
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
    float3x3 rotation = quat_to_rotmat(normalize(g.rotations));
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
    float2 mean2d = float2(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float3 projected = project_point(sigma_points[i], cam);
        if (!valid_gut_image_point(projected, cam)) {
            return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
        }
        image_points[i] = projected.xy;
        float weight = i == 0u ? weight_mean_0 : weight_sigma;
        mean2d += weight * image_points[i];
    }

    float3 cov2d = float3(0.0);
    for (uint i = 0u; i < NUM_SIGMA_POINTS; ++i) {
        float2 d = image_points[i] - mean2d;
        float weight = i == 0u ? weight_cov_0 : weight_sigma;
        cov2d += weight * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    }

    float4 cov_vs = add_blur_to_covariance2d(cov2d);
    float det = compute_det(cov_vs.xyz);
    if (!isfinite(det) || det <= 0.0f) {
        return { float3(0.0), float3(0.0), float3(0.0), float4(0.0), 0.0 };
    }

    float4x3 mat = transpose(cam.world_view_transform);
    float3 cam_pos = -mul(float3x3(mat), mat[3]);
    float3 dir = float3(normalize(mean_ws - cam_pos));
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


// #define EARLY_RETURN_ON_ZERO_GRAD 0
#define EARLY_RETURN_ON_ZERO_GRAD 0


#ifndef NO_MAIN

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

// Input buffers
layout(binding=0) StructuredBuffer<float> xyz_ws;  // 3
layout(binding=1) StructuredBuffer<float4> sh_coeffs;  // 16*3
layout(binding=2) StructuredBuffer<float4> rotations;  // 4
layout(binding=3) StructuredBuffer<float4> scales_opacs;  // 4

// Output buffers
layout(binding=4) RWStructuredBuffer<int> out_tiles_touched;
#if USE_EMULATED_INT64
layout(binding=5) RWStructuredBuffer<int32_t> out_rect_tile_space;
#else
layout(binding=5) RWStructuredBuffer<int64_t> out_rect_tile_space;
#endif
layout(binding=6) RWStructuredBuffer<int> out_radii;
layout(binding=7) RWStructuredBuffer<float2> out_xy_vs;
layout(binding=8) RWStructuredBuffer<float> out_depths;
layout(binding=9) RWStructuredBuffer<float4> out_inv_cov_vs;
layout(binding=10) RWStructuredBuffer<float> out_rgb;
layout(binding=11) RWStructuredBuffer<uint> out_overlay_flags;
layout(binding=12) ByteAddressBuffer transform_indices;
layout(binding=13) ByteAddressBuffer node_mask;
layout(binding=14) StructuredBuffer<float4> overlay_params;
layout(binding=15) StructuredBuffer<float4> model_transforms;

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
    const uint camera_model = base_camera_model(uniforms.camera_model);
    const float4x4 world_view_transform = (uniforms.world_view_transform);

    uint g_idx = dispatchThreadID.x;
    
    if (g_idx >= uniforms.num_splats)
        return;

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
        image_height, image_width);
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

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD
    uint overlay_flags = 0;
    bool active = true;
    float4 emphasis_flags = overlay_params[OVERLAY_PARAM_EMPHASIS_FLAGS];
    bool node_data_enabled = overlay_enabled(emphasis_flags.y);
    int node_count = overlay_int(emphasis_flags.z);
    int node_idx = node_data_enabled ? read_int32(transform_indices, g_idx) : -1;
    int model_transform_idx = node_data_enabled ? node_idx : 0;
    model_transform_idx = clamp(model_transform_idx, 0, int(max(model_transform_count, 1u)) - 1);
    uint model_transform_base = uint(model_transform_idx) * 4u;
    bool model_transform_enabled = model_transform_count > 0u;
    float3 mean_ws = model_transform_enabled
                         ? transform_point_rows(model_transforms, model_transform_base, gauss.xyz_ws)
                         : gauss.xyz_ws;

    float4 crop_flags = overlay_params[OVERLAY_PARAM_CROP_FLAGS];
    if (overlay_enabled(crop_flags.x)) {
        int parent_node_idx = overlay_int(crop_flags.w);
        bool applies = parent_node_idx < 0 || (node_data_enabled && node_idx == parent_node_idx);
        if (applies) {
            float3 local = transform_point_rows(overlay_params, OVERLAY_PARAM_CROP_TRANSFORM, mean_ws);
            float3 bmin = overlay_params[OVERLAY_PARAM_CROP_MIN].xyz;
            float3 bmax = overlay_params[OVERLAY_PARAM_CROP_MAX].xyz;
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
    }

    float4 ellipsoid_flags = overlay_params[OVERLAY_PARAM_ELLIPSOID_FLAGS];
    if (active && overlay_enabled(ellipsoid_flags.x)) {
        int parent_node_idx = overlay_int(ellipsoid_flags.w);
        bool applies = parent_node_idx < 0 || (node_data_enabled && node_idx == parent_node_idx);
        if (applies) {
            float3 local = transform_point_rows(overlay_params, OVERLAY_PARAM_ELLIPSOID_TRANSFORM, mean_ws);
            float3 radii = overlay_params[OVERLAY_PARAM_ELLIPSOID_RADII].xyz;
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

    if (!active) {
        out_overlay_flags[g_idx] = overlay_flags;
        out_radii[g_idx] = 0;
        out_tiles_touched[g_idx] = 0;
      #if USE_EMULATED_INT64
        out_rect_tile_space[2*g_idx+0] = 0;
        out_rect_tile_space[2*g_idx+1] = 0;
      #else
        out_rect_tile_space[g_idx] = int64_t(0);
      #endif
        return;
    }
#endif

    // projection (differentiable)
  #if EXPORT_MODE != EXPORT_MODE_VULKAN_BACKWARD
    // forward
  #if VKSPLAT_GUT_PROJECTION
    Splat_2D_Vertex splat = project_gaussian_to_camera_gut(
        gauss,
        cam,
        active_sh,
        mean_ws,
        model_transforms,
        model_transform_base,
        model_transform_enabled);
  #else
    Splat_2D_Vertex splat = project_gaussian_to_camera_model(
        gauss,
        cam,
        active_sh,
        mean_ws,
        model_transforms,
        model_transform_base,
        model_transform_enabled);
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

    if (splat.xyz_vs.z <= NEAR_CLIP ||
        splat.inv_cov_vs_opac.w <= ALPHA_THRESHOLD ||
        splat.det <= 0.0
    ) {
        #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD
            out_overlay_flags[g_idx] = overlay_flags;
            out_radii[g_idx] = 0;
            out_tiles_touched[g_idx] = 0;
          #if USE_EMULATED_INT64
            out_rect_tile_space[2*g_idx+0] = 0;
            out_rect_tile_space[2*g_idx+1] = 0;
          #else
            out_rect_tile_space[g_idx] = int64_t(0);
          #endif
        #endif
        return;
    }

    Splat_2D_Vertex splat_dist = splat;

    float radius = splat_radius(splat_dist.cov_vs.xyz, splat_dist.det);

  #if TILE_CULLING_MODE == 0
    float2 radii = float2(radius);
  #else
    float extend = no_diff extend_from_opacity(gauss.opacities);
    float2 radii = sqrt(splat_dist.cov_vs.xz) * extend;
  #endif

    rectangle rect_tile_space = get_rectangle_tile_space(
        splat_dist.xyz_vs.xy, radii, grid_height, grid_width);

    int32_t n_tiles = int32_t(rect_tile_space.max_x - rect_tile_space.min_x) * int32_t(rect_tile_space.max_y - rect_tile_space.min_y);
    if (n_tiles == 0) {
        #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD
            out_overlay_flags[g_idx] = overlay_flags;
            out_radii[g_idx] = 0;
            out_tiles_touched[g_idx] = 0;
          #if USE_EMULATED_INT64
            out_rect_tile_space[2*g_idx+0] = 0;
            out_rect_tile_space[2*g_idx+1] = 0;
          #else
            out_rect_tile_space[g_idx] = int64_t(0);
          #endif
        #endif
        return;
    }

  #if TILE_CULLING_MODE == 2
    n_tiles = no_diff count_ellipse_grid_overlaps(
        splat_dist.xyz_vs.xy, splat_dist.inv_cov_vs_opac.xyz/(extend*extend),
        rect_tile_space
    );
  #endif

    // out_radii[g_idx] = (uint32_t)radius;  // inria
    out_radii[g_idx] = (uint32_t)ceil(max(radii.x, radii.y));  // gsplat, only used in densification
    out_tiles_touched[g_idx] = n_tiles;
  #if USE_EMULATED_INT64
    out_rect_tile_space[2*g_idx+0] = (int32_t)rect_tile_space.min_x | ((int32_t)rect_tile_space.min_y << 16);
    out_rect_tile_space[2*g_idx+1] = (int32_t)rect_tile_space.max_x | ((int32_t)rect_tile_space.max_y << 16);
  #else
    out_rect_tile_space[g_idx] = reinterpret<int64_t>(rect_tile_space);
  #endif

    if (n_tiles == 0) {
        out_overlay_flags[g_idx] = overlay_flags;
        return;
    }

#endif  // EXPORT_MODE != EXPORT_MODE_VULKAN_BACKWARD

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

    out_xy_vs[g_idx] = splat.xyz_vs.xy;
    out_depths[g_idx] = splat.xyz_vs.z;
    out_inv_cov_vs[g_idx] = splat.inv_cov_vs_opac;
    write_t3_float3(splat.rgb, g_idx, out_rgb);
    out_overlay_flags[g_idx] = overlay_flags;

#elif EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD

    write_v_xyz_ws(g_idx, v_gauss.d.xyz_ws);
    write_spherical_harmonics_coeffs_grads(
      v_gauss.d.sh_coeffs, g_idx, v_sh_coeffs, active_sh, step);
    v_rotations[g_idx] = v_gauss.d.rotations;
    v_scales_opacs[g_idx] = float4(v_gauss.d.scales, v_gauss.d.opacities);

#endif
}

#endif  // NO_MAIN
