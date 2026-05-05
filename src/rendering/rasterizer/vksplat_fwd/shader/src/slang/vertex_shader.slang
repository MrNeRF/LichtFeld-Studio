#include "config.slang"

import utils;
import spherical_harmonics;


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
    const uint step = uniforms.step;
    const uint image_height = uniforms.image_height;
    const uint image_width = uniforms.image_width;
    const uint grid_height = uniforms.grid_height;
    const uint grid_width = uniforms.grid_width;
    const uint camera_model = uniforms.camera_model;
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

    // projection (differentiable)
  #if EXPORT_MODE != EXPORT_MODE_VULKAN_BACKWARD
    // forward
    Splat_2D_Vertex splat = project_gaussian_to_camera(gauss, cam, active_sh);
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

    if (n_tiles == 0)
        return;

#endif  // EXPORT_MODE != EXPORT_MODE_VULKAN_BACKWARD

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD

    out_xy_vs[g_idx] = splat.xyz_vs.xy;
    out_depths[g_idx] = splat.xyz_vs.z;
    out_inv_cov_vs[g_idx] = splat.inv_cov_vs_opac;
    write_t3_float3(splat.rgb, g_idx, out_rgb);

#elif EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD

    write_v_xyz_ws(g_idx, v_gauss.d.xyz_ws);
    write_spherical_harmonics_coeffs_grads(
      v_gauss.d.sh_coeffs, g_idx, v_sh_coeffs, active_sh, step);
    v_rotations[g_idx] = v_gauss.d.rotations;
    v_scales_opacs[g_idx] = float4(v_gauss.d.scales, v_gauss.d.opacities);

#endif
}

#endif  // NO_MAIN
