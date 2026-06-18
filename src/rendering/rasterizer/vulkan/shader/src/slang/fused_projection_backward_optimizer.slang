#define NO_MAIN
#include "vertex_shader.slang"
#undef NO_MAIN

import strategy_utils;


// Input buffers
layout(binding=0) RWStructuredBuffer<float> xyz_ws;  // 3
layout(binding=1) RWStructuredBuffer<float4> sh_coeffs;  // 16*3
layout(binding=2) RWStructuredBuffer<float4> rotations;  // 4
layout(binding=3) RWStructuredBuffer<float4> scales_opacs;  // 4

layout(binding=4) StructuredBuffer<int> tiles_touched;

layout(binding=5) StructuredBuffer<float2> v_output_xy_vs;
layout(binding=6) StructuredBuffer<float4> v_output_inv_cov_vs;
layout(binding=7) StructuredBuffer<float> v_output_rgb;

// Optimizer buffers
layout(binding=8) RWStructuredBuffer<float> g_xyz_ws;  // 2*3
layout(binding=9) RWStructuredBuffer<float4> g_sh_coeffs_1;  // 16*3
layout(binding=10) RWStructuredBuffer<float4> g_sh_coeffs_2;  // 16*3
layout(binding=11) RWStructuredBuffer<float4> g_rotations;  // 2*4
layout(binding=12) RWStructuredBuffer<float4> g_scales_opacs;  // 2*4


// Adam optimizer

typedef vector<float,1> float1;

// "standard" Adam with bias correction
__generic<let dim: int>
void optimizer_update(
    inout vector<float,dim> x, in vector<float,dim> v,
    inout vector<float,dim> g1, inout vector<float,dim> g2,
    float step, vector<float,dim> lr
) {
    static const float beta_1 = 0.9;
    static const float beta_2 = 0.999;
    static const float eps = 1e-15;

    const float g1_sc = 1.0 / (1.0 - pow(beta_1, step));
    const float g2_sc = 1.0 / (1.0 - pow(beta_2, step));
    g1 = lerp(g1, v, 1.0-beta_1);
    g2 = lerp(g2, v*v, 1.0-beta_2);
    vector<float,dim> g1_t = g1 * g1_sc;
    vector<float,dim> g2_t = g2 * g2_sc;
    x = x - lr * g1_t / (sqrt(g2_t) + eps);
}


struct ShaderUniforms {
    uint32_t step;
    uint32_t active_sh;
    uint32_t num_splats;
    uint32_t image_size;  // upper 16 bit for height, lower for width
    float fx;
    float fy;
    float cx;
    float cy;
    float4 dist_coeffs;
    float lr_means;
    float lr_quats;
    float lr_scales;
    float lr_opacities;
    float reg_scale;
    float reg_opacity;
    float lr_sh_dc;
    float lr_sh_rest;
    float4x4 world_view_transform;
};


#define GROUP_SIZE SUBGROUP_SIZE

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform ShaderUniforms uniforms
) {
    const uint active_sh = uniforms.active_sh & 0xff;
    const uint camera_model = uniforms.active_sh >> 8;
    const uint step = uniforms.step;
    const uint image_height = uniforms.image_size >> 16;
    const uint image_width = uniforms.image_size & ((1 << 16) - 1);
    const float4x4 world_view_transform = (uniforms.world_view_transform);

    uint g_idx = dispatchThreadID.x;
    
    if (g_idx >= uniforms.num_splats)
        return;

    bool is_zero_grad = (tiles_touched[g_idx] == 0);

    // load stuff
    Camera cam = load_camera(world_view_transform,
        uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy, camera_model, uniforms.dist_coeffs,
        image_height, image_width);
    Gaussian_3D gauss = {
        read_mean(g_idx, xyz_ws),
        read_spherical_harmonics_coeffs(g_idx, sh_coeffs, active_sh),
        rotations[g_idx],
        scales_opacs[g_idx].xyz,
        scales_opacs[g_idx].w
    };

    // backward
    float2 v_out_xy_vs = is_zero_grad ? float2(0) : v_output_xy_vs[g_idx];
    float4 v_out_inv_cov_vs_opac = is_zero_grad ? float4(0) :
      v_output_inv_cov_vs[g_idx];
    float v_out_depths = 0.0;
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

    // warn that v_gauss may be NaN when is_zero_grad is True
    float3 v_xyz_ws = is_zero_grad ? float3(0) : v_gauss.d.xyz_ws;
    float4 v_rotation = is_zero_grad ? float4(0) : v_gauss.d.rotations;
    float4 v_scale_opac = is_zero_grad ? float4(0) :
        float4(v_gauss.d.scales, v_gauss.d.opacities);

    // optimizers

    uint group_offset = g_idx % GROUP_SIZE;
    uint group_base = g_idx - group_offset;

    uint32_t g1_idx = 2*g_idx;
    uint32_t g2_idx = 2*g_idx+1;

    // means
    {
        float3 x = gauss.xyz_ws;
        float3 v = v_xyz_ws;
        float3 g1 = read_mean(g_idx, g_xyz_ws, 2, 0);
        float3 g2 = read_mean(g_idx, g_xyz_ws, 2, 1);
        optimizer_update(x, v, g1, g2, uniforms.step, uniforms.lr_means);
        write_mean(x, g_idx, xyz_ws);
        write_mean(g1, g_idx, g_xyz_ws, 2, 0);
        write_mean(g2, g_idx, g_xyz_ws, 2, 1);
    }

    // rotations (normalized)
    {
        float4 x = gauss.rotations;
        float4 v = v_rotation; // TODO: should ideally project when NORMALIZE_QUATS_IN_OPTIMIZER is True
        float4 g1 = g_rotations[g1_idx];
        float4 g2 = g_rotations[g2_idx];
        optimizer_update(x, v, g1, g2, uniforms.step, uniforms.lr_quats);
        rotations[g_idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER) ? normalize(x) : x;
        g_rotations[g1_idx] = g1;
        g_rotations[g2_idx] = g2;
    }

    // scales (exp) and opac (sigmoid)
    {
        // load
        float4 x = float4(gauss.scales, gauss.opacities);
        float4 v = float4(
            v_scale_opac.xyz + uniforms.reg_scale,
            v_scale_opac.w + uniforms.reg_opacity
        );
        float4 g1 = g_scales_opacs[g1_idx];
        float4 g2 = g_scales_opacs[g2_idx];

        // inverse map
        float4 u = float4(
            log(max(x.xyz, 1e-8)),  // scale: u = ln(x)
            log(max(x.w,1e-8) / max(1.0-x.w,1e-8))  // opacity: u = logit(x)
        );
        float4 dLdu = float4(
            v.xyz * x.xyz,  // scale: dL/du = dL/dx * dx/du = v * x
            v.w * x.w*(1.0-x.w)  // opacity: dL/du = dL/dx * dx/du = v * x(1-x)
        );

        // update
        float4 lr = float4(float3(uniforms.lr_scales), uniforms.lr_opacities);
        optimizer_update(u, dLdu, g1, g2, uniforms.step, lr);

        // map
        x.xyz = exp(u.xyz);  // scale: x = exp(u)
        x.w = 1.0 / (1.0+exp(-u.w));  // opacity: x = sigmoid(u)

        // save
        scales_opacs[g_idx] = x;
        g_scales_opacs[g1_idx] = g1;
        g_scales_opacs[g2_idx] = g2;
    }

    // SH

    uint reorder_offset = g_idx % SH_REORDER_SIZE;
    uint reorder_base = g_idx - reorder_offset;

    float lr0 = uniforms.lr_sh_dc;
    float lr1 = uniforms.lr_sh_rest;
    uint num_sh = get_num_sh(uniforms.active_sh);
    for (int di = 0; di < 12; di++) {
        uint cid = 12 * g_idx + di;
        uint bid = 12 * reorder_base + di * SH_REORDER_SIZE + reorder_offset;
        if (is_sh_skipped(uniforms.step, cid, num_sh))
            continue;

        float4 lr = (cid%12 == 0) ? float4(lr0, lr0, lr0, lr1) : float4(lr1);

        float4 x = get_sh_coeff_4(gauss.sh_coeffs, di);
        float4 v = is_zero_grad ? float4(0) :
            get_sh_coeff_4(v_gauss.d.sh_coeffs, di);
        // default optimizer (Adam)
        float4 g1 = g_sh_coeffs_1[bid];
        float4 g2 = g_sh_coeffs_2[bid];
        optimizer_update(x, v, g1, g2, uniforms.step, lr);
        sh_coeffs[bid] = x;
        g_sh_coeffs_1[bid] = g1;
        g_sh_coeffs_2[bid] = g2;
    }
}
