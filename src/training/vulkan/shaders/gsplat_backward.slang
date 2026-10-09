// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "gsplat_common.slangh"
struct GsplatRasterParams {
    uint64_t camera;
    Ptr<float, Access.ReadWrite> means, scales, quats, opacities, colors;
    Ptr<float, Access.ReadWrite> bg_color, bg_image;
    Ptr<int, Access.ReadWrite> tile_offsets;
    Ptr<uint, Access.ReadWrite> gaussian_ids;
    Ptr<float, Access.ReadWrite> alpha;
    Ptr<int, Access.ReadWrite> last_ids;
    Ptr<float, Access.ReadWrite> v_image, v_alpha, grads, densification, error_map, edge_map, edge_scores;
    Ptr<float, Access.ReadWrite> v_depth;
    Ptr<uint, Access.ReadWrite> depths;
    Ptr<float, Access.ReadWrite> rendered_depth;
    uint count, width, height, tiles_x, tiles_y, edge_factor, mode;
    Ptr<float, Access.ReadWrite> v_normal;
};
float gsplat_background(GsplatRasterParams p, uint c, uint pixel) {
    return p.bg_image != 0 ? p.bg_image[c * p.width * p.height + pixel] : p.bg_color != 0 ? p.bg_color[c]
                                                                                          : 0;
}
void gsplat_atomic_add(Ptr<float, Access.ReadWrite> output, uint index, float value) {
#if GSPLAT_NATIVE_ATOMIC
    float previous;
    InterlockedAdd(output[index], value, previous);
#else
    Ptr<uint, Access.ReadWrite> words = (Ptr<uint, Access.ReadWrite>)uint64_t(output);
    uint observed = words[index];
    for (;;) {
        uint assumed = observed;
        InterlockedCompareExchange(words[index], assumed, asuint(asfloat(assumed) + value), observed);
        if (observed == assumed)
            break;
    }
#endif
}
// One pixel of the backward walk.
struct GsplatPixel {
    float3 origin, dir, v_c, buffer, depth_axis;
    float T, T_final, v_a, bg_accum, error, edge_weight;
    float v_depth, depth_buffer;
    int bin_final;
    bool active, radial_depth;
    float3x3 camera_R;
    float3 v_normal, normal_buffer;
};

static GsplatPixel gsplat_pixel(GsplatRasterParams p, const uint i, const uint j) {
    const uint plane = p.width * p.height;
    const uint pixel = min(i * p.width + j, plane - 1u);
    const GsplatCamera camera = gsplat_camera_load((Ptr<float, Access.ReadWrite>)p.camera);
    const GsplatRay ray = gsplat_pixel_ray(camera, float2(float(j) + 0.5f, float(i) + 0.5f));
    GsplatPixel px;
    px.camera_R=camera.R;
    px.radial_depth=camera.model!=kGsplatPinhole;
    px.v_normal=p.v_normal!=0?float3(p.v_normal[pixel],p.v_normal[plane+pixel],p.v_normal[2*plane+pixel]):float3(0.f);
    px.normal_buffer=0;
    px.origin = ray.origin;
    px.dir = ray.dir;
    px.active = i < p.height && j < p.width && ray.valid;
    px.T_final = 1.0f - p.alpha[pixel];
    px.T = px.T_final;
    px.bin_final = px.active ? p.last_ids[pixel] : 0;
    px.v_c = float3(p.v_image[pixel], p.v_image[plane + pixel], p.v_image[2u * plane + pixel]);
    px.v_a = p.v_alpha[pixel];
    px.bg_accum = 0.0f;
    if (p.bg_image != 0 || p.bg_color != 0) {
        for (uint c = 0; c < 3u; ++c)
            px.bg_accum += gsplat_background(p, c, pixel) * px.v_c[c];
    }
    px.error = p.error_map != 0 ? p.error_map[pixel] : 0.0f;
    px.edge_weight = p.edge_map != 0 ? p.edge_map[pixel] : 0.0f;
    px.buffer = float3(0.0f);
    px.v_depth = p.v_depth != 0 ? p.v_depth[pixel] : 0;
    px.depth_axis = float3(camera.R[0][2], camera.R[1][2], camera.R[2][2]);
    if (p.v_depth != 0 && (p.mode == 2 || p.mode == 4)) {
        px.v_depth /= max(p.alpha[pixel], 1e-10f);
        if (p.alpha[pixel] > 1e-10f)
            px.v_a -= px.v_depth * p.rendered_depth[pixel];
    }
    px.depth_buffer = 0;
    return px;
}

struct GsplatGrad {
    float3 rgb, mean, scale;
    float4 quat;
    float opacity, dens_w, dens_e, edge;
};

// Un-blends one gaussian from a pixel and adds its gradient; false when the
// gaussian did not contribute to the pixel.
static bool gsplat_contribute(inout GsplatPixel px, const int isect, const float4 xyzo, const float4 m0,
                              const float4 m1, const float4 m2, const float4 quat, const float3 rgb, const float depth, const bool have_bg,
                              inout GsplatGrad grad) {
    if (!px.active || isect > px.bin_final)
        return false;
    const float3x3 Mt = float3x3(m0.xyz, m1.xyz, m2.xyz);
    const float opac = xyzo.w;
    const float3 o_minus_mu = px.origin - xyzo.xyz;
    const float3 gro = mul(o_minus_mu, Mt);
    const float3 grd = mul(px.dir, Mt);
    const float3 grd_n = gsplat_safe_normalize(grd);
    const float3 gcrod = cross(grd_n, gro);
    const float power = -0.5f * dot(gcrod, gcrod);
    const float vis = exp(power);
    const float alpha = min(0.999f, opac * vis);
    if (power > 0.0f || alpha < kGsplatAlphaMin)
        return false;
    const float ra = 1.0f / (1.0f - alpha);
    px.T *= ra;
    const float fac = alpha * px.T;
    grad.rgb += fac * px.v_c;
    grad.mean += (fac * px.v_depth) * (px.radial_depth ? normalize(xyzo.xyz-px.origin) : px.depth_axis);
    uint axis=gsplat_normal_axis(float3(m0.w,m1.w,m2.w));
    float3 world_normal=gsplat_quat_to_rotmat(quat)[axis];
    float facing=dot(world_normal,xyzo.xyz-px.origin)>0?-1.f:1.f;
    float3 normal=mul(world_normal,px.camera_R)*facing;
    float3x3 v_R=float3x3(0.f);
    v_R[axis]=mul(px.v_normal,transpose(px.camera_R))*(fac*facing);
    gsplat_quat_to_rotmat_vjp(quat,v_R,grad.quat);
    grad.dens_w += fac;
    grad.dens_e += fac * px.error;
    const float edge_contribution = fac * px.edge_weight;
    if (px.edge_weight > 0.0f && isfinite(edge_contribution))
        grad.edge += edge_contribution;
    float v_alpha = dot(rgb * px.T - px.buffer * ra, px.v_c) + px.T_final * ra * px.v_a;
    v_alpha += (depth * px.T - px.depth_buffer * ra) * px.v_depth;
    v_alpha += dot(normal*px.T-px.normal_buffer*ra,px.v_normal);
    if (have_bg)
        v_alpha += -px.T_final * ra * px.bg_accum;
    if (opac * vis <= 0.999f) {
        const float v_gradDist = -0.5f * vis * (opac * v_alpha);
        const float3 v_gcrod = 2.0f * v_gradDist * gcrod;
        const float3 v_grd_n = -cross(v_gcrod, gro);
        const float3 v_gro = cross(v_gcrod, grd_n);
        const float3 v_grd = gsplat_safe_normalize_bw(grd, v_grd_n);
        const float3x3 v_Mt = gsplat_outer(v_grd, px.dir) + gsplat_outer(v_gro, o_minus_mu);
        grad.mean -= mul(v_gro, transpose(Mt));
        gsplat_preci_half_vjp(quat, float3(m0.w, m1.w, m2.w), gsplat_quat_to_rotmat(quat), transpose(v_Mt), grad.quat,
                              grad.scale);
        grad.opacity += vis * v_alpha;
    }
    px.buffer += rgb * fac;
    px.depth_buffer += depth * fac;
    px.normal_buffer += normal*fac;
    return true;
}

struct GsplatAccumulateParams {
    uint64_t camera;
    Ptr<float, Access.ReadWrite> means;
    Ptr<float, Access.ReadWrite> scales;
    Ptr<float, Access.ReadWrite> opacities;
    Ptr<float, Access.ReadWrite> grads;
    Ptr<int, Access.ReadWrite> radii;
    Ptr<float, Access.ReadWrite> means2d;
    Ptr<float, Access.ReadWrite> means_grad;
    Ptr<float, Access.ReadWrite> scaling_grad;
    Ptr<float, Access.ReadWrite> rotation_grad;
    Ptr<float, Access.ReadWrite> opacity_grad;
    Ptr<float, Access.ReadWrite> sh0_grad;
    Ptr<float, Access.ReadWrite> sh_rest_grad; // swizzled float, or null
    Ptr<float, Access.ReadWrite> grad_norms;   // densification row 0, or null
    Ptr<float, Access.ReadWrite> shares;       // [N] or null
    uint count;
    uint degree;
    uint layout_rest;
    uint width, height;
    float flatten_weight;
};

void gsplat_accumulate(GsplatAccumulateParams p, uint g) {
    if (g >= p.count)
        return;
    Ptr<float, Access.ReadWrite> v = (Ptr<float, Access.ReadWrite>)(uint64_t(p.grads) + uint64_t(g) * kGsplatGradStride * 4);
    const float3 scale = exp(gsplat_load3(p.scales, g));
    const float opacity = gsplat_sigmoid(p.opacities[g]);
    for (uint k = 0; k < 3u; ++k) {
        p.means_grad[g * 3u + k] += v[k];
        p.scaling_grad[g * 3u + k] += v[3u + k] * scale[k];
    }
    uint axis=gsplat_normal_axis(scale);
    p.scaling_grad[g*3+axis] += p.flatten_weight*scale[axis]/float(p.count);
    for (uint k = 0; k < 4u; ++k)
        p.rotation_grad[g * 4u + k] += v[6u + k];
    p.opacity_grad[g] += v[10] * opacity * (1.0f - opacity);

    float basis[16];
    gsplat_sh_basis(gsplat_load3(p.means, g) - gsplat_camera_load((Ptr<float, Access.ReadWrite>)p.camera).origin, p.degree, basis);
    const uint bases = (p.degree + 1u) * (p.degree + 1u);
    for (uint c = 0; c < 3u; ++c) {
        const float v_color = v[11u + c];
        p.sh0_grad[g * 3u + c] += basis[0] * v_color;
        for (uint k = 1; k < bases && p.sh_rest_grad != 0; ++k)
            p.sh_rest_grad[gsplat_sh_float_index(g, k - 1u, c, p.layout_rest)] += basis[k] * v_color;
    }

    if (p.grad_norms != 0)
        p.grad_norms[g] += sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const int rx = p.radii[g * 2u], ry = p.radii[g * 2u + 1u];
    if (p.shares != 0 && rx > 0 && ry > 0) {
        const float x = p.means2d[g * 2u], y = p.means2d[g * 2u + 1u];
        const float w = float(p.width), h = float(p.height);
        const float dx = max(0.0f, min(w, x + float(rx)) - max(0.0f, x - float(rx)));
        const float dy = max(0.0f, min(h, y + float(ry)) - max(0.0f, y - float(ry)));
        p.shares[g] = max(p.shares[g], (dx / w) * (dy / h));
    }
}

struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
[SpecializationConstant] const uint stage = 0;
// Stage one Gaussian per lane to amortize workgroup synchronization.
groupshared float4 xyzo[128], m0[128], m1[128], m2[128], quats[128], colors[128];
groupshared uint ids[128];
groupshared float reduction[17][128];
groupshared int last_active[128];
void raster(GsplatRasterParams p, uint tile, uint lane) {
    uint i = tile / p.tiles_x * 16 + lane / 8;
    uint j = tile % p.tiles_x * 16 + (lane % 8) * 2;
    GsplatPixel px0 = gsplat_pixel(p, i, j), px1 = gsplat_pixel(p, i, j + 1);
    bool bg = p.bg_color != 0 || p.bg_image != 0;
    int begin = p.tile_offsets[tile], end = p.tile_offsets[tile + 1];
    const int wave_last = WaveActiveMax(max(px0.bin_final, px1.bin_final));
    last_active[lane] = max(px0.bin_final, px1.bin_final);
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 64; step != 0; step >>= 1) {
        if (lane < step)
            last_active[lane] = max(last_active[lane], last_active[lane + step]);
        GroupMemoryBarrierWithGroupSync();
    }
    // Later intersections did not contribute to any pixel in this tile.
    end = min(end, last_active[0] + 1);
    for (int batch_end = end - 1; batch_end >= begin; batch_end -= 128) {
        GroupMemoryBarrierWithGroupSync();
        int index = batch_end - int(lane);
        if (lane < 128 && index >= begin) {
            uint g = p.gaussian_ids[index];
            float3 scale = exp(gsplat_load3(p.scales, g));
            float4 quat = gsplat_load4(p.quats, g);
            float3x3 M = gsplat_world_to_unit(quat, scale);
            xyzo[lane] = float4(gsplat_load3(p.means, g), gsplat_sigmoid(p.opacities[g]));
            m0[lane] = float4(M[0], scale.x);
            m1[lane] = float4(M[1], scale.y);
            m2[lane] = float4(M[2], scale.z);
            quats[lane] = quat;
            colors[lane] = float4(gsplat_load3(p.colors, g), asfloat(p.depths[g]));
            ids[lane] = g;
        }
        GroupMemoryBarrierWithGroupSync();
        // A subgroup can skip intersections beyond all of its pixels.
        // Other subgroup widths retain the shared-memory reduction barriers.
        const int first = WaveGetLaneCount() == 32 ? max(0, batch_end - wave_last) : 0;
        for (int t = first; t < min(128, batch_end + 1 - begin); ++t) {
            GsplatGrad grad = (GsplatGrad)0;
            bool hit0 = gsplat_contribute(px0, batch_end - t, xyzo[t], m0[t], m1[t], m2[t], quats[t], colors[t].xyz, colors[t].w, bg, grad);
            bool hit1 = gsplat_contribute(px1, batch_end - t, xyzo[t], m0[t], m1[t], m2[t], quats[t], colors[t].xyz, colors[t].w, bg, grad);
            float values[17] = {grad.mean.x, grad.mean.y, grad.mean.z, grad.scale.x, grad.scale.y, grad.scale.z,
                                grad.quat.x, grad.quat.y, grad.quat.z, grad.quat.w, grad.opacity, grad.rgb.x, grad.rgb.y, grad.rgb.z,
                                grad.dens_w, grad.dens_e, grad.edge};
            bool wave_hit = WaveActiveAnyTrue(hit0 || hit1);
            if (WaveGetLaneCount() == 32) {
                if (wave_hit) {
                    const float edge = p.edge_scores != 0 ? WaveActiveSum(values[16]) : 0;
                    // Reduce each field to its own lane using the CUDA XOR tree.
                    const uint wl = WaveGetLaneIndex();
                    [unroll] for (uint k = 0; k < 16; ++k)
                        values[k] += WaveReadLaneAt(values[k], wl ^ 16);
                    [unroll] for (uint stride = 8; stride > 0; stride >>= 1) {
                        [unroll] for (uint k = 0; k < stride; ++k) {
                            const float own = (wl & stride) != 0 ? values[k + stride] : values[k];
                            const float send = (wl & stride) != 0 ? values[k] : values[k + stride];
                            values[k] = own + WaveReadLaneAt(send, wl ^ stride);
                        }
                    }
                    const uint g = ids[t];
                    if (wl < 14 && values[0] != 0)
                        gsplat_atomic_add(p.grads, g * 14 + wl, values[0]);
                    if (wl >= 14 && wl < 16 && p.densification != 0 && p.error_map != 0)
                        gsplat_atomic_add(p.densification, (wl - 14) * p.count + g, values[0]);
                    if (wl == 16 && p.edge_scores != 0)
                        gsplat_atomic_add(p.edge_scores, g, edge * p.edge_factor);
                }
            } else {
                for (uint k = 0; k < 17; ++k)
                    reduction[k][lane] = values[k];
                GroupMemoryBarrierWithGroupSync();
                // Logical CUDA warps on devices with a different subgroup width.
                for (uint step = 16; step != 0; step >>= 1) {
                    if (lane % 32 < step)
                        for (uint k = 0; k < 17; ++k)
                            reduction[k][lane] += reduction[k][lane + step];
                    GroupMemoryBarrierWithGroupSync();
                }
                for (uint k = 0; k < 17; ++k)
                    values[k] = reduction[k][lane];
                GroupMemoryBarrierWithGroupSync();
            }
            if (WaveGetLaneCount() != 32 && lane % 32 == 0) {
                uint g = ids[t];
                [unroll] for (uint k = 0; k < 14; ++k) if (values[k] != 0)
                    gsplat_atomic_add(p.grads, g * 14 + k, values[k]);
                if (p.densification != 0 && p.error_map != 0) {
                    gsplat_atomic_add(p.densification, g, values[14]);
                    gsplat_atomic_add(p.densification, p.count + g, values[15]);
                }
                if (p.edge_scores != 0)
                    // The CUDA RGB dual-pixel path reduces the edge sum twice.
                    gsplat_atomic_add(p.edge_scores, g, values[16] * p.edge_factor);
            }
        }
    }
}
[numthreads(128, 1, 1)] void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint3 grid = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    if (stage == 1) {
        GsplatAccumulateParams p = ((Ptr<GsplatAccumulateParams, Access.Read>)push.parameters)[0];
        for (uint i = gid.x * 128 + lane; i < p.count; i += grid.x * 128)
            gsplat_accumulate(p, i);
    } else {
        GsplatRasterParams p = ((Ptr<GsplatRasterParams, Access.Read>)push.parameters)[0];
        for (uint tile = gid.x; tile < p.tiles_x * p.tiles_y; tile += grid.x)
            raster(p, tile, lane);
    }
}
